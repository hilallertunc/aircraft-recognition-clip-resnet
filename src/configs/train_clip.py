# src/configs/train_clip.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch.optim as optim
from PIL import Image

from ..utils import load_config, set_seed

# dataset
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, list_path, label_path):
        self.image_paths = []
        self.texts = []
        
        label_names = []
        with open(label_path, "r") as f:
            for line in f:
                try:
                    _, label = line.strip().split(" ", 1)
                    label_names.append(label)
                except: continue
        
        with open(list_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i >= len(label_names): break
                _, filename = line.strip().split(" ", 1)
                
                self.image_paths.append(os.path.join(image_dir, filename))
                self.texts.append(f"A photo of a {label_names[i]} aircraft")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        return image, text

def main(args):
    cfg = load_config(args.config)
    set_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # model
    model_id = "openai/clip-vit-base-patch32"
    print(f"[INFO] Loading CLIP Model: {model_id}")
    
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # katman dondura
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.visual_projection.parameters():
        param.requires_grad = True
    for param in model.text_projection.parameters():
        param.requires_grad = True

    # 2. vision encoder son blok acma
    for param in model.vision_model.encoder.layers[-1].parameters():
        param.requires_grad = True
    
    # parametre sayisi
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable Parameters: {trainable_params}")

    # dataset
    train_dataset = CLIPDataset(
        cfg.paths.images_dir,
        f"{cfg.paths.splits_dir}/{cfg.data.train_list}",
        f"{cfg.paths.splits_dir}/{cfg.data.variant_train}"
    )

    def collate_fn(batch):
        images = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        return processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)

    # Batch Size 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)

    #training

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, weight_decay=0.1)
    
    # Epoch
    epochs = 20
    
    # Scheduler(loss un stabil olması icin)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # training loop
    print(f"[INFO] Training Started for {epochs} Epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"[RESULT] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.8f}")
        
        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            save_path = f"{cfg.paths.checkpoints_dir}/clip_finetuned"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"[INFO] Model Saved: {save_path}")

    print("[INFO] Training Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)