# src/configs/test_clip.py
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from ..utils import load_config

class CLIPTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, list_path, label_path):
        self.image_dir = image_dir
        self.image_paths = []
        self.labels = []
        
        with open(label_path, "r") as f:
            for line in f:
                _, label = line.strip().split(" ", 1)
                self.labels.append(label)
        
        self.all_classes = sorted(list(set(self.labels)))
        
        with open(list_path, "r") as f:
            for line in f:
                _, filename = line.strip().split(" ", 1)
                self.image_paths.append(os.path.join(image_dir, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return Image.open(self.image_paths[idx]).convert("RGB"), self.labels[idx]

def main(args):
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = f"{cfg.paths.checkpoints_dir}/clip_finetuned"
    print(f"[INFO] Loading Model: {model_path}")
    
    try:
        model = CLIPModel.from_pretrained(model_path, use_safetensors=True).to(device)
        processor = CLIPProcessor.from_pretrained(model_path)
    except:
        print("[ERROR] Model not found. Please run train_clip.py first.")
        return

    test_dataset = CLIPTestDataset(
        cfg.paths.images_dir,
        f"{cfg.paths.splits_dir}/{cfg.data.test_list}",
        f"{cfg.paths.splits_dir}/{cfg.data.variant_test}"
    )
    print(f"[INFO] Test Samples: {len(test_dataset)}")

    print("[INFO] Computing Text Embeddings...")
    text_queries = [f"A photo of a {cls} aircraft" for cls in test_dataset.all_classes]
    
    text_inputs = processor(text=text_queries, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    print("[INFO] Starting Test...")
    
    y_true = []
    y_pred = []
    
    top1, top5, top10 = 0, 0, 0
    total = len(test_dataset)
    
    for i in tqdm(range(total)):
        img, true_label = test_dataset[i]
        
        img_input = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feature = model.get_image_features(**img_input)
            img_feature /= img_feature.norm(p=2, dim=-1, keepdim=True)
        
        similarity = (100.0 * img_feature @ text_features.T).softmax(dim=-1)
        
        _, indices = torch.topk(similarity, k=10)
        pred_indices = indices[0].cpu().numpy()
        
        pred_labels = [test_dataset.all_classes[idx] for idx in pred_indices]
        
        # Metrics Calculation
        if true_label == pred_labels[0]: top1 += 1
        if true_label in pred_labels[:5]: top5 += 1
        if true_label in pred_labels[:10]: top10 += 1
        
        # F1 Score icin listeleri doldur
        y_true.append(true_label)
        y_pred.append(pred_labels[0]) # Top-1 tahmini baz aliyoruz

    # F1 Score Hesabi (Weighted: Sinif dengesizligini gozetir)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_score_sklearn = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "="*40)
    print(" TEST RESULTS (CLIP Text-to-Image)")
    print("="*40)
    print(f"Recall@1  (Accuracy) : %{top1/total*100:.2f}")
    print(f"Recall@5             : %{top5/total*100:.2f}")
    print(f"Recall@10            : %{top10/total*100:.2f}")
    print("-" * 40)
    print(f"F1 Score (Weighted)  : {f1:.4f}")
    print(f"Precision (Weighted) : {precision:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)