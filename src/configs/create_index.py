import torch
import os
import argparse
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from tqdm import tqdm
import sys

#yol
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.model import ImageEncoder
from src.utils import load_config

def main(args):
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BILGI] Cihaz: {device}")

    # model yukle
    print("[1/3] Modeller Yükleniyor...")
    resnet = ImageEncoder(backbone=cfg.train.backbone, proj_dim=cfg.train.proj_dim).to(device)
    if os.path.exists(f"{cfg.paths.checkpoints_dir}/best.pt"):
        resnet.load_state_dict(torch.load(f"{cfg.paths.checkpoints_dir}/best.pt", map_location=device, weights_only=True))
        resnet.eval()
    
    clip_path = f"{cfg.paths.checkpoints_dir}/clip_finetuned"
    try:
        clip = CLIPModel.from_pretrained(clip_path).to(device)
        processor = CLIPProcessor.from_pretrained(clip_path)
    except:
        print(" HATA: CLIP modeli bulunamadı!")
        return

    T_val = transforms.Compose([
        transforms.Resize((cfg.train.img_size, cfg.train.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #isim haritasi
    print("[2/3] Uçak İsimleri CSV Dosyalarından Taranıyor...")
    filename_to_label = {}
    
    # Veri seti dosyaları 
    csv_files = ["train.csv", "val.csv", "test.csv"]
    
    # Olası sutun isimleri
    possible_id_cols = ['filename', 'image_id', 'id']
    possible_label_cols = ['variant', 'label', 'family', 'manufacturer']

    for csv_file in csv_files:
        
        paths_to_check = [
            csv_file, 
            os.path.join("data", csv_file),
            os.path.join(project_root, csv_file)
        ]
        
        found_path = None
        for p in paths_to_check:
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path:
            print(f"   -> Okunuyor: {found_path}")
            try:
                df = pd.read_csv(found_path)
                
                # dosya adi bul
                col_file = None
                for col in possible_id_cols:
                    if col in df.columns:
                        col_file = col
                        break
                if not col_file: col_file = df.columns[0] # Bulamazsa ilk sütunu al

                # ucak ismi
                col_label = None
                for col in possible_label_cols:
                    if col in df.columns:
                        col_label = col
                        break
                if not col_label: col_label = df.columns[1] # Bulamazsa ikinci sütunu al

                print(f"      Sütunlar bulundu -> Dosya: '{col_file}' | İsim: '{col_label}'")

                # Sözlüğe ekle
                for index, row in df.iterrows():
                    fname = str(row[col_file])
                    if not fname.endswith(".jpg"): fname += ".jpg"
                    
                    label_name = str(row[col_label])
                    filename_to_label[fname] = label_name
                    
            except Exception as e:
                print(f"      Hata oluştu: {e}")
        else:
            print(f" Uyarı: {csv_file} bulunamadı (Sorun değil, diğerlerine bakılıyor).")

    # veritabai olustur
    print(f"[3/3] İndeksleme Başlıyor... (Toplam {len(filename_to_label)} isim bulundu)")
    
    all_files = [f for f in os.listdir(cfg.paths.images_dir) if f.endswith(".jpg")]
    
    resnet_vectors = []
    clip_vectors = []
    valid_paths = []
    valid_labels = []

    with torch.no_grad():
        for filename in tqdm(all_files):
            full_path = os.path.join(cfg.paths.images_dir, filename)
            
            # Eğer ismini bulduysak veritabanına ekle
            if filename in filename_to_label:
                try:
                    pil_img = Image.open(full_path).convert("RGB")
                    
                    # ResNet
                    res_input = T_val(pil_img).unsqueeze(0).to(device)
                    resnet_vectors.append(resnet(res_input).cpu())
                    
                    # CLIP
                    clip_input = processor(images=pil_img, return_tensors="pt").to(device)
                    clip_emb = clip.get_image_features(**clip_input)
                    clip_emb /= clip_emb.norm(p=2, dim=-1, keepdim=True)
                    clip_vectors.append(clip_emb.cpu())
                    
                    # Kayıt
                    valid_paths.append(full_path)
                    valid_labels.append(filename_to_label[filename]) 
                    
                except:
                    pass

    # Veritabanı Paketleme
    db = {
        "paths": valid_paths,
        "labels": valid_labels,
        "resnet_embeddings": torch.cat(resnet_vectors) if resnet_vectors else torch.empty(0),
        "clip_embeddings": torch.cat(clip_vectors) if clip_vectors else torch.empty(0)
    }

    os.makedirs("outputs/index", exist_ok=True)
    torch.save(db, "outputs/index/database.pt")
    print(f"\n🎉 Veritabanı BAŞARIYLA GÜNCELLENDİ! Toplam {len(valid_paths)} uçak eklendi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=str, default="src/configs/config.yaml"); args = parser.parse_args(); main(args)