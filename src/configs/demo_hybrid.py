# src/configs/demo_hybrid.py
import torch
import os
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import torch.nn.functional as F

# Senin kendi model yapın
from ..model import ImageEncoder
from ..utils import load_config

def main(args):
    # 1. Ayarları Yükle
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[BILGI] Cihaz: {device}")

    
    # MODUL 1:  RESNET50 
   
    print("-" * 50)
    print("[1/3] ResNet50 (Gorsel-Gorsel) Modeli Yukleniyor...")
    
    
    resnet_model = ImageEncoder(backbone=cfg.train.backbone, proj_dim=cfg.train.proj_dim).to(device)
    
    # best.pt 
    ckpt_path = f"{cfg.paths.checkpoints_dir}/best.pt"
    try:
        resnet_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        resnet_model.eval() # Test moduna al
        print(f"[BASARILI] ResNet50 Yuklendi: {ckpt_path}")
    except Exception as e:
        print(f"[HATA] ResNet50 yuklenemedi! {e}")
        return

   
    # MODUL 2:  CLIP MODELI 
  
    print("-" * 50)
    print("[2/3] CLIP (Metin-Gorsel) Modeli Indiriliyor/Yukleniyor...")
    
   
    clip_id = "openai/clip-vit-base-patch32"
    try:
        
        clip_model = CLIPModel.from_pretrained(clip_id, use_safetensors=True).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_id)
        print(f"[BASARILI] CLIP Yuklendi: {clip_id}")
    except Exception as e:
        print(f"[HATA] CLIP yuklenemedi! 'pip install safetensors' yaptin mi? Hata: {e}")
       
        try:
             print("[UYARI] Safetensors basarisiz, standart yukleme deneniyor...")
             clip_model = CLIPModel.from_pretrained(clip_id).to(device)
             clip_processor = CLIPProcessor.from_pretrained(clip_id)
             print("[BASARILI] CLIP (Standart) Yuklendi.")
        except Exception as e2:
             print(f"[KRITIK HATA] {e2}")
             return

    
    # TEST 
    
    print("-" * 50)
    print("[3/3] Test Basliyor...")

    # rastgele 
    test_image_path = None
    try:
        with open(f"{cfg.paths.splits_dir}/{cfg.data.test_list}", "r") as f:
            line = f.readline() 
            if line:
                _, filename = line.strip().split(" ", 1)
                test_image_path = os.path.join(cfg.paths.images_dir, filename)
    except Exception as e:
        print(f"[UYARI] Test listesi okunamadi: {e}")
    
    if not test_image_path or not os.path.exists(test_image_path):
        print("[UYARI] Test resmi bulunamadi. Lutfen veri setini kontrol et.")
        return

    print(f"[BILGI] Ornek Resim: {test_image_path}")
    raw_image = Image.open(test_image_path).convert("RGB")

    # ResNet50
    print("\n--- SENARYO A: Kullanici Resim Yukledi (Gorsel Arama) ---")
    
   
    T_val = transforms.Compose([
        transforms.Resize((cfg.train.img_size, cfg.train.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = T_val(raw_image).unsqueeze(0).to(device) # Batch boyutu ekle
    
    with torch.no_grad():
        resnet_emb = resnet_model(img_tensor)
    
    print(f"-> ResNet50 Ciktisi: {resnet_emb.shape} boyutunda vektor.")
    print("[BILGI] Bu vektor, veritabanindaki diger ResNet vektorleriyle karsilastirilip EN BENZER ucak bulunur.")

    #CLIP 
    print("\n--- SENARYO B: Kullanici Yazi Yazdi (Metin Arama) ---")
    
    user_query = "A photo of a fighter jet" # Örnek sorgu
    print(f"[SORGU]: '{user_query}'")
    
    # 1. Metni Vektore cevir
    text_inputs = clip_processor(text=[user_query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True) # Normalize et
    
    # 2. Resmi  CLIP Vektorune cevir
    image_inputs = clip_processor(images=raw_image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    # 3. Benzerlik 
    similarity = (text_features @ image_features.T).item()
    
    print(f"-> CLIP Metin Vektoru: {text_features.shape}")
    print(f"-> CLIP Resim Vektoru: {image_features.shape}")
    print(f"[SONUC] Sorgu ile Resim Arasindaki Benzerlik Skoru: {similarity:.4f}")
    
    if similarity > 0.2:
        print("[SONUC] Eslesme Basarili! Bu resim sorguya benziyor.")
    else:
        print("[SONUC] Dusuk benzerlik.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)