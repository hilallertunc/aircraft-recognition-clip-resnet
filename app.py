import os
import sys
import torch
from flask import Flask, render_template, request, send_file
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

# proje yollari
sys.path.append('.')
from src.model import ImageEncoder
from src.utils import load_config

app = Flask(__name__)

# ayarlar
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = "src/configs/config.yaml"
DB_PATH = "outputs/index/database.pt"
CHECKPOINT_RESNET = "outputs/checkpoints/best.pt"
CHECKPOINT_CLIP = "outputs/checkpoints/clip_finetuned"

# modelleri yukle
print("[SİSTEM] Modeller yükleniyor, lütfen bekleyin...")
cfg = load_config(CONFIG_PATH)

# resnet50- gorsel arama
resnet = ImageEncoder(backbone=cfg.train.backbone, proj_dim=cfg.train.proj_dim).to(DEVICE)
if os.path.exists(CHECKPOINT_RESNET):
    resnet.load_state_dict(torch.load(CHECKPOINT_RESNET, map_location=DEVICE, weights_only=True))
resnet.eval()

# clip - metin arama
clip = CLIPModel.from_pretrained(CHECKPOINT_CLIP).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CHECKPOINT_CLIP)

#veritabani 
if os.path.exists(DB_PATH):
    database = torch.load(DB_PATH, map_location=DEVICE, weights_only=True)
    print("Sistem hazir!")
else:
    print("HATA: database.pt bulunamadı! Lütfen 'create_index.py' dosyasını çalıştırın.")
    database = None

# --- ARAMA MOTORU FONKSİYONU ---
# app.py içindeki search_engine fonksiyonunu bununla değiştir:

def search_engine(query, mode="text", top_k=10):
    if database is None: return []
    
    db_paths = database["paths"]
    db_labels = database["labels"] # İsimleri veritabanından al
    
    if mode == "image":
        T_val = transforms.Compose([
            transforms.Resize((cfg.train.img_size, cfg.train.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if query.mode != 'RGB': query = query.convert("RGB")
        img_tensor = T_val(query).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            query_emb = resnet(img_tensor)
            db_embs = database["resnet_embeddings"].to(DEVICE)
            
    elif mode == "text":
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            query_emb = clip.get_text_features(**inputs)
            query_emb /= query_emb.norm(p=2, dim=-1, keepdim=True)
            db_embs = database["clip_embeddings"].to(DEVICE)

    sims = torch.mm(query_emb, db_embs.T).squeeze()
    values, indices = torch.topk(sims, k=top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[i].item()
        
        # HTML'e sadece İsim ve Resim Yolunu gönderiyoruz. Skoru göndermiyoruz.
        results.append({
            "path": db_paths[idx], 
            "name": db_labels[idx] # Artık gerçek uçak ismi
        })
        
    return results

# backend

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    search_type = "text"
    query_display = ""

    if request.method == 'POST':
        # metin arama gelirse
        if 'text_query' in request.form and request.form['text_query']:
            text = request.form['text_query']
            results = search_engine(text, mode="text")
            search_type = "text"
            query_display = text
            
        # resim yukleme gelirse
        elif 'image_query' in request.files:
            file = request.files['image_query']
            if file.filename != '':
                img = Image.open(file.stream)
                results = search_engine(img, mode="image")
                search_type = "image"
                query_display = "Yüklenen Görsel"

    return render_template('index.html', results=results, search_type=search_type, query=query_display)


@app.route('/image/<path:filepath>')
def serve_image(filepath):
    return send_file(filepath)

if __name__ == '__main__':
    app.run(debug=True)