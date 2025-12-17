import os, torch
from PIL import Image
from torch.utils.data import Dataset

class AircraftDataset(Dataset):
    def __init__(self, images_dir, list_path, label_path=None, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = []
        self.image_files = []
        self.labels = []

        #  Görüntü listesini oku (iid, filename)
        with open(list_path, "r") as f:
            for line in f:
                iid, fn = line.strip().split(" ", 1)
                self.image_ids.append(iid)
                self.image_files.append(os.path.join(images_dir, fn))
        
        iid_to_label = {}
        missing_labels = 0
        
        if label_path:
            #String etiketleri oku 
            with open(label_path, "r") as f:
                for line in f:
                    try:
                        iid, label = line.strip().split(" ", 1)
                        iid_to_label[iid] = label
                    except ValueError:
                        print(f"Hatalı satır (atlandı): {line.strip()}")
            
            #String etiketleri tamsayı ID'lere (örn: 42) donustur
            unique_labels = sorted(list(set(iid_to_label.values())))
            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            
            for iid in self.image_ids:
                string_label = iid_to_label.get(iid)
                if string_label in label_to_int:
                    self.labels.append(label_to_int[string_label])
                else:
                    self.labels.append(-1) # Eksik etiketler için -1 kullan
                    missing_labels += 1
        else:
            self.labels = [-1] * len(self.image_ids) # etiket yoksa
            missing_labels = len(self.image_ids)
        
        print(f"[DATASET] loaded: {len(self.image_ids)} samples (missing_labels={missing_labels}) from {os.path.basename(list_path)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        iid = self.image_ids[idx]
        y = self.labels[idx]
        try:
            im = Image.open(path).convert("RGB")
            if self.transform:
                im = self.transform(im)
            return im, iid, y
        except Exception as e:
            print(f"Hata: Görüntü yüklenemedi {path} (ID: {iid}). Hata: {e}")
            # Hatalı görüntü yerine boş bir tensör döndür
            return torch.zeros((3, 224, 224)), iid, y