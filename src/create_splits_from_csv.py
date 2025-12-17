
import argparse, os, pandas as pd

#  CSV'deki sütun adları
POSS_ID = ["image_id", "image", "id", "filename", "file", "name"]
POSS_VAR = ["variant", "family_variant", "label", "class", "target", "Labels"] 
POSS_FAM = ["family", "Classes"]                                        
POSS_MAN = ["manufacturer", "maker", "brand"]


def pick_col(df, candidates):
    
    for c in candidates:
        if c in df.columns: return c
    
    for col in df.columns:
        for c in candidates:
            if col.lower() == c.lower(): return col
    return None

def normalize_row(row, id_col, fname_col, images_dir):

    fn = str(row[id_col]).strip()
    
    if fname_col:
        fn = str(row[fname_col]).strip()
        iid = str(row[id_col]).strip()
    else:
        iid = os.path.splitext(fn)[0] 
        
    if iid is None:
        raise ValueError("image_id bulunamadı.")
        
    exists = os.path.exists(os.path.join(images_dir, fn))
    return iid, fn, exists

def write_list(list_path, rows):
    with open(list_path, "w", encoding="utf-8") as f:
        for iid, fn in rows:
            f.write(f"{iid} {fn}\n")

def write_labels(label_path, pairs):
    with open(label_path, "w", encoding="utf-8") as f:
        for iid, lab in pairs:
            if iid is not None and lab is not None:
                f.write(f"{iid} {lab}\n")

def process_split(csv_path, images_dir, out_dir, split_name):
    print(f"İşleniyor: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"HATA: {csv_path} dosyası okunamadı. Hata: {e}")
        return

   
    id_col  = pick_col(df, ["filename", "id"]) 
    fname_col = None 
    
    var_col = pick_col(df, POSS_VAR) 
    fam_col = pick_col(df, POSS_FAM) 
    man_col = pick_col(df, POSS_MAN)

    if id_col is None:
        print(f"HATA: {csv_path} dosyasında 'filename' veya 'id' sütunu bulunamadı. Atlanıyor.")
        return
    if var_col is None:
        print(f"UYARI: {csv_path} dosyasında 'variant' (Labels) sütunu bulunamadı.")
    if fam_col is None:
        print(f"UYARI: {csv_path} dosyasında 'family' (Classes) sütunu bulunamadı.")


    rows_for_list, rows_for_variant, rows_for_family, rows_for_man = [], [], [], []
    
    for _, row in df.iterrows():
        try:
            iid, fn, _ = normalize_row(row, id_col, fname_col, images_dir)
            rows_for_list.append((iid, fn))
            
            
            if var_col: rows_for_variant.append((iid, str(row[var_col]).strip()))
            if fam_col: rows_for_family.append((iid, str(row[fam_col]).strip()))
            if man_col: rows_for_man.append((iid, str(row[man_col]).strip()))
        except Exception as e:
            print(f"Satır işlenirken hata (atlandı): {row}, Hata: {e}")

    os.makedirs(out_dir, exist_ok=True)
    
    
    write_list(os.path.join(out_dir, f"images_{split_name}.txt"), rows_for_list)
    
    
    if rows_for_variant:
        print(f"-> 'variant' (varyant) etiketleri '{var_col}' sütunundan yazılıyor...")
        write_labels(os.path.join(out_dir, f"images_variant_{split_name}.txt"), rows_for_variant)
    if rows_for_family:
        print(f"-> 'family' (aile) etiketleri '{fam_col}' sütunundan yazılıyor...")
        write_labels(os.path.join(out_dir, f"images_family_{split_name}.txt"), rows_for_family)
    if rows_for_man:
        write_labels(os.path.join(out_dir, f"images_manufacturer_{split_name}.txt"), rows_for_man)
        
    print(f"[OK] {split_name}: {len(rows_for_list)} satır {out_dir} klasörüne yazıldı.")

def main():
    ap = argparse.ArgumentParser(description="CSV dosyalarından .txt split'leri oluşturur.")
    ap.add_argument("--train_csv", required=True, help="Eğitim verilerini içeren .csv dosyasının yolu.")
    ap.add_argument("--val_csv", required=True, help="Validasyon verilerini içeren .csv dosyasının yolu.")
    ap.add_argument("--test_csv", required=True, help="Test verilerini içeren .csv dosyasının yolu.")
    ap.add_argument("--images_dir", required=True, help="Tüm .jpg dosyalarının bulunduğu ana klasör.")
    ap.add_argument("--out_dir", default="data", help="Oluşturulan .txt dosyalarının kaydedileceği klasör.")
    args = ap.parse_args()

    
    process_split(args.train_csv, args.images_dir, args.out_dir, "train")
    process_split(args.val_csv,   args.images_dir, args.out_dir, "val")
    process_split(args.test_csv,  args.images_dir, args.out_dir, "test")
    print("\nTüm işlemler tamamlandı.")

if __name__ == "__main__":
    main()