
import argparse, torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from ..utils import load_config, set_seed
from ..dataset import AircraftDataset
from ..sampler import ClassAwareSampler
from ..model import ImageEncoder
from ..losses import SupConLoss, MultiSimilarityLoss

# Veri Yukleyiciler

def get_loaders(cfg):
    # transformasyonlar
    T_tr = transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.train.img_size, scale=(0.6, 1.0), ratio=(3/4, 4/3)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    T_va = transforms.Compose([
        transforms.Resize((cfg.train.img_size, cfg.train.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # datasetler
    tr = AircraftDataset(
        cfg.paths.images_dir,
        f"{cfg.paths.splits_dir}/{cfg.data.train_list}",
        f"{cfg.paths.splits_dir}/{cfg.data.variant_train}",
        transform=T_tr
    )
    va = AircraftDataset(
        cfg.paths.images_dir,
        f"{cfg.paths.splits_dir}/{cfg.data.val_list}",
        f"{cfg.paths.splits_dir}/{cfg.data.variant_val}",
        transform=T_va
    )
    
    # Batch=32
    sampler = ClassAwareSampler(tr, classes_per_batch=8, samples_per_class=4) 

    dl_tr = DataLoader(tr, batch_sampler=sampler, num_workers=cfg.train.num_workers)
    dl_va = DataLoader(va, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    
    return dl_tr, dl_va


# 2) Embedding cikarici
@torch.inference_mode()
def embed(model, loader, device):
    model.eval()
    all_features, all_ids, all_y = [], [], []
    for im, iid, y in tqdm(loader, desc="Embedding", leave=False):
        im = im.to(device)
        feats = model(im)
        all_features.append(feats.cpu())
        all_ids.extend(iid)
        all_y.append(y)
    return (
        torch.cat(all_features),
        torch.cat(all_y),
        np.array(all_ids)
    )


# Recall@k Hesaplayici
@torch.inference_mode()
def recall_at_k(qF, qy, qi, gF, gy, gi, k_list=(1, 5, 10)):
    # qF: query features, gF: gallery features
    sim = qF @ gF.T
    
    # query'lerin gallery'deki kendi index'lerini bul
    qi_map = {iid: i for i, iid in enumerate(gi)}
    self_idx = [qi_map.get(iid, -1) for iid in qi]
    
    # kendisiyle eslesmesini engelle
    sim[torch.arange(len(qF)), self_idx] = -float("inf")

    # etiketlere gore eslesme matrisi
    match = (qy.unsqueeze(1) == gy.unsqueeze(0)).float()

   
    _, top_idx = torch.topk(sim, k=max(k_list), dim=1)

    top_match = torch.gather(match, 1, top_idx)

    res = {}
    for k in k_list:
        
        r = top_match[:, :k].any(dim=1).float().mean().item()
        res[k] = r
    return res


# ana egitim (fine-tuning)

def main(args):
    cfg = load_config(args.config)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    dl_tr, dl_va = get_loaders(cfg)

    model = ImageEncoder(cfg.train.backbone, cfg.train.proj_dim).to(device)
    
    if cfg.train.loss == "supcon":
        print("[INFO] Loss: SupConLoss")
        criterion = SupConLoss(temperature=0.07)
    elif cfg.train.loss == "ms_loss":
        # parametreler
        alpha = getattr(cfg.train, "ms_alpha", 2.0)
        beta = getattr(cfg.train, "ms_beta", 50.0)
        base = getattr(cfg.train, "ms_base", 0.5)
        print(f"[INFO] Loss: MultiSimilarityLoss (alpha={alpha}, beta={beta}, base={base})")
        criterion = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
    else:
        raise ValueError(f"Bilinmeyen loss: {cfg.train.loss}")
    
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    
    scheduler = None
    if hasattr(cfg, "scheduler"):
        sch_cfg = cfg.scheduler
        if sch_cfg.type == "step":
            print(f"[INFO] Scheduler: StepLR(step_size={sch_cfg.step_size}, gamma={sch_cfg.gamma})")
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, step_size=sch_cfg.step_size, gamma=sch_cfg.gamma
            )
        else:
            print("[INFO] No scheduler")
            
    # egitim dongusu
    best_r1 = 0.0
    for e in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0
        
        
        step = 0 
        pbar = tqdm(dl_tr, desc=f"Epoch {e}/{cfg.train.epochs}")
        for step, (im, _, y) in enumerate(pbar, 1): 
        
            if step >= cfg.train.steps_per_epoch:
                break
            
            im, y = im.to(device), y.to(device)
            
            opt.zero_grad()
            z = model(im)
            loss = criterion(z, y)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}")
        
        
        if step == 0:
            print("\n" + "="*50)
            print("HATA: Eğitim verisi yükleyicisi (DataLoader) boş.")
            print("Log'da 'missing_labels=3334' görüyorsanız,")
            print("Lütfen 'images_variant_train.txt' dosyasını kontrol edin.")
            print("Etiketler yüklenemiyor.")
            print("="*50 + "\n")
            return 
        
            
        avg_loss = total_loss / step 
        
        # Validation
        gF, gy, gi = embed(model, dl_va, device)
        recalls = recall_at_k(gF, gy, gi, gF, gy, gi, tuple(cfg.eval.recall_k))
        
        print(f"[E{e}] avg_loss={avg_loss:.4f} | Val Recall@{cfg.eval.recall_k}: {recalls}")
        
        # Scheduler
        if scheduler:
            print(f"[E{e}] lr={scheduler.get_last_lr()[0]:.6f}")
            scheduler.step()

        # Model kaydetme
        if recalls[1] > best_r1:
            best_r1 = recalls[1]
            torch.save(model.state_dict(), f"{cfg.paths.checkpoints_dir}/best.pt")
            print(" Best model kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)