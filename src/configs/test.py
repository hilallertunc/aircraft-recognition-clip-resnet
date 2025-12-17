import argparse
from ..model import ImageEncoder
from ..utils import load_config, set_seed
from ..dataset import AircraftDataset


import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

# Veri Yukleme

def get_loader(cfg):
    T_va = transforms.Compose([
        transforms.Resize((cfg.train.img_size, cfg.train.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    te = AircraftDataset(
        cfg.paths.images_dir,
        f"{cfg.paths.splits_dir}/{cfg.data.test_list}",
        f"{cfg.paths.splits_dir}/{cfg.data.variant_test}",
        transform=T_va
    )
    dl_te = DataLoader(te, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    return dl_te


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

# Recall@k Hesaplayıcı
@torch.inference_mode()
def recall_at_k(qF, qy, qi, gF, gy, gi, k_list=(1, 5, 10)):
    # qF: query features, gF: gallery features
    sim = qF @ gF.T
    
    qi_map = {iid: i for i, iid in enumerate(gi)}
    self_idx = [qi_map.get(iid, -1) for iid in qi]
    
    # Kendisiyle eslesmeyi engelle 
    sim[torch.arange(len(qF)), self_idx] = -float("inf")

    # Etiketlere göre eslesme matrisi
    match = (qy.unsqueeze(1) == gy.unsqueeze(0)).float()

    # Puanları sıralama
    _, top_idx = torch.topk(sim, k=max(k_list), dim=1)

    # etiketleri toplama
    top_match = torch.gather(match, 1, top_idx)

    res = {}
    for k in k_list:
        # ilk k'da en az 1 eşleşme var mı?
        r = top_match[:, :k].any(dim=1).float().mean().item()
        res[k] = r
    return res

# ana Test
def main(args):
    cfg = load_config(args.config)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    dl_te = get_loader(cfg)

    model = ImageEncoder(cfg.train.backbone, cfg.train.proj_dim).to(device)
    model.load_state_dict(torch.load(f"{cfg.paths.checkpoints_dir}/best.pt", map_location=device, weights_only=True))
    
    gF, gy, gi = embed(model, dl_te, device)
    recalls = recall_at_k(gF, gy, gi, gF, gy, gi, tuple(cfg.eval.recall_k))
    
    print(f"Test Recall@{cfg.eval.recall_k}: {recalls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)