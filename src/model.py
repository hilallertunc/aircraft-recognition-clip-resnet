import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),  # 128-d embedding
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)  # Cosine similarity için normalize et

class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet50", proj_dim=128):
        super().__init__()
        if backbone == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = m.fc.in_features  
            m.fc = nn.Identity()  # Son katmanı kaldırma
            self.backbone = m

        self.proj = ProjectionHead(feat_dim, 512, proj_dim)  

    def forward(self, x):
        feats = self.backbone(x)  
        return self.proj(feats)   
