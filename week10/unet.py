import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MPS is the backend for Apple Silicon GPUs (M1/M2/M3). If on Intel Mac, it falls back to CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

num_steps = 1000
betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
        )
        self.emb_layer = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, out_ch))

    def forward(self, x, t_emb):
        h = self.conv(x)
        emb = self.emb_layer(t_emb).view(-1, h.shape[1], 1, 1)
        return F.gelu(h + emb)

class MiniUNetCFG(nn.Module):
    def __init__(self, in_channels=1, time_emb_dim=64, num_classes=3): # classes: 0, 1, and 2 (null)
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Class embedding layer
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        self.inc = ConvBlock(in_channels, 16, time_emb_dim)
        self.down1 = ConvBlock(16, 32, time_emb_dim)
        self.down2 = ConvBlock(32, 64, time_emb_dim)
        
        self.up1 = ConvBlock(64 + 32, 32, time_emb_dim)
        self.up2 = ConvBlock(32 + 16, 16, time_emb_dim)
        self.outc = nn.Conv2d(16, in_channels, 1)

    def forward(self, x, t, c):
        # Time embedding (simple sine/cosine)
        t_norm = t.float() / num_steps
        t_emb = torch.cat([torch.sin(t_norm * 3.1415), torch.cos(t_norm * 3.1415)], dim=-1)
        t_emb = nn.Linear(2, self.time_emb_dim).to(x.device)(t_emb.view(-1, 2))
        
        # Add class embedding to time embedding
        c_emb = self.class_emb(c)
        emb = t_emb + c_emb

        x1 = self.inc(x, emb)                     # 28x28
        x2 = self.down1(F.max_pool2d(x1, 2), emb) # 14x14
        x3 = self.down2(F.max_pool2d(x2, 2), emb) # 7x7

        x = F.interpolate(x3, scale_factor=2, mode="nearest") # 14x14
        x = self.up1(torch.cat([x, x2], dim=1), emb)
        
        x = F.interpolate(x, scale_factor=2, mode="nearest") # 28x28
        x = self.up2(torch.cat([x, x1], dim=1), emb)
        
        return self.outc(x)