import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ConvBlock(nn.Module):
    """Double conv with GroupNorm and time embedding injection."""
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


class MiniUNetFlow(nn.Module):
    """
    A minimal U-Net for flow matching on 28x28 grayscale images.

    Compared to MiniUNetCFG (week10/unet.py) the key differences are:
      1. No class conditioning — this model is unconditional.
      2. t is already in [0, 1] (continuous), so no division by num_steps.
      3. self.time_proj is stored as a proper nn.Module attribute instead of
         being instantiated inside forward() (which prevented it from learning).
      4. The model predicts a velocity field v_theta(x_t, t) rather than noise.

    Architecture:
        Encoder: 28x28 → 14x14 → 7x7
        Bottleneck: 7x7
        Decoder: 7x7 → 14x14 → 28x28  (with skip connections)
        Output: 1-channel velocity field, same spatial size as input
    """

    def __init__(self, in_channels: int = 1, time_emb_dim: int = 64):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time projection: [sin(pi*t), cos(pi*t)] -> time_emb_dim
        # Stored as a module attribute so its parameters are learned.
        self.time_proj = nn.Linear(2, time_emb_dim)

        # Encoder
        self.inc   = ConvBlock(in_channels, 16, time_emb_dim)
        self.down1 = ConvBlock(16, 32, time_emb_dim)
        self.down2 = ConvBlock(32, 64, time_emb_dim)

        # Decoder (with skip connections doubling channel count on input)
        self.up1 = ConvBlock(64 + 32, 32, time_emb_dim)
        self.up2 = ConvBlock(32 + 16, 16, time_emb_dim)

        # Output head
        self.outc = nn.Conv2d(16, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28) noisy image at time t
            t: (B,) continuous timestep in [0, 1]

        Returns:
            (B, 1, 28, 28) predicted velocity field
        """
        # Build sinusoidal time embedding — t is already in [0, 1]
        t_sin_cos = torch.stack(
            [torch.sin(t * math.pi), torch.cos(t * math.pi)], dim=-1
        )  # (B, 2)
        emb = self.time_proj(t_sin_cos)  # (B, time_emb_dim)

        # Encoder
        x1 = self.inc(x, emb)                      # (B, 16, 28, 28)
        x2 = self.down1(F.max_pool2d(x1, 2), emb)  # (B, 32, 14, 14)
        x3 = self.down2(F.max_pool2d(x2, 2), emb)  # (B, 64,  7,  7)

        # Decoder
        x = F.interpolate(x3, scale_factor=2, mode="nearest")      # (B, 64, 14, 14)
        x = self.up1(torch.cat([x, x2], dim=1), emb)               # (B, 32, 14, 14)

        x = F.interpolate(x, scale_factor=2, mode="nearest")       # (B, 32, 28, 28)
        x = self.up2(torch.cat([x, x1], dim=1), emb)               # (B, 16, 28, 28)

        return self.outc(x)                                         # (B,  1, 28, 28)
