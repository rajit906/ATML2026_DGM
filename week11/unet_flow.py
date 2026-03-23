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


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Multi-frequency sinusoidal embedding for continuous t in [0, 1].

    Uses `dim//2` log-spaced frequencies (identical in spirit to DDPM / Transformer
    positional encoding).  A single sin/cos pair (the old approach) gives only one
    frequency and cannot distinguish nearby noise levels; this gives dim//2 frequencies
    spanning a wide range, so the model sees a rich, unique fingerprint for every t.

    Args:
        t:          (B,) timesteps in [0, 1]
        dim:        embedding dimension (must be even)
        max_period: controls the minimum frequency

    Returns:
        (B, dim) sinusoidal features
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=t.dtype, device=t.device)
        / half
    )                                        # (half,) — log-spaced frequencies
    args = t[:, None] * freqs[None]          # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


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


class MiniUNetContinuous(nn.Module):
    """
    A minimal U-Net for continuous-time VP-SDE / PFODE on 28x28 grayscale images.

    Compared to MiniUNetCFG (week10/unet.py) the key differences are:
      1. No class conditioning — this model is unconditional.
      2. t is already in [0, 1] (continuous), so no division by num_steps.
      3. Uses a rich multi-frequency sinusoidal embedding (see sinusoidal_embedding)
         followed by a learned SiLU MLP, giving the model a unique fingerprint for
         every noise level.
      4. The model predicts noise ε rather than a velocity field.

    Architecture:
        Encoder: 28x28 → 14x14 → 7x7
        Bottleneck: 7x7
        Decoder: 7x7 → 14x14 → 28x28  (with skip connections)
        Output: 1-channel predicted noise ε, same spatial size as input
    """

    def __init__(self, in_channels: int = 1, time_emb_dim: int = 64):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding: sinusoidal features -> learned SiLU MLP -> time_emb_dim
        # The MLP lets the model learn which frequency combinations matter most.
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

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
            (B, 1, 28, 28) predicted noise ε
        """
        # Rich multi-frequency time embedding
        emb = self.time_embed(sinusoidal_embedding(t, self.time_emb_dim))  # (B, time_emb_dim)

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
