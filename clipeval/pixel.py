"""Per-clip pixel metrics.

Moved unchanged out of scripts/eval_video_metrics.py so the numbers stay comparable to
runs already recorded. Every function takes (N, H, W, 3) float32 tensors in [0, 255] and
returns a (N,) tensor of per-frame scores.
"""

import torch
import torch.nn.functional as F


def psnr(pred, target):
    """pred, target: (N, H, W, 3) float32 in [0, 255]. Returns (N,)."""
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))


def _gaussian_window(size=11, sigma=1.5, device='cpu', dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(pred, target, data_range=255.0):
    """pred, target: (N, H, W, 3) float32 in [0, 255]. Returns (N,)."""
    x = pred.permute(0, 3, 1, 2)
    y = target.permute(0, 3, 1, 2)
    n, c, _, _ = x.shape
    win = _gaussian_window(device=x.device, dtype=x.dtype).expand(c, 1, 11, 11)

    def filt(z):
        return F.conv2d(z, win, padding=0, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_x = filt(x * x) - mu_x2
    sigma_y = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).reshape(n, -1).mean(dim=1)


class Lpips:
    """Lazy wrapper so importing clipeval does not pull in the lpips package."""

    def __init__(self, device, net='alex', chunk=16):
        import lpips as _lpips
        self.fn = _lpips.LPIPS(net=net).to(device).eval()
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, pred, target):
        pn = pred.permute(0, 3, 1, 2) / 127.5 - 1
        rn = target.permute(0, 3, 1, 2) / 127.5 - 1
        return torch.cat([self.fn(pn[i:i + self.chunk], rn[i:i + self.chunk]).flatten()
                          for i in range(0, pn.shape[0], self.chunk)])
