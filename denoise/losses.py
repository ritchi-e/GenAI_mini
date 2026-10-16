"""Training objective: Charbonnier (PSNR) + SSIM + frequency (sharpness) + DISTS (perceptual).

Each term targets one of the evaluation axes; weights live in the YAML config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x, y):
        return torch.sqrt((x - y) ** 2 + self.eps2).mean()


def _gauss_window(size=11, sigma=1.5, channels=3, device=None, dtype=None):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).to(dtype)
    return w.expand(channels, 1, size, size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM, Gaussian window (matches skimage's gaussian_weights=True setting closely)."""

    def __init__(self, window=11, sigma=1.5):
        super().__init__()
        self.window, self.sigma = window, sigma

    def forward(self, x, y):
        x, y = x.float(), y.float()
        c = x.shape[1]
        w = _gauss_window(self.window, self.sigma, c, x.device, x.dtype)
        mu_x = F.conv2d(x, w, groups=c)
        mu_y = F.conv2d(y, w, groups=c)
        sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
        syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
        sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
        return 1 - ssim.mean()


class FFTLoss(nn.Module):
    """L1 between amplitude/phase spectra (as real/imag); helps recover high frequencies after blur."""

    def forward(self, x, y):
        fx = torch.fft.rfft2(x.float(), norm="ortho")
        fy = torch.fft.rfft2(y.float(), norm="ortho")
        return (torch.view_as_real(fx) - torch.view_as_real(fy)).abs().mean()


class DISTSLoss(nn.Module):
    """Wrapper around the official DISTS_pytorch implementation (frozen VGG16 features)."""

    def __init__(self):
        super().__init__()
        from DISTS_pytorch import DISTS
        self.net = DISTS()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def train(self, mode=True):  # keep the metric network in eval mode
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x, y):
        return self.net(x.float().clamp(0, 1), y.float(), require_grad=True, batch_average=True)


class CombinedLoss(nn.Module):
    def __init__(self, w_charb=1.0, w_ssim=0.2, w_fft=0.05, w_dists=0.0):
        super().__init__()
        self.w = dict(charb=w_charb, ssim=w_ssim, fft=w_fft, dists=w_dists)
        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.fft = FFTLoss()
        self.dists = DISTSLoss() if w_dists > 0 else None

    def set_dists_weight(self, w):
        if w > 0 and self.dists is None:
            raise RuntimeError("DISTS loss was not initialised (w_dists=0 at construction)")
        self.w["dists"] = w

    def forward(self, pred, target):
        terms = {"charb": self.charb(pred, target)}
        if self.w["ssim"] > 0:
            terms["ssim"] = self.ssim(pred, target)
        if self.w["fft"] > 0:
            terms["fft"] = self.fft(pred, target)
        if self.dists is not None and self.w["dists"] > 0:
            terms["dists"] = self.dists(pred, target)
        total = sum(self.w[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}
