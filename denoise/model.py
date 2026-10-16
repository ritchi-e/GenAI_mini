"""Compact NAFNet-style U-Net for blind image restoration.

Design choices (driven by the competition's 4 axes: FLOPs, PSNR, SSIM, DISTS):
  * NAFBlocks (Chen et al., "Simple Baselines for Image Restoration", ECCV'22):
    no nonlinear activations, SimpleGate + simplified channel attention. They give
    state-of-the-art PSNR/SSIM per FLOP on denoising and deblurring.
  * 4-level U-Net so most blocks run at 1/8 resolution (cheap) while keeping a
    large receptive field, which matters for blur (motion / defocus) removal.
  * Global residual: the network predicts a correction added to the input.
  * Only plain conv / mul / add / mean ops, so fvcore / ptflops count it exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel LayerNorm for NCHW tensors, written with basic ops (FLOP-counter friendly)."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = x * self.sca(x)
        y = inp + self.conv3(x) * self.beta

        x = self.sg(self.conv4(self.norm2(y)))
        return y + self.conv5(x) * self.gamma


class NAFUNet(nn.Module):
    def __init__(self, in_ch=3, width=24, enc_blocks=(1, 1, 2, 4), middle_blocks=4,
                 dec_blocks=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, in_ch, 3, padding=1)

        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad_to = 2 ** len(enc_blocks)

    def forward(self, inp):
        """inp: (B,3,H,W) float in [0,1]. Returns restored image (not clamped)."""
        _, _, h, w = inp.shape
        ph, pw = (-h) % self.pad_to, (-w) % self.pad_to
        x_in = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp

        x = self.intro(x_in)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = dec(up(x) + skip)
        x = self.ending(x) + x_in
        return x[:, :, :h, :w]


def build_model(cfg_model):
    cfg = dict(cfg_model)
    cfg.pop("name", None)
    return NAFUNet(**cfg)
