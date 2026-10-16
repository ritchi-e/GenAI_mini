"""Evaluate a checkpoint with the competition metrics (PSNR, SSIM, DISTS on uint8 + FLOPs).

    python evaluate.py --weights weights/model.pth --data data/val [--save-dir results/val]

<data> must contain noisy/ and clean/ sub-folders with matching file names. A per-domain
breakdown is printed using the file-name prefix (natural_, medical_, lowlight_).
"""
import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from count_flops import measure
from inference import Denoiser, read_rgb_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/model.pth")
    ap.add_argument("--data", default="data/val")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    den = Denoiser(args.weights, args.device)
    try:
        from DISTS_pytorch import DISTS
        dists = DISTS().to(den.device).eval()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] DISTS unavailable: {e}")
        dists = None

    names = sorted(os.listdir(os.path.join(args.data, "noisy")))
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    rows, groups = [], defaultdict(list)
    for n in names:
        noisy = read_rgb_u8(os.path.join(args.data, "noisy", n))
        clean = read_rgb_u8(os.path.join(args.data, "clean", n))
        out = den(noisy)
        assert out.dtype == np.uint8 and out.shape == clean.shape
        r = {
            "psnr": peak_signal_noise_ratio(clean, out, data_range=255),
            "ssim": structural_similarity(clean, out, channel_axis=2, data_range=255),
            "psnr_in": peak_signal_noise_ratio(clean, noisy, data_range=255),
        }
        if dists is not None:
            with torch.no_grad():
                t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255).to(den.device)  # noqa: E731
                r["dists"] = float(dists(t(out), t(clean)))
        rows.append(r)
        groups[n.split("_")[0]].append(r)
        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, n), out[..., ::-1])

    def summary(rs):
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}

    print(f"\n{'group':<10} {'n':>4} {'PSNR_in':>8} {'PSNR':>8} {'SSIM':>7} {'DISTS':>7}")
    for g, rs in sorted(groups.items()) + [("ALL", rows)]:
        s = summary(rs)
        print(f"{g:<10} {len(rs):>4} {s['psnr_in']:>8.3f} {s['psnr']:>8.3f} {s['ssim']:>7.4f} "
              f"{s.get('dists', float('nan')):>7.4f}")
    print()
    for k, v in measure(den.model.cpu()).items():
        print(f"{k:>15}: {v:.4f}" if v is not None else f"{k:>15}: n/a")


if __name__ == "__main__":
    main()
