"""Inference: corrupted 256x256x3 image(s) in -> denoised 256x256x3 uint8 image(s) out.

CLI:
    python inference.py --weights weights/model.pth --input noisy_dir_or_file --output out_dir

Python API:
    from inference import Denoiser
    den = Denoiser("weights/model.pth")
    clean_u8 = den(noisy_u8)          # (256,256,3) uint8 RGB -> (256,256,3) uint8 RGB
"""
import argparse
import os

import cv2
import numpy as np
import torch

from denoise.model import NAFUNet
from denoise.utils import get_device, load_weights, tensor_to_uint8, uint8_to_tensor

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def model_from_checkpoint(path, map_location="cpu"):
    """Build the model with the architecture stored in the checkpoint (falls back to default)."""
    ck = torch.load(path, map_location=map_location)
    arch = {}
    if isinstance(ck, dict):
        arch = ck.get("arch") or (ck.get("config") or {}).get("model") or {}
    arch = {k: v for k, v in arch.items() if k != "name"}
    model = NAFUNet(**arch)
    load_weights(model, path, map_location)
    return model.eval()


class Denoiser:
    def __init__(self, weights, device=None):
        self.device = torch.device(device) if device else get_device()
        self.model = model_from_checkpoint(weights).to(self.device)

    @torch.no_grad()
    def __call__(self, img_u8):
        """img_u8: (H,W,3) uint8 RGB (grayscale (H,W) also accepted). Returns (H,W,3) uint8 RGB."""
        img = np.asarray(img_u8)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.dtype != np.uint8:   # accept float input in [0,1] or [0,255]
            img = img.astype(np.float32)
            img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).round().astype(np.uint8)
        x = uint8_to_tensor(img[..., :3]).to(self.device)
        return tensor_to_uint8(self.model(x))   # clamp to [0,1] -> round -> uint8 [0,255]


def read_rgb_u8(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img[..., ::-1].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/model.pth")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    den = Denoiser(args.weights, args.device)
    if os.path.isdir(args.input):
        files = sorted(os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(IMG_EXT))
    else:
        files = [args.input]
    os.makedirs(args.output, exist_ok=True)
    for f in files:
        img = read_rgb_u8(f)
        if img.shape[:2] != (256, 256):
            print(f"[warn] {os.path.basename(f)} is {img.shape[1]}x{img.shape[0]}, not 256x256 (processed as-is)")
        out = den(img)
        name = os.path.splitext(os.path.basename(f))[0] + ".png"
        cv2.imwrite(os.path.join(args.output, name), out[..., ::-1])
    print(f"Denoised {len(files)} image(s) -> {args.output}")


if __name__ == "__main__":
    main()
