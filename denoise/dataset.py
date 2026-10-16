"""Training / validation datasets."""
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations import augment_content, degrade, quantize_u8

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(folder, "**", "*" + ext), recursive=True)
    return sorted(files)


def read_rgb(path):
    """Read any image as float32 RGB in [0,1] (grayscale is replicated to 3 channels)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[..., :3]
    img = img[..., ::-1]  # BGR -> RGB
    return np.ascontiguousarray(img, dtype=np.float32) / 255.0


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class DegradationDataset(Dataset):
    """Samples a domain by weight, a random tile, a random crop, then degrades it on the fly.

    `domains` maps a sub-folder of `root` to its sampling weight, e.g.
    {"natural": 0.6, "medical": 0.3, "lowlight": 0.1}. Missing/empty folders are skipped.
    """

    def __init__(self, root, domains, patch_size=128, length=10 ** 7, strength=1.0,
                 lowlight_p=0.15, gray_p=0.08):
        self.patch = patch_size
        self.length = length
        self.strength = strength
        self.lowlight_p, self.gray_p = lowlight_p, gray_p
        self.files, weights = [], []
        for name, w in domains.items():
            files = list_images(os.path.join(root, name))
            if files and w > 0:
                self.files.append(files)
                weights.append(w)
                print(f"[data] domain '{name}': {len(files)} images, weight {w}")
            else:
                print(f"[data] domain '{name}' skipped (no images or zero weight)")
        if not self.files:
            raise RuntimeError(f"No training images found under {root}. Run prepare_data.py first.")
        self.weights = np.asarray(weights, np.float64) / np.sum(weights)
        self._rng = None

    def __len__(self):
        return self.length

    def _get_rng(self):
        if self._rng is None:
            # torch seeds each worker differently (and differently per epoch / rank)
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 63))
        return self._rng

    def _random_crop(self, img, rng):
        h, w = img.shape[:2]
        p = self.patch
        if h < p or w < p:
            s = p / min(h, w)
            img = cv2.resize(img, (max(p, round(w * s)), max(p, round(h * s))), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        y, x = rng.integers(0, h - p + 1), rng.integers(0, w - p + 1)
        return img[y:y + p, x:x + p]

    def __getitem__(self, _):
        rng = self._get_rng()
        d = rng.choice(len(self.files), p=self.weights)
        files = self.files[d]
        clean = read_rgb(files[int(rng.integers(0, len(files)))])
        clean = self._random_crop(clean, rng)
        clean = augment_content(clean, rng, self.lowlight_p, self.gray_p)
        clean = quantize_u8(clean)
        noisy = degrade(clean, rng, self.strength)
        return to_tensor(noisy), to_tensor(clean)


class PairedFolder(Dataset):
    """Fixed validation pairs: <root>/noisy/*.png and <root>/clean/*.png with matching names."""

    def __init__(self, root, limit=None):
        self.noisy = list_images(os.path.join(root, "noisy"))
        if limit:
            self.noisy = self.noisy[:limit]
        self.clean = [os.path.join(root, "clean", os.path.basename(p)) for p in self.noisy]
        if not self.noisy:
            raise RuntimeError(f"No validation pairs in {root}")

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, i):
        return to_tensor(read_rgb(self.noisy[i])), to_tensor(read_rgb(self.clean[i])), os.path.basename(self.noisy[i])
