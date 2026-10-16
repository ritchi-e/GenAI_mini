"""Download and prepare all training / validation data.

Run this on the LOGIN node (compute nodes usually have no internet):
    python prepare_data.py --root data

Sources (all usable for research; licences in README.md):
  natural  : DIV2K train HR (800 high-quality 2K photos; 60 held out for validation)
  medical  : MedMNIST+ 224px  - PneumoniaMNIST (chest X-ray), OrganCMNIST (CT),
             BreastMNIST (ultrasound), RetinaMNIST (fundus)          [CC BY 4.0]
             Brain-Tumor-Classification MRI (Kaggle/HF mirror)        [MIT]
  lowlight : LOL dataset (normal-light 'high' images; darkened at train time)

Output layout:
  <root>/raw/...                      downloaded archives (can be deleted afterwards)
  <root>/train/natural/*.png          multi-scale 384px tiles
  <root>/train/medical/*.png
  <root>/train/lowlight/*.png
  <root>/val/{clean,noisy}/*.png      fixed 256x256 validation pairs (seeded degradations)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denoise.degradations import degrade, quantize_u8  # noqa: E402

SOURCES = {
    # Hugging Face mirror of the 800 DIV2K training HR images (much faster than the ETH server);
    # the official ETH URL is used automatically if the mirror fails.
    "DIV2K_train_HR.zip": ["https://huggingface.co/datasets/yangtao9009/DIV2K/resolve/main/DIV2K_train.zip",
                           "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"],
    # MedMNIST+ 224px: Hugging Face mirror first (fast), official Zenodo record as fallback
    **{f"{k}_224.npz": [f"https://huggingface.co/datasets/convergedmachine/Enhanced-MedMNIST/resolve/main/{k}_224.npz",
                        f"https://zenodo.org/records/10519652/files/{k}_224.npz?download=1"]
       for k in ("pneumoniamnist", "organcmnist", "breastmnist")},
    "retinamnist_224.npz": "https://zenodo.org/records/10519652/files/retinamnist_224.npz?download=1",
    "brain_mri_Training.zip": "https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification/resolve/main/Training.zip",
    "brain_mri_Testing.zip": "https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification/resolve/main/Testing.zip",
    "lol_dataset.zip": "https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip",
}
MEDMNIST_TRAIN_CAP = {"pneumoniamnist": 4000, "organcmnist": 4000, "breastmnist": 600, "retinamnist": 1200}


# ----------------------------------------------------------------------------- helpers
def download(urls, dst, attempts=5):
    """Resumable download; each URL is retried (resuming the partial file) before the next mirror."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  [skip] {os.path.basename(dst)} already downloaded")
        return
    urls = [urls] if isinstance(urls, str) else urls
    tmp = dst + ".part"
    for url in urls:
        for attempt in range(1, attempts + 1):
            print(f"  [get ] {url}" + (f" (attempt {attempt}, resuming)" if attempt > 1 else ""), flush=True)
            try:
                if shutil.which("curl"):
                    subprocess.run(["curl", "-L", "-sS", "--http1.1", "--fail", "--retry", "5",
                                    "--retry-all-errors", "-C", "-", "-o", tmp, url], check=True)
                else:
                    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
                os.replace(tmp, dst)
                return
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {e}", flush=True)
        if os.path.exists(tmp):   # partial file belongs to this mirror; start clean on the next one
            os.remove(tmp)
    raise RuntimeError(f"could not download {os.path.basename(dst)}")


def write_png(path, img_u8):
    if img_u8.ndim == 3:
        img_u8 = img_u8[..., ::-1]
    cv2.imwrite(path, img_u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else img[..., ::-1]  # RGB uint8


def tile(img, out_dir, stem, tile_size=384, step=320, scales=(1.0, 0.5, 0.25), min_side=256):
    """Save multi-scale crops. Downscaled versions mimic whole-scene 256x256 test images."""
    n = 0
    h0, w0 = img.shape[:2]
    for s in scales:
        im = img if s == 1.0 else cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        if min(h, w) < min_side:
            continue
        if h <= tile_size or w <= tile_size:
            write_png(os.path.join(out_dir, f"{stem}_s{int(s * 100)}_full.png"), im)
            n += 1
            continue
        ys = list(range(0, h - tile_size + 1, step)) + [h - tile_size]
        xs = list(range(0, w - tile_size + 1, step)) + [w - tile_size]
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                write_png(os.path.join(out_dir, f"{stem}_s{int(s * 100)}_{y}_{x}.png"),
                          im[y:y + tile_size, x:x + tile_size])
                n += 1
    return n


def to_256(img):
    """Resize short side to 256 and centre-crop: the competition's fixed 256x256 format."""
    h, w = img.shape[:2]
    s = 256 / min(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    im = cv2.resize(img, (max(256, round(w * s)), max(256, round(h * s))), interpolation=interp)
    h, w = im.shape[:2]
    y, x = (h - 256) // 2, (w - 256) // 2
    return im[y:y + 256, x:x + 256]


def gray3(img):
    return np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img


# ----------------------------------------------------------------------------- per-source
def prep_div2k(raw, train_dir, val_items, n_val=60):
    with zipfile.ZipFile(os.path.join(raw, "DIV2K_train_HR.zip")) as z:
        members = sorted(m for m in z.namelist() if m.lower().endswith(".png"))
        print(f"  DIV2K: {len(members)} images")
        n = 0
        for i, m in enumerate(members):
            img = decode(z.read(m))
            stem = "div2k_" + os.path.splitext(os.path.basename(m))[0]
            # the last n_val images (0741-0800) are held out for validation
            if i >= len(members) - n_val:
                val_items.append(("natural_" + stem, to_256(img), False))
            else:
                n += tile(img, train_dir, stem)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(members)}", flush=True)
        print(f"    -> {n} training tiles, {n_val} validation images")


def prep_medmnist(raw, train_dir, val_items):
    for key, cap in MEDMNIST_TRAIN_CAP.items():
        data = np.load(os.path.join(raw, f"{key}_224.npz"))
        rng = np.random.default_rng(0)
        if "train_images" in data.files:     # official Zenodo layout (train/val/test splits)
            tr = np.concatenate([data["train_images"], data["val_images"]])
            te = data["test_images"]
        else:                                # HF mirror layout: single 'images' array
            allim = data["images"]
            perm = rng.permutation(len(allim))
            te, tr = allim[np.sort(perm[:64])], allim[np.sort(perm[64:])]
        idx = rng.permutation(len(tr))[:cap]
        for i in idx:
            img = gray3(tr[i])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            write_png(os.path.join(train_dir, f"{key}_{i:06d}.png"), img)
        for j in rng.permutation(len(te))[:8]:
            img = cv2.resize(gray3(te[j]), (256, 256), interpolation=cv2.INTER_CUBIC)
            val_items.append((f"medical_{key}_{j:05d}", img, False))
        print(f"  {key}: {len(idx)} train, 8 val")


def prep_brain_mri(raw, train_dir, val_items):
    for name, is_val in (("brain_mri_Training.zip", False), ("brain_mri_Testing.zip", True)):
        with zipfile.ZipFile(os.path.join(raw, name)) as z:
            members = sorted(m for m in z.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png")))
            rng = np.random.default_rng(1)
            if is_val:
                members = [members[k] for k in rng.permutation(len(members))[:12]]
            n = 0
            for m in members:
                img = decode(z.read(m))
                if img is None:
                    continue
                stem = "brainmri_" + os.path.splitext(m.replace("/", "_").replace(" ", "_"))[0]
                if is_val:
                    val_items.append(("medical_" + stem, to_256(img), False))
                else:
                    write_png(os.path.join(train_dir, stem + ".png"), img)
                    n += 1
            print(f"  {name}: {n if not is_val else len(members)} {'val' if is_val else 'train'} images")


def prep_lol(raw, train_dir, val_items):
    with zipfile.ZipFile(os.path.join(raw, "lol_dataset.zip")) as z:
        members = sorted(m for m in z.namelist()
                         if m.lower().endswith(".png") and "/high/" in m.lower() and "__macosx" not in m.lower())
        n = 0
        for m in members:
            img = decode(z.read(m))
            if img is None:
                continue
            stem = "lol_" + os.path.splitext(m.replace("/", "_"))[0]
            if "eval" in m.lower():
                # darkened clean target, same as the train-time low-light simulation
                dark = np.power(to_256(img).astype(np.float32) / 255.0, 1.6) * 0.25
                val_items.append(("lowlight_" + stem, np.round(dark * 255).astype(np.uint8), False))
            else:
                n += tile(img, train_dir, stem, scales=(1.0, 0.6))
        print(f"  LOL: {n} train tiles")


def build_val(val_items, val_dir, seed=1234):
    for sub in ("clean", "noisy"):
        os.makedirs(os.path.join(val_dir, sub), exist_ok=True)
    rng = np.random.default_rng(seed)
    for name, img, _ in val_items:
        clean = quantize_u8(img.astype(np.float32) / 255.0)
        noisy = degrade(clean, rng, strength=1.0)
        # make sure every validation image is actually degraded
        while np.abs(noisy - clean).mean() < 1.0 / 255:
            noisy = degrade(clean, rng, strength=1.0)
        write_png(os.path.join(val_dir, "clean", name + ".png"), np.round(clean * 255).astype(np.uint8))
        write_png(os.path.join(val_dir, "noisy", name + ".png"), np.round(noisy * 255).astype(np.uint8))
    print(f"  validation set: {len(val_items)} pairs -> {val_dir}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["div2k", "medmnist", "brainmri", "lol"],
                    help="sources to skip")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    raw = os.path.join(args.root, "raw")
    os.makedirs(raw, exist_ok=True)
    groups = {"div2k": ["DIV2K_train_HR.zip"],
              "medmnist": [f"{k}_224.npz" for k in MEDMNIST_TRAIN_CAP],
              "brainmri": ["brain_mri_Training.zip", "brain_mri_Testing.zip"],
              "lol": ["lol_dataset.zip"]}

    print("== Downloading")
    for g, files in groups.items():
        if g in args.skip:
            continue
        for f in files:
            download(SOURCES[f], os.path.join(raw, f))
    if args.download_only:
        return

    print("== Preparing")
    dirs = {d: os.path.join(args.root, "train", d) for d in ("natural", "medical", "lowlight")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    val_items = []
    if "div2k" not in args.skip:
        prep_div2k(raw, dirs["natural"], val_items)
    if "medmnist" not in args.skip:
        prep_medmnist(raw, dirs["medical"], val_items)
    if "brainmri" not in args.skip:
        prep_brain_mri(raw, dirs["medical"], val_items)
    if "lol" not in args.skip:
        prep_lol(raw, dirs["lowlight"], val_items)

    val_dir = os.path.join(args.root, "val")
    if os.path.isdir(val_dir):
        shutil.rmtree(val_dir)
    build_val(val_items, val_dir)
    for k, d in dirs.items():
        print(f"  train/{k}: {len(glob.glob(os.path.join(d, '*.png')))} files")
    print("Done.")


if __name__ == "__main__":
    main()
