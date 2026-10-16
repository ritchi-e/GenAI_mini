# Mini Competition: Small Generative Model for Image Denoising

A compact, FLOP-efficient **blind image restoration** model. It takes one corrupted
256×256×3 RGB image and returns the restored 256×256×3 **uint8** image. It is trained on
broadly randomized synthetic corruptions of natural, medical and low-light images, so it
generalizes to degradations it has not seen. The repo is set up to clone and run directly
on the institute's **DGX / DPU-HPC cluster (Slurm)**.

| | |
|---|---|
| Architecture | NAFNet-style 4-level U-Net (`denoise/model.py`), global residual |
| Default model (`configs/small.yaml`) | **4.45 GFLOPs** (fvcore) / 4.52 GMACs (ptflops) at 1×3×256×256, 6.5 M params |
| Other sizes | `tiny` 2.0 GFLOPs / 2.9 M · `base` 11.1 GFLOPs / 20.9 M |
| Loss | Charbonnier + 0.2·(1−SSIM) + 0.05·FFT, then a short DISTS fine-tune |
| Output | `uint8`, [0,255], always 256×256×3 (clamp → ×255 → round) |

---

## 1. Quick start on the DGX cluster (step by step)

All commands run from a terminal on your own machine or on the cluster login node.

### 1.1 Log in and clone

```bash
ssh <username>@172.25.1.11
```

```bash
git clone <this-repo-url> denoise-competition && cd denoise-competition
```

### 1.2 Set up the Python environment (login node, once)

```bash
bash slurm/setup_env.sh
```

This loads `module load python/3.10.pytorch`, creates `.venv` on top of it (it reuses the
module's PyTorch and never reinstalls it), installs `requirements.txt`, and caches the VGG16
weights that DISTS needs. Compute nodes are often offline, which is why this step runs on
the login node.

### 1.3 Download and prepare the data on the server (about 5.5 GB download, about 10 GB after tiling)

The datasets are **not** in the git repo. `prepare_data.py` downloads them straight onto the
cluster from fast Hugging Face mirrors, and falls back to the official servers (ETH, Zenodo) if a mirror fails.

Download on the **login node** (it needs internet):

```bash
module load python/3.10.pytorch && source .venv/bin/activate
```

```bash
python prepare_data.py --download-only
```

Then do the CPU-heavy unpacking and tiling, either directly:

```bash
python prepare_data.py
```

or as a batch job if heavy work on the login node is not allowed:

```bash
sbatch slurm/prepare_data.sbatch
```

Every step can be resumed. Finished downloads are skipped, and an interrupted download
continues where it stopped. If your home directory has a small quota, put the data on
scratch with `--root /path/to/scratch/data` and pass `DATA_ROOT=/path/to/scratch/data` to the jobs
(see 1.5).

> **No internet on the cluster at all?** Run `python prepare_data.py` on any other machine,
> then copy the prepared folders over:
> `rsync -av data/train data/val <username>@172.25.1.11:~/denoise-competition/data/`

### 1.4 Choose the partition

```bash
sinfo
```

Open each file in `slurm/*.sbatch` and replace `#SBATCH --partition=small` with a
partition from the `sinfo` output that has GPUs. If the partition needs an explicit
time limit, uncomment the `--time` line in `slurm/train.sbatch`.

### 1.5 Smoke test (about 2 minutes on the GPU), then train

```bash
sbatch slurm/smoke_test.sbatch
```

Check `logs/denoise-smoke-<jobid>.out`. It must end with `SMOKE TEST PASSED`. Then start the real training:

```bash
sbatch slurm/train.sbatch
```

Useful variations:

```bash
sbatch --export=ALL,CONFIG=configs/base.yaml slurm/train.sbatch
```

```bash
sbatch --gres=gpu:4 --cpus-per-task=32 slurm/train.sbatch
```

```bash
sbatch --export=ALL,DATA_ROOT=/scratch/$USER/data slurm/train.sbatch
```

With more than one GPU the job switches to DDP (`torchrun`) on its own. `batch` in the config
is the **global** batch size, split across the GPUs.

Monitor the job:

```bash
squeue -u $USER
```

```bash
tail -f logs/denoise-train-<jobid>.out
```

Cancel it with `scancel <jobid>`. Metrics are also written to `runs/small/log.csv`.

**Wall-time limits:** the trainer writes `runs/<name>/last.pth` every 2,000 iterations. Slurm
sends a signal 5 minutes before the limit, and the trainer then saves and exits cleanly. To
continue, **submit the same command again**. Training resumes from `last.pth` on its own.

### 1.6 Export and evaluate

```bash
sbatch --export=ALL,RUN=runs/small slurm/eval.sbatch
```

This writes `weights/model.pth`, the submission weights, plus `weights/model.onnx` if the
`onnx` package is installed. It then prints PSNR / SSIM / DISTS per domain and FLOPs on the
held-out validation set, and saves the restored images to `results/val/`.

---

## 2. Using the trained model (submission inference code)

CLI: process a folder or a single image.

```bash
python inference.py --weights weights/model.pth --input path/to/noisy --output path/to/out
```

Python API:

```python
from inference import Denoiser
den = Denoiser("weights/model.pth")      # picks CUDA / MPS / CPU automatically
clean = den(noisy)                       # noisy: (256,256,3) uint8 RGB -> clean: (256,256,3) uint8 RGB
```

* The input may be uint8, or float in [0,1] or [0,255]. Grayscale (H,W) inputs are replicated to 3 channels.
* The output is **always uint8 in [0,255]**: the model output is clamped to [0,1], multiplied by 255, rounded and cast.
* `weights/model.pth` is a plain `torch.save` dict `{"state_dict", "arch"}`. Load it with
  `NAFUNet(**ckpt["arch"]).load_state_dict(ckpt["state_dict"])`.

FLOPs, measured exactly as the organizers will (input `(1,3,256,256)`):

```bash
python count_flops.py --weights weights/model.pth
```

---

## 3. Method

### 3.1 Model: `denoise/model.py`
A 4-level U-Net built from **NAFBlocks** (Chen et al., *Simple Baselines for Image
Restoration*, ECCV 2022). Each block is LayerNorm → 1×1 conv → 3×3 depthwise conv →
SimpleGate (x₁·x₂) → simplified channel attention → 1×1 conv, followed by a gated FFN, with
learned residual scales. The blocks have no nonlinear activation functions, which gives a
very good quality-per-FLOP trade-off on denoising and deblurring benchmarks. Most blocks sit
at 1/8 resolution, where they are cheap and still cover the large receptive field that
blur removal needs. A global skip makes the network predict a residual correction. The
model uses only conv, elementwise and mean ops, so fvcore and ptflops count it fully.

### 3.2 Training data: `prepare_data.py`
| Domain (sampling weight) | Source | Licence |
|---|---|---|
| natural (0.6) | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) train HR, 800 2K photos (fast [HF mirror](https://huggingface.co/datasets/yangtao9009/DIV2K), official ETH URL as fallback) | research benchmark |
| medical (0.3) | [MedMNIST+](https://zenodo.org/records/10519652) 224 px: PneumoniaMNIST (chest X-ray), OrganCMNIST (CT), BreastMNIST (ultrasound), RetinaMNIST (fundus) | CC BY 4.0 |
| medical | [Brain Tumor MRI](https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification) (T1/T2 brain MRI slices) | MIT |
| low-light (0.1) | [LOL dataset](https://huggingface.co/datasets/geekyrakshit/LoL-Dataset) normal-light images | research benchmark |

Large photos are cut into 384 px tiles at scales 1, ½ and ¼. The downscaled tiles mimic
whole scenes resized to 256×256, which is what the test images most likely look like.
Medical images are resized to 256 px.

**Validation set** (`data/val/`, 119 pairs, fixed seed): 60 held-out DIV2K photos (0741–0800),
32 MedMNIST *test* images, 12 MRI *test* images, and 15 LOL-eval images that were darkened.
Each one has its own random degradation chain. None of these images are used for training.

### 3.3 Degradation model: `denoise/degradations.py` (on the fly, never the same twice)
For each training patch, a random chain is applied in image-formation order:

1. **Blur** (p 0.45): anisotropic Gaussian (σ 0.3–3.5), linear or curved **motion** blur (3–25 px), **defocus** disk (r 1–6)
2. **Chromatic aberration** (p 0.2): lateral R/B scaling and shift, plus optional single-channel blur
3. **Resolution loss** (p 0.12): down and up resizing ×1.3–4 with random interpolation
4. **Noise** (p 0.8; a second noise is mixed in with p 0.2), one of: **Gaussian** (σ 1–55, colour or luminance),
   **Poisson** (peak 5–500), **speckle**, **salt & pepper** (0.2–10 %), **Rician** (MRI),
   uniform, spatially correlated (film grain / demosaic), row/column **banding**
5. Quantization (p 0.05), **JPEG** q 5–95 (p 0.35), late impulse noise (p 0.06)
6. Rounding to the uint8 grid, exactly like the competition inputs

About 3 % of samples are left clean, so the model also learns not to over-smooth clean
inputs. Content augmentation: flips and 90° rotations, grayscale conversion (p 0.08) and
**low-light simulation** (p 0.15: the clean target is darkened with gamma 1–2.2 and gain
0.08–0.6 before corruption, which produces the high relative noise of dark photos). The
extra noise families beyond the brief's list, and the random combinations, target the
undisclosed held-out corruptions.

`docs/degradation_examples.png` shows 24 random draws on one image.

### 3.4 Objective and schedule (`train.py`, `denoise/losses.py`)
* `L = Charbonnier + 0.2·(1 − SSIM) + 0.05·FFT-L1 (+ 0.05·DISTS in the last stage)`.
  Each term targets one competition axis: PSNR, SSIM, sharpness after blur, and DISTS.
* AdamW (β = 0.9, 0.9), lr 1e-3 with 2k-iteration warmup and cosine decay to 1e-6, grad-clip 0.5,
  bf16 autocast, **EMA** of weights (0.999). The EMA weights are the ones exported.
* Progressive patches (`small`, 200k iterations): 128 px × batch 32 → 256 px × 16 (the true test size)
  → 256 px with the DISTS perceptual fine-tune.
* Every 5k iterations the EMA model is scored on the validation set (PSNR, SSIM, DISTS).
  `best.pth` keeps the checkpoint with the best balanced score `PSNR/40 + SSIM − DISTS`.

### 3.5 Choosing a model size
Ranking is per metric, so FLOPs count as much as quality. `small` (4.5 GFLOPs) is the
default balance. If there is enough GPU time, train `tiny` and `base` as well, compare them
with `evaluate.py`, and submit the one with the best overall trade-off. An overview of
NAFNet-class sizes: the original NAFNet-32 is about 16 GMACs, and `base` here is 11 GMACs.

---

## 4. Repository layout

```
denoise/
  model.py           NAFUNet architecture
  degradations.py    random degradation pipeline + content augmentation
  dataset.py         on-the-fly training dataset, paired validation dataset
  losses.py          Charbonnier, SSIM, FFT, DISTS losses
  utils.py           config loading, uint8 conversion, checkpoint loading
configs/             tiny / small (default) / base / debug
train.py             training (single GPU or DDP, AMP, EMA, auto-resume)
inference.py         submission inference (CLI + Denoiser class)
evaluate.py          PSNR / SSIM / DISTS (uint8) per domain + FLOPs
count_flops.py       fvcore + ptflops at (1,3,256,256)
export.py            best.pth -> weights/model.pth (+ optional ONNX)
prepare_data.py      download + tiling + fixed validation set
slurm/               setup_env.sh, env.sh, smoke_test / prepare_data / train / eval .sbatch
```

Override any config value from the command line with dotted keys, for example:

```bash
python train.py --config configs/small.yaml --opt train.total_iters=100000 data.num_workers=12
```

## 5. Troubleshooting
* **`module: command not found` inside a job**: add `source /etc/profile.d/modules.sh` at the top of `slurm/env.sh`.
* **Job pending for a long time**: use `squeue -u $USER` to see why, and try another partition from `sinfo`.
* **Training is slow (low it/s, GPU underused)**: degradations run on the CPU, so request more cores (`--cpus-per-task=32`). The number of loader workers follows that value on its own.
* **CUDA out of memory**: lower `batch` in the config's `stages`.
* **`DISTS` download error on a compute node**: run `bash slurm/setup_env.sh` again on the login node so the VGG16 weights are cached in `~/.cache/torch`.
* **Old PyTorch without bf16** (V100): AMP falls back to fp16 with a GradScaler on its own.
