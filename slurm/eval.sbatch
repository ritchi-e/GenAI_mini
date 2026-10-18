#!/bin/bash
# Export the best checkpoint to weights/model.pth and evaluate it (PSNR / SSIM / DISTS / FLOPs).
#   sbatch --export=ALL,RUN=runs/small slurm/eval.sbatch
#SBATCH --job-name=denoise-eval
#SBATCH --partition=phd           # <-- set to a partition name from `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
RUN=${RUN:-runs/small}
CKPT=${CKPT:-$RUN/best.pth}
python export.py --ckpt "$CKPT" --out weights/model.pth --onnx weights/model.onnx
python evaluate.py --weights weights/model.pth --data "${DATA_ROOT:-data}/val" --save-dir results/val
