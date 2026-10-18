#!/bin/bash
# 2-minute GPU check of the whole pipeline (tiny model, 60 iterations). Run this before the real job.
#SBATCH --job-name=denoise-smoke
#SBATCH --partition=phd           # <-- set to a partition name from `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
rm -rf runs/debug
python train.py --config configs/debug.yaml --opt data.root="${DATA_ROOT:-data}/train" data.val_root="${DATA_ROOT:-data}/val"
python export.py --ckpt runs/debug/final.pth --out runs/debug/model.pth
python evaluate.py --weights runs/debug/model.pth --data "${DATA_ROOT:-data}/val"
echo "SMOKE TEST PASSED"
