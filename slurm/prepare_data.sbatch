#!/bin/bash
# CPU-only job: unpack/tile the downloaded archives and build the validation set (~20-40 min).
# Download first on the login node:  python prepare_data.py --download-only
#SBATCH --job-name=denoise-data
#SBATCH --partition=small           # <-- set to a partition name from `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
python prepare_data.py --root "${DATA_ROOT:-data}"
