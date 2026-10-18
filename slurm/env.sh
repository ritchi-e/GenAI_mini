# Sourced by every Slurm job: loads the module and activates the venv.
module purge 2>/dev/null || true
module load python/3.10.pytorch
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
