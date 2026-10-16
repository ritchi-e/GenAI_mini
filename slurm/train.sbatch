#!/bin/bash
# Main training job.
#   sbatch slurm/train.sbatch                                           # configs/small.yaml, 1 GPU
#   sbatch --export=ALL,CONFIG=configs/base.yaml slurm/train.sbatch     # other config
#   sbatch --gres=gpu:4 --cpus-per-task=32 slurm/train.sbatch           # multi-GPU (DDP via torchrun)
#   sbatch --export=ALL,EXTRA_OPTS="train.total_iters=100000" slurm/train.sbatch   # config overrides
# Training checkpoints to runs/<name>/last.pth and auto-resumes: if the job hits the
# wall-time limit, just submit the same command again.
#SBATCH --job-name=denoise-train
#SBATCH --partition=small           # <-- set to a partition name from `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # degradations are generated on CPU: more cores = faster
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300         # 5 min before the time limit -> checkpoint and exit cleanly
##SBATCH --time=24:00:00            # uncomment and set if your partition needs an explicit limit

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh

CONFIG=${CONFIG:-configs/small.yaml}
DATA_ROOT=${DATA_ROOT:-data}
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Job $SLURM_JOB_ID on $(hostname) | config=$CONFIG | GPUs=$NGPU | CPUs=$SLURM_CPUS_PER_TASK"
nvidia-smi

ARGS="--config $CONFIG --opt data.root=$DATA_ROOT/train data.val_root=$DATA_ROOT/val $EXTRA_OPTS"
if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" train.py $ARGS &
else
    python train.py $ARGS &
fi
PID=$!
# forward Slurm's warning signal to the trainer process(es); they save last.pth and exit
if [ "$NGPU" -gt 1 ]; then
    trap 'echo "caught USR1, forwarding to workers"; pkill -USR1 -P $PID' USR1   # torchrun's children
else
    trap 'echo "caught USR1, forwarding"; kill -USR1 $PID' USR1
fi
wait $PID; wait $PID
