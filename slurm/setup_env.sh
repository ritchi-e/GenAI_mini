#!/bin/bash
# One-time environment setup. Run on the LOGIN node (needs internet):
#     bash slurm/setup_env.sh
# Creates .venv on top of the cluster's PyTorch module and installs the extra packages.
set -euo pipefail
cd "$(dirname "$0")/.."

module purge 2>/dev/null || true
module load python/3.10.pytorch

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Pin numpy/torch to the versions shipped by the module so pip never replaces them
python - <<'PY' > .constraints.txt
import importlib
for pkg in ("numpy", "torch", "torchvision"):
    try:
        print(f"{pkg}=={importlib.import_module(pkg).__version__.split('+')[0]}")
    except ImportError:
        pass
PY
echo "Pinned:"; cat .constraints.txt
python -m pip install -r requirements.txt -c .constraints.txt

# DISTS uses ImageNet VGG16 weights from torchvision: download them now (compute nodes may be offline)
python -c "from DISTS_pytorch import DISTS; DISTS(); print('DISTS weights cached')"

python -c "import torch, cv2, skimage, yaml; print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
echo "Environment ready. Activate later with:  module load python/3.10.pytorch && source .venv/bin/activate"
