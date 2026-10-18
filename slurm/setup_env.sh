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

# Pin numpy/torch to the versions shipped by the module so pip never replaces them.
# torchvision may fail to import if the module's build mismatches torch CUDA — skip pinning it then.
python - <<'PY' > .constraints.txt
import importlib
for pkg in ("numpy", "torch", "torchvision"):
    try:
        print(f"{pkg}=={importlib.import_module(pkg).__version__.split('+')[0]}")
    except Exception:
        pass
PY
echo "Pinned:"; cat .constraints.txt

# Cluster module sometimes ships torchvision built for a different CUDA major than torch.
# Install a matching wheel into the venv so it shadows the broken system package.
python - <<'PY'
import subprocess, sys
import torch
try:
    import torchvision  # noqa: F401
    print("system torchvision OK:", torchvision.__version__)
except Exception as e:
    print("system torchvision unusable:", e)
    ver = torch.__version__.split("+")[0]
    major, minor, *_ = ver.split(".")
    tv_ver = {"2.0": "0.15.2", "2.1": "0.16.2", "2.2": "0.17.2"}.get(f"{major}.{minor}", "0.16.2")
    tag = "cu" + "".join(torch.version.cuda.split(".")[:2]) if torch.version.cuda else "cpu"
    spec = f"torchvision=={tv_ver}+{tag}"
    print(f"Installing {spec} to match torch {torch.__version__}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", spec,
        "--index-url", f"https://download.pytorch.org/whl/{tag}",
    ])
PY

python -m pip install -r requirements.txt -c .constraints.txt

# DISTS uses ImageNet VGG16 weights from torchvision: download them now (compute nodes may be offline)
python -c "from DISTS_pytorch import DISTS; DISTS(); print('DISTS weights cached')"

python -c "import torch, cv2, skimage, yaml; print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
echo "Environment ready. Activate later with:  module load python/3.10.pytorch && source .venv/bin/activate"
