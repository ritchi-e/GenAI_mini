import os
import random

import numpy as np
import torch
import yaml


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        # dotted overrides, e.g.  train.iters=1000  data.root=/scratch/data
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_uint8(t):
    """(3,H,W) or (1,3,H,W) float tensor in ~[0,1] -> (H,W,3) uint8 numpy (round + clip)."""
    if t.dim() == 4:
        t = t[0]
    a = t.detach().float().clamp(0, 1).mul(255.0).round().byte().cpu().numpy()
    return a.transpose(1, 2, 0)


def uint8_to_tensor(img):
    """(H,W,3) uint8 numpy -> (1,3,H,W) float tensor in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_weights(model, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ("ema", "model", "state_dict", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    ckpt = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0
