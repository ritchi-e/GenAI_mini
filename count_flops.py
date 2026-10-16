"""Measure FLOPs exactly as the organisers will: fixed input tensor (1, 3, 256, 256).

    python count_flops.py --config configs/small.yaml
    python count_flops.py --weights weights/model.pth

Note: fvcore counts one fused multiply-add as one "flop" (i.e. it reports MACs);
ptflops reports MACs explicitly. Both numbers are printed.
"""
import argparse
import warnings

import torch

from denoise.model import build_model
from denoise.utils import count_params, load_config


def measure(model):
    model = model.eval()
    x = torch.zeros(1, 3, 256, 256)
    res = {"params_M": count_params(model) / 1e6}
    try:
        from fvcore.nn import FlopCountAnalysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            res["fvcore_GFLOPs"] = fca.total() / 1e9
    except ImportError:
        res["fvcore_GFLOPs"] = None
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, (3, 256, 256), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        res["ptflops_GMACs"] = macs / 1e9
    except ImportError:
        res["ptflops_GMACs"] = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--weights", default=None, help="if given, architecture is read from the checkpoint")
    args = ap.parse_args()
    if args.weights:
        from inference import model_from_checkpoint
        model = model_from_checkpoint(args.weights)
    else:
        model = build_model(load_config(args.config)["model"])
    for k, v in measure(model).items():
        print(f"{k:>15}: {v:.4f}" if v is not None else f"{k:>15}: (package not installed)")


if __name__ == "__main__":
    main()
