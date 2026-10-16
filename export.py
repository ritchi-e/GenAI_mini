"""Export the trained EMA weights for submission.

    python export.py --ckpt runs/small/best.pth --out weights/model.pth [--onnx weights/model.onnx]

The .pth file is a plain dict {"state_dict": ..., "arch": {...}} loadable with torch.load
(no training state, ~fp32). ONNX export uses a fixed (1,3,256,256) input.
"""
import argparse
import os

import torch

from denoise.model import NAFUNet
from denoise.utils import load_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="weights/model.pth")
    ap.add_argument("--onnx", default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    arch = {k: v for k, v in ck["config"]["model"].items() if k != "name"}
    model = NAFUNet(**arch)
    load_weights(model, args.ckpt)
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch}, args.out)
    print(f"saved {args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)  arch={arch}")

    if args.onnx:
        x = torch.rand(1, 3, 256, 256)
        try:
            torch.onnx.export(model, x, args.onnx, input_names=["input"], output_names=["output"], opset_version=17)
            print(f"saved {args.onnx}")
        except Exception as e:  # noqa: BLE001  (optional: needs `pip install onnx`)
            print(f"[skip] ONNX export failed ({type(e).__name__}: {e}); the .pth file is complete on its own")


if __name__ == "__main__":
    main()
