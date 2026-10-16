"""Train the denoiser.

Single GPU:   python train.py --config configs/small.yaml
Multi GPU:    torchrun --standalone --nproc_per_node=4 train.py --config configs/small.yaml
Overrides:    python train.py --config configs/small.yaml --opt train.total_iters=1000 data.num_workers=4

Training resumes automatically from <out_dir>/last.pth, so a job killed by the Slurm
wall-time limit can simply be resubmitted.
"""
import argparse
import csv
import math
import os
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from denoise.dataset import DegradationDataset, PairedFolder
from denoise.losses import CombinedLoss
from denoise.model import build_model
from denoise.utils import count_params, load_config, seed_everything, tensor_to_uint8

STOP = {"flag": False}


def _on_signal(signum, _frame):
    print(f"[signal] received {signum}; will checkpoint and exit", flush=True)
    STOP["flag"] = True


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, dev


def stage_for(it, stages):
    for s in stages:
        if it < s["until"]:
            return s
    return stages[-1]


def make_loader(cfg, stage, world, rank, it):
    d = cfg["data"]
    ds = DegradationDataset(d["root"], d["domains"], patch_size=stage["patch"],
                            strength=d.get("strength", 1.0), lowlight_p=d.get("lowlight_p", 0.15),
                            gray_p=d.get("gray_p", 0.08))
    workers = d.get("num_workers")
    if workers is None:
        workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) // world - 1)
    # different random stream per rank and per (re)start
    torch.manual_seed(cfg["train"]["seed"] + 1000 * rank + it)
    return DataLoader(ds, batch_size=stage["batch"] // world, num_workers=workers, shuffle=False,
                      pin_memory=True, drop_last=True, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None)


def lr_at(it, cfg):
    t = cfg["train"]
    warm, total, base, minlr = t["warmup_iters"], t["total_iters"], t["lr"], t["min_lr"]
    if it < warm:
        return base * (it + 1) / warm
    p = (it - warm) / max(1, total - warm)
    return minlr + 0.5 * (base - minlr) * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def validate(model, cfg, device, dists_metric=None):
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    ds = PairedFolder(cfg["data"]["val_root"], limit=cfg["train"].get("val_limit"))
    model.eval()
    ps, ss, ds_ = [], [], []
    for i in range(len(ds)):
        noisy, clean, _ = ds[i]
        out = model(noisy.unsqueeze(0).to(device)).float()
        o8, c8 = tensor_to_uint8(out), tensor_to_uint8(clean)
        ps.append(psnr_fn(c8, o8, data_range=255))
        ss.append(ssim_fn(c8, o8, channel_axis=2, data_range=255))
        if dists_metric is not None:
            x = torch.from_numpy(o8).permute(2, 0, 1)[None].float().div(255).to(device)
            y = torch.from_numpy(c8).permute(2, 0, 1)[None].float().div(255).to(device)
            ds_.append(float(dists_metric(x, y)))
    model.train()
    return float(np.mean(ps)), float(np.mean(ss)), (float(np.mean(ds_)) if ds_ else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--opt", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)

    rank, world, device = setup_distributed()
    main_proc = rank == 0
    seed_everything(cfg["train"]["seed"] + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(cfg["model"]).to(device)
    ema = build_model(cfg["model"]).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    if main_proc:
        print(f"[model] params: {count_params(model) / 1e6:.3f} M", flush=True)

    t = cfg["train"]
    stages = t["stages"]
    max_dists = max(s.get("w_dists", 0.0) for s in stages)
    loss_fn = CombinedLoss(t["loss"]["charb"], t["loss"]["ssim"], t["loss"]["fft"], w_dists=max_dists).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=(0.9, 0.9), weight_decay=t["weight_decay"])

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = use_cuda and t.get("amp", True)
    scaler_on = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_on)
    except (AttributeError, TypeError):  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_on)

    start_it, best = 0, -1e9
    last_path = os.path.join(out_dir, "last.pth")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_it, best = ck["iter"], ck.get("best", best)
        if main_proc:
            print(f"[resume] from {last_path} at iter {start_it}", flush=True)

    net = DDP(model, device_ids=[device.index]) if world > 1 else model

    dists_metric = None
    if main_proc and t.get("val_dists", True):
        try:
            from DISTS_pytorch import DISTS
            dists_metric = DISTS().to(device).eval()
        except Exception as e:  # noqa: BLE001
            print(f"[val] DISTS unavailable ({e}); skipping DISTS in validation")

    def save(path, it):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "iter": it, "best": best, "config": cfg}, path + ".tmp")
        os.replace(path + ".tmp", path)

    log_path = os.path.join(out_dir, "log.csv")
    if main_proc and not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "lr", "loss", "charb", "ssim", "fft", "dists",
                                    "val_psnr", "val_ssim", "val_dists", "it_per_s"])

    stage = stage_for(start_it, stages)
    loader = make_loader(cfg, stage, world, rank, start_it)
    data_iter = iter(loader)
    loss_fn.set_dists_weight(stage.get("w_dists", 0.0))
    if main_proc:
        print(f"[stage] {stage}", flush=True)

    it = start_it
    tic, run = time.time(), {}
    net.train()
    while it < t["total_iters"]:
        new_stage = stage_for(it, stages)
        if new_stage is not stage:
            stage = new_stage
            del data_iter, loader
            loader = make_loader(cfg, stage, world, rank, it)
            data_iter = iter(loader)
            loss_fn.set_dists_weight(stage.get("w_dists", 0.0))
            if main_proc:
                print(f"[stage] {stage}", flush=True)

        noisy, clean = next(data_iter)
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        lr = lr_at(it, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = net(noisy)
        loss, terms = loss_fn(pred.float(), clean)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            d = t["ema_decay"] if it > t["warmup_iters"] else 0.0
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        it += 1
        for k, v in terms.items():
            run[k] = run.get(k, 0.0) + v
        run["loss"] = run.get("loss", 0.0) + float(loss.detach())
        run["n"] = run.get("n", 0) + 1

        # stop flag must be agreed by all ranks
        stop = STOP["flag"]
        if world > 1:
            flag = torch.tensor([1.0 if stop else 0.0], device=device)
            dist.all_reduce(flag)
            stop = flag.item() > 0

        if main_proc and (it % t["log_every"] == 0 or stop):
            speed = run["n"] / (time.time() - tic)
            avg = {k: v / run["n"] for k, v in run.items() if k != "n"}
            msg = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"[it {it:>7d}] lr={lr:.2e} {msg} | {speed:.2f} it/s", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it, lr, avg["loss"], avg.get("charb", ""), avg.get("ssim", ""),
                                        avg.get("fft", ""), avg.get("dists", ""), "", "", "", f"{speed:.2f}"])
            tic, run = time.time(), {}

        if main_proc and (it % t["val_every"] == 0 or it == t["total_iters"]):
            p, s, dd = validate(ema, cfg, device, dists_metric)
            score = p / 40.0 + s - (0 if math.isnan(dd) else dd)   # balanced model-selection score
            print(f"[val {it}] PSNR={p:.3f} SSIM={s:.4f} DISTS={dd:.4f} score={score:.4f}", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it, "", "", "", "", "", "", f"{p:.4f}", f"{s:.5f}", f"{dd:.5f}", ""])
            if score > best:
                best = score
                torch.save({"ema": ema.state_dict(), "config": cfg, "iter": it,
                            "val": {"psnr": p, "ssim": s, "dists": dd}}, os.path.join(out_dir, "best.pth"))
                print(f"[val] new best -> {out_dir}/best.pth", flush=True)

        if main_proc and (it % t["save_every"] == 0 or stop or it == t["total_iters"]):
            save(last_path, it)
        if stop:
            break

    if main_proc:
        torch.save({"ema": ema.state_dict(), "config": cfg, "iter": it}, os.path.join(out_dir, "final.pth"))
        print("[done]", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
