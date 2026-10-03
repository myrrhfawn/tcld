"""P0.4 smoke test: the pinned D-FINE + checkpoint + our adapter work end to end.

Checks
1. model builds, checkpoint loads, GPU (if requested) is visible;
2. COCO AP on the fixed 50-image val subset via our own eval loop
   (tcld.eval.coco_subset) — numbers are written to runs/smoke/<model>.json;
3. the same AP via upstream `train.py --test-only` on the same subset; both must
   agree (|ΔAP| <= 0.1; upstream prints AP rounded to 3 decimals, hence the
   tolerance) — proves our loader/postprocess/eval path matches upstream.
   Note: predict() freezes params so nn.MultiheadAttention takes the same fused
   kernel path as upstream's EMA module; without that the subset AP differs by
   ~0.25 purely from top-k query-selection sensitivity to 1e-6 rounding;
4. distribution mode: corners -> boxes identity on real data, bin values printed;
5. FPS of plain eval-mode inference (reference for P2.3).

Usage: python scripts/smoke.py --profile laptop --model dfine_s [--skip-upstream]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from tcld.eval.coco_subset import evaluate_coco_subset
from tcld.model import load_dfine
from tcld.paths import DFINE_ROOT, data_dir, model_config, profile_config, runs_dir


def run_upstream_test_only(model: str, ckpt: Path, img_dir: Path, ann: Path, device: str) -> float:
    """Run external/D-FINE/train.py --test-only on the subset and parse AP@[.5:.95]."""
    cfg = model_config(model)
    cmd = [
        sys.executable, str(DFINE_ROOT / "train.py"),
        "-c", str(cfg), "-r", str(ckpt), "--test-only",
        "-u",
        f"val_dataloader.dataset.img_folder={img_dir}/",
        f"val_dataloader.dataset.ann_file={ann}",
        "val_dataloader.total_batch_size=8",
        "val_dataloader.num_workers=2",
        "HGNetv2.pretrained=False",
        f"output_dir={runs_dir() / 'smoke' / 'upstream'}",
    ]
    env = {"CUDA_VISIBLE_DEVICES": "" if device == "cpu" else "0"}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run(cmd, cwd=str(DFINE_ROOT), capture_output=True, text=True, env=env)
    out = proc.stdout + proc.stderr
    ap = None
    for line in out.splitlines():
        if "Average Precision" in line and "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line:
            ap = float(line.strip().split("=")[-1])
            break
    if ap is None:
        print(out[-4000:])
        raise RuntimeError("could not parse AP from upstream --test-only output")
    return ap * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="laptop")
    ap.add_argument("--model", default="dfine_s")
    ap.add_argument("--skip-upstream", action="store_true")
    ap.add_argument("--n-fps", type=int, default=30)
    args = ap.parse_args()

    profile = yaml.safe_load(open(profile_config(args.profile)))
    device = profile.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: profile wants cuda but it is not available -> cpu")
        device = "cpu"
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0),
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    print("torch", torch.__version__, "device", device)

    # 1. build + checkpoint
    m = load_dfine(args.model, device=device)
    print(f"model {args.model} loaded from {m.checkpoint_path}; reg_max={m.reg_max}, "
          f"num_classes={m.num_classes}, params={sum(p.numel() for p in m.model.parameters())/1e6:.1f}M")

    coco = data_dir() / "coco"
    img_dir = coco / "val2017"
    ann = coco / "annotations" / "instances_val2017_smoke50.json"
    if not ann.exists():
        raise FileNotFoundError(f"{ann} missing — run `make download-coco-subset`")

    # 2. our eval loop
    size = tuple(profile.get("input_size", [640, 640]))
    stats = evaluate_coco_subset(
        m, img_dir, ann, device=device, input_size=size,
        batch_size=int(profile.get("eval_batch_size", 8)),
        num_workers=int(profile.get("num_workers", 4)),
    )
    print(f"[tcld eval]  AP={stats['AP']:.2f} AP50={stats['AP50']:.2f} AP75={stats['AP75']:.2f} "
          f"on {stats['num_images']} images")

    # 3. upstream reference on the same subset
    result = {"model": args.model, "profile": args.profile, "device": device,
              "tcld_eval": stats, "reference_full_coco_AP": m.tcld_meta.get("coco_ap_reference")}
    if not args.skip_upstream:
        ap_up = run_upstream_test_only(args.model, m.checkpoint_path, img_dir, ann, device)
        print(f"[upstream]   AP={ap_up:.2f}")
        result["upstream_AP"] = ap_up
        diff = abs(ap_up - stats["AP"])
        print(f"|ΔAP| = {diff:.3f}  ->", "OK" if diff <= 0.1 else "MISMATCH")
        if diff > 0.1:
            raise SystemExit("our eval path disagrees with upstream --test-only")

    # 4. distribution mode on a real batch
    from tcld.eval.coco_subset import load_image_batch
    images, _ = load_image_batch(img_dir, ann, n=2, input_size=size)
    images = images.to(device)
    with torch.no_grad():
        d = m.forward_distributions(images, all_layers=True)
        rec = m.boxes_from_corners(d.corners, d.ref_points)
    err = (rec - d.boxes).abs().max().item()
    probs = d.probs()
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)  # [B,Q,4] shannon over bins
    conf = d.logits.sigmoid().max(-1).values
    top = conf > 0.4
    print(f"[dist mode] corners {tuple(d.corners.shape)}, aux layers {len(d.aux)}, "
          f"corners->boxes max err {err:.2e}, W(n)[:4]={[round(v, 3) for v in m.bin_values()[:4].tolist()]}, "
          f"mean bin-entropy (conf>0.4): {ent[top].mean().item() if top.any() else float('nan'):.3f} nats")
    assert err < 1e-4, "corners->boxes identity broken"
    result["dist_mode"] = {"corners_shape": list(d.corners.shape), "max_box_err": err}

    # 5. FPS (eval mode, batch 1, fp32 and amp)
    x = images[:1]
    orig = torch.tensor([[size[1], size[0]]], device=device)
    for amp in (False, True):
        if amp and device == "cpu":
            continue
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
            for _ in range(5):
                m.predict(x, orig)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.n_fps):
                m.predict(x, orig)
            if device == "cuda":
                torch.cuda.synchronize()
        fps = args.n_fps / (time.perf_counter() - t0)
        print(f"[fps] batch=1 amp={amp}: {fps:.1f} img/s")
        result[f"fps_amp{int(amp)}"] = fps

    out = runs_dir() / "smoke" / f"{args.model}_{args.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
