"""Zero-shot / adapted-model evaluation per camera.

Examples
  python scripts/run_eval.py --dataset mot --profile laptop --model dfine_s
  python scripts/run_eval.py --dataset virat --scenes 0100 0101 --split-seed 0
  python scripts/run_eval.py --dataset virat --checkpoint /runs/x/adapt_0100.pth --cameras 0100

MOT: each sequence is one camera; the last (1 - train_frac) of the sequence is the
eval part (default train_frac 0.7). VIRAT: clips of a scene are split into
train / eval clips with `split_clips_by_camera` (eval_frac 0.3, seeded).
Writes runs/eval/<tag>.json and prints the camera mean.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import yaml

from tcld.data import mot, virat
from tcld.data.common import split_clips_by_camera
from tcld.eval.run import evaluate_cameras, save_results, summarize
from tcld.model import load_dfine
from tcld.paths import profile_config, runs_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mot", "virat"], required=True)
    ap.add_argument("--profile", default="laptop")
    ap.add_argument("--model", default="dfine_s")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint (adapted model)")
    ap.add_argument("--cameras", nargs="*", default=None, help="MOT sequence names / VIRAT scene ids")
    ap.add_argument("--scenes", nargs="*", default=None, help="alias of --cameras for virat")
    ap.add_argument("--frame-stride", type=int, default=30)
    ap.add_argument("--train-frac", type=float, default=0.7, help="MOT: head fraction reserved for adaptation")
    ap.add_argument("--eval-frac", type=float, default=0.3, help="VIRAT: fraction of clips held out per scene")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    profile = yaml.safe_load(open(profile_config(args.profile)))
    device = profile.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    size = tuple(profile.get("input_size", [640, 640]))
    model = load_dfine(args.model, checkpoint=args.checkpoint, device=device)

    cams = args.cameras or args.scenes
    if args.dataset == "mot":
        clips = mot.list_sequences(names=cams or None)
        cameras = {c.camera: [c] for c in clips}
        start_frac = args.train_frac
    else:
        clips = virat.list_clips(scenes=cams or None)
        split = split_clips_by_camera(clips, eval_frac=args.eval_frac, seed=args.split_seed)
        cameras = {cam: ev for cam, (_, ev) in split.items() if ev}
        start_frac = 0.0
    if not cameras:
        raise SystemExit("no cameras / clips found — check data layout under $TCLD_DATA")
    print(f"dataset={args.dataset} cameras={list(cameras)} model={args.model} ckpt={model.checkpoint_path}")

    t0 = time.time()
    results = evaluate_cameras(model, cameras, device, frame_stride=args.frame_stride,
                               start_frac=start_frac, input_size=size,
                               batch_size=int(profile.get("eval_batch_size", 8)))
    mean = summarize(results)
    print("MEAN over cameras:", json.dumps({k: round(v, 2) for k, v in mean.items()}))
    tag = args.tag or f"{args.dataset}_{args.model}_{'adapted' if args.checkpoint else 'zeroshot'}"
    out = runs_dir() / "eval" / f"{tag}.json"
    save_results(results, out, extra={"args": vars(args), "runtime_s": time.time() - t0,
                                      "checkpoint": str(model.checkpoint_path)})
    print("wrote", out)


if __name__ == "__main__":
    main()
