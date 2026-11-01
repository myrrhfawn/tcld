"""Run per-camera adaptation for an experiment config.

  python scripts/run_adapt.py --exp configs/exp/tcld_dev.yml --profile laptop --dataset virat --cameras 0101 --seed 0

Outputs: runs/<exp_name>/<dataset>/result_<camera>_seed<seed>.json, history, checkpoint,
and one row appended to results/experiments.csv (APm/AP50/AP75 co + w for base and adapted).
"""

from __future__ import annotations

import argparse
import os
import csv
import subprocess
import time
from pathlib import Path

import torch
import yaml

from tcld.data import mot, virat
from tcld.data.common import split_clips_by_camera
from tcld.paths import DFINE_ROOT, RESULTS, profile_config, runs_dir
from tcld.train.adapt import adapt_camera

CSV_COLS = ["exp_id", "phase", "method", "model", "profile", "cameras", "seed", "iters", "delta", "motion",
            "loss_variant", "reg", "APm_co", "AP50_co", "AP75_co", "APm_w", "AP50_w", "APG_m_co", "APG_50_co",
            "APG_75_co", "coco_minival_AP", "runtime_min", "gpu", "dfine_commit", "config_path", "notes"]


def append_row(row: dict) -> None:
    path = RESULTS / "experiments.csv"
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--profile", default="laptop")
    ap.add_argument("--dataset", choices=["mot", "virat", "scenes100"], required=True)
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None, help="override exp.iters")
    ap.add_argument("--train-frac", type=float, default=0.7, help="MOT: head fraction for adaptation")
    ap.add_argument("--adapt-seq", default=None,
                    help="MOT cross-recording protocol: adapt on this (unlabeled) sequence in full and evaluate on the "
                         "whole labeled --cameras sequence (eval start 0); e.g. --cameras MOT17-04 --adapt-seq MOT17-03")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    exp = yaml.safe_load(open(args.exp))
    if exp.get("corruption"):
        os.environ["TCLD_CORRUPT"] = exp["corruption"]   # read by FrameReader in this process and the loader workers
    if args.iters is not None:
        exp["iters"] = args.iters
    profile = yaml.safe_load(open(profile_config(args.profile)))
    exp_name = exp.get("name", Path(args.exp).stem)
    out_dir = runs_dir() / exp_name / args.dataset
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=DFINE_ROOT, capture_output=True, text=True).stdout.strip()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    for cam in args.cameras:
        if args.dataset == "mot":
            clips = mot.list_sequences(names=[cam])
            if not clips:
                raise SystemExit(f"unknown MOT sequence {cam}")
            seq = clips[0]
            if args.adapt_seq:
                # cross-recording protocol: another recording of the same camera (no labels) for
                # adaptation, the whole labeled sequence for evaluation
                other = mot.list_sequences(names=[args.adapt_seq], with_gt=False)
                if not other:
                    raise SystemExit(f"unknown MOT sequence {args.adapt_seq}")
                train_clips, eval_clips, eval_start = other, [seq], 0.0
            else:
                # head of the sequence (unlabeled) for adaptation; tail for evaluation
                head = seq.__class__(**{**seq.__dict__, "num_frames": int(seq.num_frames * args.train_frac)})
                train_clips, eval_clips, eval_start = [head], [seq], args.train_frac
        elif args.dataset == "scenes100":
            from tcld.data import scenes100

            clips = scenes100.list_cameras(ids=[cam])
            if not clips:
                raise SystemExit(f"no Scenes100 frames for video {cam} under {scenes100.frames_root()}")
            train_clips, eval_clips, eval_start = clips, [], 0.0   # no manual annotations available -> no eval
        else:
            clips = virat.list_clips(scenes=[cam])
            tr, ev = split_clips_by_camera(clips, 0.3, args.split_seed)[cam]
            train_clips, eval_clips, eval_start = tr, ev, 0.0
        t0 = time.time()
        res = adapt_camera(exp, profile, cam, train_clips, eval_clips, args.seed, out_dir, eval_start_frac=eval_start)
        base, fin = res["zero_shot"], res["final"]
        crit = exp.get("criterion", {})
        append_row({
            "exp_id": f"{exp_name}_{args.dataset}_{cam}_s{args.seed}", "phase": exp.get("phase", "P4"),
            "method": exp.get("method", "tcld"), "model": exp.get("model", profile.get("default_model")),
            "profile": args.profile, "cameras": cam, "seed": args.seed, "iters": res["iters"],
            "delta": "/".join(map(str, exp.get("deltas", [1, 2, 5]))), "motion": crit.get("motion", "flow"),
            "loss_variant": f"{crit.get('tcl', {}).get('mode', 'asym')}-{crit.get('tcl', {}).get('divergence', 'kl')}",
            "reg": f"anchor{crit.get('anchor_weight', 1.0)}c{crit.get('anchor_cls_weight', 1.0)}b{crit.get('anchor_box_weight', 1.0)}_ag{crit.get('motion_agree_px', 0)}{'s' if crit.get('motion_agree_soft') else ''}_sz{crit.get('size_weight', 0)}{'o' if crit.get('size_only') else ''}_sc{int(bool(crit.get('use_scale')))}_L{int(bool(crit.get('all_layers')))}_fr{len(exp.get('freeze', ['b','e']))}_k{crit.get('ensemble_k', 1)}_t{exp.get('tcl_teacher', 'ema')}_ema{exp.get('ema_decay', 0.999)}_lr{exp.get('lr')}_iou{crit.get('iou_thresh', 0.5)}",
            "APm_co": round(fin.get("APm_co", float("nan")), 3), "AP50_co": round(fin.get("AP50_co", float("nan")), 3),
            "AP75_co": round(fin.get("AP75_co", float("nan")), 3), "APm_w": round(fin.get("APm_w", float("nan")), 3),
            "AP50_w": round(fin.get("AP50_w", float("nan")), 3),
            "APG_m_co": round(fin.get("APm_co", 0) - base.get("APm_co", 0), 3),
            "APG_50_co": round(fin.get("AP50_co", 0) - base.get("AP50_co", 0), 3),
            "APG_75_co": round(fin.get("AP75_co", 0) - base.get("AP75_co", 0), 3),
            "runtime_min": round((time.time() - t0) / 60, 1), "gpu": gpu, "dfine_commit": commit,
            "config_path": args.exp, "notes": f"{args.notes}{' adapt_seq=' + args.adapt_seq if args.adapt_seq else ''} final={res['final_source']}".strip(),
        })
        print(f"[{cam}] base APm={base.get('APm_co', float('nan')):.2f} -> final APm={fin.get('APm_co', float('nan')):.2f} "
              f"(APG {fin.get('APm_co', 0) - base.get('APm_co', 0):+.2f}, source={res['final_source']})")


if __name__ == "__main__":
    main()
