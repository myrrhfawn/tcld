"""Per-camera evaluation of a detector on a video dataset with dense GT (P1.3 / P2).

For every camera: take the eval clips (or the eval part of a single long clip),
sample frames with `frame_stride`, run the detector, map COCO ids to the 2-class
target space, evaluate with the Scenes100 protocol (tcld.eval.two_class).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tcld.data.common import FrameGT, FrameReader, VideoClip
from tcld.data.pairs import _resize_rgb
from tcld.eval.two_class import Detections, coco_dets_to_target, evaluate_camera
from tcld.model.base import DistributionalDetector


def eval_frames_of_clip(clip: VideoClip, frame_stride: int, start_frac: float = 0.0) -> List[int]:
    """Frames to evaluate: every `frame_stride`-th frame from `start_frac` of the clip
    that has GT. For long single-clip cameras (MOT) use start_frac to hold out the
    tail of the sequence while the head is used for unlabeled adaptation."""
    first = int(clip.num_frames * start_frac)
    frames = [f for f in range(first, clip.num_frames, frame_stride)]
    if clip.gt is not None:
        frames = [f for f in frames if f in clip.gt]
    return frames


@torch.no_grad()
def detect_frames(
    model: DistributionalDetector,
    clip: VideoClip,
    frames: Sequence[int],
    device: str,
    input_size: Tuple[int, int] = (640, 640),
    batch_size: int = 8,
    score_thresh: float = 0.001,
) -> Dict[int, Detections]:
    reader = FrameReader(clip)
    out: Dict[int, Detections] = {}
    try:
        for k in range(0, len(frames), batch_size):
            fs = frames[k : k + batch_size]
            imgs, sizes = [], []
            for f in fs:
                rgb = reader.read(f)
                h0, w0 = rgb.shape[:2]
                imgs.append(torch.from_numpy(_resize_rgb(rgb, input_size)).permute(2, 0, 1).float() / 255)
                sizes.append([w0, h0])
            x = torch.stack(imgs).to(device)
            res = model.predict(x, torch.tensor(sizes, device=device))
            for f, r in zip(fs, res):
                keep = r["scores"] > score_thresh
                out[f] = coco_dets_to_target(
                    r["boxes"][keep].cpu().numpy(), r["scores"][keep].cpu().numpy(), r["labels"][keep].cpu().numpy()
                )
    finally:
        reader.close()
    return out


def evaluate_cameras(
    model: DistributionalDetector,
    cameras: Dict[str, List[VideoClip]],
    device: str,
    frame_stride: int = 30,
    start_frac: float = 0.0,
    input_size: Tuple[int, int] = (640, 640),
    batch_size: int = 8,
    masks: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """cameras: camera id -> list of eval clips (with gt). Returns per-camera metrics."""
    results: Dict[str, Dict[str, float]] = {}
    for cam, clips in cameras.items():
        t0 = time.time()
        frames_pairs: List[Tuple[FrameGT, Detections]] = []
        n_frames = 0
        for clip in clips:
            fs = eval_frames_of_clip(clip, frame_stride, start_frac)
            dets = detect_frames(model, clip, fs, device, input_size, batch_size)
            frames_pairs.extend((clip.gt[f], dets[f]) for f in fs)
            n_frames += len(fs)
        r = evaluate_camera(frames_pairs, mask=(masks or {}).get(cam))
        r["eval_seconds"] = time.time() - t0
        results[cam] = r
        if verbose:
            print(f"[{cam}] frames={n_frames} gt={int(r.get('n_gt', 0))} "
                  f"APm_co={r.get('APm_co', float('nan')):.2f} AP50_co={r.get('AP50_co', float('nan')):.2f} "
                  f"AP75_co={r.get('AP75_co', float('nan')):.2f} APm_w={r.get('APm_w', float('nan')):.2f} "
                  f"({r['eval_seconds']:.0f}s)")
    return results


def summarize(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["APm_co", "AP50_co", "AP75_co", "APm_w", "AP50_w", "AP75_w"]
    out = {"n_cameras": float(len(results))}
    for k in keys:
        vals = [r[k] for r in results.values() if k in r and not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def save_results(results: Dict[str, Dict[str, float]], path: Path, extra: Optional[dict] = None) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"per_camera": results, "mean": summarize(results), **(extra or {})}
    path.write_text(json.dumps(payload, indent=2, default=float))
