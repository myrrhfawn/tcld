"""Per-camera 2-class (person / vehicle) COCO evaluation, replicating the Scenes100
protocol (external/scenes100/adaptation/evaluator.py::eval_AP) on any dataset that
provides FrameGT:

* detections in COCO-80 category ids are mapped to {person, vehicle}
others dropped;
* a detection or GT box touching an ignore region (any of its 4 corners inside an
  ignore box / polygon mask) is removed before evaluation — Scenes100 does exactly
  this with its non-evaluation polygons (`_check_overlap`);
* annotation ids start at 1: pycocotools treats a matched GT id of 0 as 'no match'
  (Scenes100's eval_AP numbers its GT from 0 and silently loses one box per video);
* metrics per camera: mAP@[.5:.95], AP50, AP75 over the 2 classes ("co" = class
  mean as COCOeval reports it), plus the class-frequency weighted variants ("w");
* APG = mean over cameras of (adapted - base).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from tcld.data.common import COCO_TO_TARGET, TARGET_CAT_IDS, TARGET_CLASSES, FrameGT


@dataclass
class Detections:
    boxes: np.ndarray    # [N, 4] xyxy pixels
    scores: np.ndarray   # [N]
    labels: np.ndarray   # [N] target labels in {0, 1}


def coco_dets_to_target(boxes: np.ndarray, scores: np.ndarray, coco_labels: np.ndarray) -> Detections:
    keep = np.isin(coco_labels, list(COCO_TO_TARGET))
    labels = np.array([COCO_TO_TARGET[int(c)] for c in coco_labels[keep]], np.int64)
    return Detections(boxes[keep].astype(np.float32), scores[keep].astype(np.float32), labels)


def _corners_in_boxes(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """True for boxes having at least one corner strictly inside any region (xyxy)."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes), bool)
    xs = np.stack([boxes[:, 0], boxes[:, 2], boxes[:, 0], boxes[:, 2]], 1)  # [N,4]
    ys = np.stack([boxes[:, 1], boxes[:, 1], boxes[:, 3], boxes[:, 3]], 1)
    inside = (
        (xs[:, :, None] > regions[None, None, :, 0]) & (xs[:, :, None] < regions[None, None, :, 2])
        & (ys[:, :, None] > regions[None, None, :, 1]) & (ys[:, :, None] < regions[None, None, :, 3])
    )
    return inside.any(axis=(1, 2))


def _corners_in_mask(boxes: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None or len(boxes) == 0:
        return np.zeros(len(boxes), bool)
    H, W = mask.shape
    x1 = np.clip(boxes[:, 0].astype(int), 0, W - 1)
    x2 = np.clip(boxes[:, 2].astype(int), 0, W - 1)
    y1 = np.clip(boxes[:, 1].astype(int), 0, H - 1)
    y2 = np.clip(boxes[:, 3].astype(int), 0, H - 1)
    return mask[y1, x1] | mask[y1, x2] | mask[y2, x1] | mask[y2, x2]


def filter_ignore(boxes: np.ndarray, gt: FrameGT, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Keep-mask for boxes not touching GT ignore boxes nor the polygon mask."""
    drop = _corners_in_boxes(boxes, gt.ignore_boxes) | _corners_in_mask(boxes, mask)
    return ~drop


def evaluate_camera(
    frames: Sequence[Tuple[FrameGT, Detections]],
    mask: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """frames: list of (gt, dets) for the evaluated frames of ONE camera.
    Returns {APm_co, AP50_co, AP75_co, APm_w, AP50_w, AP75_w, AP_person, AP_vehicle, n_gt, n_frames}."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    class_count = np.zeros(len(TARGET_CLASSES))
    for i, (gt, d) in enumerate(frames):
        images.append({"id": i, "width": 0, "height": 0, "file_name": ""})
        keep_gt = filter_ignore(gt.boxes, gt, mask)
        for b, lab in zip(gt.boxes[keep_gt], gt.labels[keep_gt]):
            x1, y1, x2, y2 = map(float, b)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": TARGET_CAT_IDS[int(lab)],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                         "iscrowd": 0, "segmentation": []})
            class_count[int(lab)] += 1
        keep_d = filter_ignore(d.boxes, gt, mask)
        for b, s, lab in zip(d.boxes[keep_d], d.scores[keep_d], d.labels[keep_d]):
            x1, y1, x2, y2 = map(float, b)
            dets.append({"image_id": i, "category_id": TARGET_CAT_IDS[int(lab)],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    coco_dict = {
        "images": images, "annotations": anns,
        "categories": [{"id": TARGET_CAT_IDS[i], "name": n} for i, n in enumerate(TARGET_CLASSES)],
    }
    out: Dict[str, float] = {"n_gt": float(len(anns)), "n_frames": float(len(frames)), "n_dets": float(len(dets))}
    if not anns:
        return out
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(coco_dict, f)
        gt_path = f.name
    quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with quiet:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(dets) if dets else coco_gt.loadRes([{"image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.0}])
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        out.update({"APm_co": ev.stats[0] * 100, "AP50_co": ev.stats[1] * 100, "AP75_co": ev.stats[2] * 100})
        per_class = {}
        for lab, name in enumerate(TARGET_CLASSES):
            if class_count[lab] == 0:
                per_class[name] = (np.nan, np.nan, np.nan)
                continue
            ev.params.catIds = [TARGET_CAT_IDS[lab]]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            per_class[name] = (ev.stats[0] * 100, ev.stats[1] * 100, ev.stats[2] * 100)
    Path(gt_path).unlink(missing_ok=True)
    w = class_count / class_count.sum()
    for k, key in enumerate(("APm_w", "AP50_w", "AP75_w")):
        out[key] = float(np.nansum([w[lab] * per_class[n][k] for lab, n in enumerate(TARGET_CLASSES)]))
    for name in TARGET_CLASSES:
        out[f"APm_{name}"] = float(per_class[name][0])
    return out


def apg(base: Dict[str, Dict[str, float]], adapted: Dict[str, Dict[str, float]],
        keys: Iterable[str] = ("APm_co", "AP50_co", "AP75_co", "APm_w", "AP50_w")) -> Dict[str, float]:
    """Average AP gain over cameras present in both dicts (Scenes100 eq. 6)."""
    cams = sorted(set(base) & set(adapted))
    out = {"n_cameras": float(len(cams))}
    for k in keys:
        deltas = [adapted[c][k] - base[c][k] for c in cams if k in base[c] and k in adapted[c]]
        out[f"APG_{k}"] = float(np.mean(deltas)) if deltas else float("nan")
        out[f"APG_{k}_std"] = float(np.std(deltas)) if deltas else float("nan")
    return out
