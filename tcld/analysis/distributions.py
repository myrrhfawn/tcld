"""P3.3 — how do edge distributions of the same object change between frames?

For a clip with dense GT tracks and a frozen detector:
  for anchor frames t (stride) and deltas Δ:
    run the detector in distribution mode on frames t and t+Δ,
    match detections to GT boxes (IoU, same class) and pair them by GT track id,
    for every matched pair and each of the 4 edges record:
      - divergences between the two edge distributions in absolute pixels, under
        motion compensation variants: none / gt (GT centre shift + size scale) /
        pred (predicted-centre shift):  KL(b‖a→b), KL(a→b‖b), JS, W1, kept mass;
      - sharpness of each: differential entropy (nats, pixel cells), pixel std;
      - localisation error of each decoded edge vs GT (pixels) and box IoU vs GT;
      - context: class, GT box size, GT motion magnitude, confidence.
The long-format table (one row per pair×edge) feeds scripts/summarize_study.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision

from tcld.data.common import COCO_TO_TARGET, FrameGT, FrameReader, VideoClip
from tcld.data.pairs import _resize_rgb
from tcld.model.base import DistOutput
from tcld.model.dfine_adapter import DFINEAdapter
from tcld.tcl import grid
from tcld.tcl.sharpness import differential_entropy, pixel_std

EDGES = ["left", "top", "right", "bottom"]


def _target_label_indices(adapter: DFINEAdapter) -> Dict[int, int]:
    """D-FINE logit index -> target class, using the COCO category remap table."""
    from src.data.dataset import mscoco_label2category

    out = {}
    for idx, cat in mscoco_label2category.items():
        if cat in COCO_TO_TARGET:
            out[int(idx)] = COCO_TO_TARGET[cat]
    return out


@dataclass
class FrameDist:
    """Distribution-mode output for ONE frame restricted to target classes."""
    scores: torch.Tensor      # [Q] max prob over target classes
    labels: torch.Tensor      # [Q] target class (0/1)
    boxes_px: torch.Tensor    # [Q, 4] xyxy pixels (decoded)
    probs: torch.Tensor       # [Q, 4, R+1]
    pos: torch.Tensor         # [Q, 4, R+1] absolute pixel positions of bins
    ref_px: torch.Tensor      # [Q, 4] cxcywh pixels of the anchor
    out: DistOutput


def frame_dists(adapter: DFINEAdapter, d: DistOutput, b: int, orig_size: Tuple[int, int],
                label_map: Dict[int, int]) -> FrameDist:
    W, H = orig_size
    logits = d.logits[b]                     # [Q, C]
    prob_all = logits.sigmoid()
    idx = torch.tensor(sorted(label_map), device=logits.device)
    cls_of_idx = torch.tensor([label_map[int(i)] for i in idx.tolist()], device=logits.device)
    sub = prob_all[:, idx]                   # [Q, K]
    scores, k = sub.max(-1)
    labels = cls_of_idx[k]
    boxes = torchvision.ops.box_convert(d.boxes[b], "cxcywh", "xyxy") * torch.tensor([W, H, W, H], device=logits.device)
    probs = torch.softmax(d.corners[b], -1)
    pos = grid.bin_positions(d.ref_points[b], adapter.bin_values(), d.reg_scale, (W, H))
    ref_px = d.ref_points[b] * torch.tensor([W, H, W, H], device=logits.device)
    return FrameDist(scores, labels, boxes, probs, pos, ref_px, d)


def match_to_gt(fd: FrameDist, gt: FrameGT, score_thresh: float, iou_thresh: float) -> Dict[int, int]:
    """GT index -> query index (best IoU among same-class queries above threshold)."""
    if len(gt.boxes) == 0:
        return {}
    gtb = torch.as_tensor(gt.boxes, device=fd.boxes_px.device)
    iou = torchvision.ops.box_iou(gtb, fd.boxes_px)                       # [G, Q]
    same = torch.as_tensor(gt.labels, device=iou.device)[:, None] == fd.labels[None, :]
    ok = same & (fd.scores[None, :] >= score_thresh)
    iou = torch.where(ok, iou, torch.zeros_like(iou))
    best, q = iou.max(-1)
    out = {}
    used = set()
    for g in torch.argsort(best, descending=True).tolist():
        if best[g] < iou_thresh:
            break
        qi = int(q[g])
        if qi in used:
            continue
        used.add(qi)
        out[g] = qi
    return out


def _edge_coords(boxes_xyxy: np.ndarray) -> np.ndarray:
    return boxes_xyxy  # columns already are left, top, right, bottom


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (p * ((p + eps).log() - (q + eps).log())).sum(-1)


def _js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


@torch.no_grad()
def study_pair(
    adapter: DFINEAdapter,
    fa: FrameDist, fb: FrameDist,
    gta: FrameGT, gtb: FrameGT,
    ma: Dict[int, int], mb: Dict[int, int],
    context: dict,
    flow: Optional[np.ndarray] = None,
) -> List[dict]:
    """Rows for all GT tracks matched in both frames. `flow` (H, W, 2, pixels of the
    original frame, a->b) adds the optical-flow motion variant."""
    from tcld.tcl.motion import shift_from_flow
    ta = {int(t): g for g, t in enumerate(gta.track_ids)}
    tb = {int(t): g for g, t in enumerate(gtb.track_ids)}
    rows: List[dict] = []
    for tid, ga in ta.items():
        gb = tb.get(tid)
        if gb is None or ga not in ma or gb not in mb:
            continue
        qa, qb = ma[ga], mb[gb]
        box_a_gt, box_b_gt = gta.boxes[ga], gtb.boxes[gb]
        ca = np.array([(box_a_gt[0] + box_a_gt[2]) / 2, (box_a_gt[1] + box_a_gt[3]) / 2])
        cb = np.array([(box_b_gt[0] + box_b_gt[2]) / 2, (box_b_gt[1] + box_b_gt[3]) / 2])
        size_a = np.array([box_a_gt[2] - box_a_gt[0], box_a_gt[3] - box_a_gt[1]])
        size_b = np.array([box_b_gt[2] - box_b_gt[0], box_b_gt[3] - box_b_gt[1]])
        pa, pb = fa.probs[qa], fb.probs[qb]                       # [4, R+1]
        pos_a, pos_b = fa.pos[qa], fb.pos[qb]                     # [4, R+1]
        dev = pa.device
        # motion variants: shift applied to frame-a positions to land in frame-b coordinates
        pred_shift = (fb.boxes_px[qb][:2] + fb.boxes_px[qb][2:]) / 2 - (fa.boxes_px[qa][:2] + fa.boxes_px[qa][2:]) / 2
        flow_shift, flow_valid = (None, False)
        if flow is not None:
            fs, fv = shift_from_flow(flow, fa.boxes_px[qa][None].cpu())
            flow_shift, flow_valid = fs[0].to(dev), bool(fv[0])
        variants = {
            "none": pos_a,
            "gt": grid.shift_positions(pos_a[None], torch.tensor(cb - ca, device=dev, dtype=pos_a.dtype)[None],
                                       torch.tensor(size_b / np.maximum(size_a, 1e-6), device=dev, dtype=pos_a.dtype)[None],
                                       torch.tensor(ca, device=dev, dtype=pos_a.dtype)[None])[0],
            "pred": grid.shift_positions(pos_a[None], pred_shift[None])[0],
        }
        if flow is not None:
            variants["flow"] = grid.shift_positions(pos_a[None], (flow_shift if flow_valid else pred_shift)[None])[0]
        h_a = differential_entropy(pos_a, pa)                     # [4]
        h_b = differential_entropy(pos_b, pb)
        s_a, s_b = pixel_std(pos_a, pa), pixel_std(pos_b, pb)
        err_a = np.abs(fa.boxes_px[qa].cpu().numpy() - box_a_gt)   # [4] per edge
        err_b = np.abs(fb.boxes_px[qb].cpu().numpy() - box_b_gt)
        iou_a = float(torchvision.ops.box_iou(fa.boxes_px[qa][None], torch.as_tensor(box_a_gt, device=dev)[None]))
        iou_b = float(torchvision.ops.box_iou(fb.boxes_px[qb][None], torch.as_tensor(box_b_gt, device=dev)[None]))
        div = {}
        for name, pos_a_v in variants.items():
            a_on_b, kept = grid.resample(pos_a_v[None], pa[None], pos_b[None])
            a_on_b, kept = a_on_b[0], kept[0]
            div[name] = {
                "kl_b_a": _kl(pb, a_on_b), "kl_a_b": _kl(a_on_b, pb), "js": _js(pb, a_on_b),
                "w1": grid.wasserstein1(pos_a_v[None], pa[None], pos_b[None], pb[None])[0], "kept": kept,
            }
        for e, ename in enumerate(EDGES):
            row = dict(context)
            row.update({
                "track": tid, "edge": ename, "cls": int(gta.labels[ga]),
                "score_a": float(fa.scores[qa]), "score_b": float(fb.scores[qb]),
                "gt_w": float(size_a[0]), "gt_h": float(size_a[1]),
                "motion_px": float(np.linalg.norm(cb - ca)), "scale_ratio": float(np.sqrt(np.prod(size_b) / max(np.prod(size_a), 1e-6))),
                "pred_shift_err": float(torch.norm(pred_shift.cpu() - torch.tensor(cb - ca, dtype=torch.float32))),
                "flow_shift_err": float(torch.norm(flow_shift.cpu() - torch.tensor(cb - ca, dtype=torch.float32))) if flow_valid else float("nan"),
                "flow_valid": bool(flow_valid),
                "h_a": float(h_a[e]), "h_b": float(h_b[e]), "std_a": float(s_a[e]), "std_b": float(s_b[e]),
                "err_a": float(err_a[e]), "err_b": float(err_b[e]), "iou_a": iou_a, "iou_b": iou_b,
            })
            for name, dd in div.items():
                for k, v in dd.items():
                    row[f"{k}_{name}"] = float(v[e])
            rows.append(row)
    return rows


@torch.no_grad()
def study_clip(
    adapter: DFINEAdapter,
    clip: VideoClip,
    deltas: Sequence[int] = (1, 2, 5, 10, 30),
    stride: int = 30,
    device: str = "cuda",
    input_size: Tuple[int, int] = (640, 640),
    score_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    max_anchors: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    use_flow: bool = True,
    flow_downscale: int = 2,
    flow_fn=None,
) -> pd.DataFrame:
    """flow_fn(frame_a, frame_b) -> [H, W, 2] overrides Farneback (e.g. motion.RaftFlow)."""
    from tcld.tcl.motion import dense_flow

    if flow_fn is None:
        def flow_fn(a, b):
            return dense_flow(a, b, downscale=flow_downscale)

    assert clip.gt is not None, "study needs dense GT"
    label_map = _target_label_indices(adapter)
    reader = FrameReader(clip)
    end = clip.num_frames if end_frame is None else min(end_frame, clip.num_frames)
    anchors = [t for t in range(start_frame, end - max(deltas), stride) if t in clip.gt]
    if max_anchors is not None and len(anchors) > max_anchors:
        anchors = anchors[:: max(1, len(anchors) // max_anchors)][:max_anchors]
    rows: List[dict] = []
    orig = (clip.width, clip.height)

    def run(frame: int):
        rgb = reader.read(frame)
        x = torch.from_numpy(_resize_rgb(rgb, input_size)).permute(2, 0, 1).float()[None].to(device) / 255
        d = adapter.forward_distributions(x)
        return frame_dists(adapter, d, 0, orig, label_map), rgb

    try:
        for t in anchors:
            fa, rgb_a = run(t)
            ma = match_to_gt(fa, clip.gt[t], score_thresh, iou_thresh)
            for delta in deltas:
                tb_ = t + delta
                if tb_ not in clip.gt or tb_ >= clip.num_frames:
                    continue
                fb, rgb_b = run(tb_)
                mb = match_to_gt(fb, clip.gt[tb_], score_thresh, iou_thresh)
                flow = flow_fn(rgb_a, rgb_b) if use_flow else None
                ctx = {"clip": clip.clip_id, "camera": clip.camera, "t": t, "delta": delta,
                       "fps": clip.fps, "n_gt_a": len(clip.gt[t].boxes), "n_matched_a": len(ma)}
                rows.extend(study_pair(adapter, fa, fb, clip.gt[t], clip.gt[tb_], ma, mb, ctx, flow=flow))
    finally:
        reader.close()
    return pd.DataFrame(rows)
