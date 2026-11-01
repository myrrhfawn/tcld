"""Cross-frame matching of detections WITHOUT ground truth (P4.3).

Given the detector outputs on frames a and b (boxes in pixels, per-query target
scores / labels), find pairs (qa, qb) that are most likely the same object:

1. keep queries with score >= `score_thresh` (per frame);
2. greedy / Hungarian assignment on IoU between boxes of frame a — after shifting
   them by an externally supplied motion estimate (see tcld.tcl.motion) — and boxes
   of frame b; same target class required; IoU >= `iou_thresh`;
3. suppress duplicate queries of the same object inside a frame first (NMS), so a
   GT object is represented by one query per frame.

Returns index pairs plus the IoU and the min score of each pair for weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torchvision


@dataclass
class MatchResult:
    qa: torch.Tensor      # [M] query indices in frame a
    qb: torch.Tensor      # [M] query indices in frame b
    iou: torch.Tensor     # [M] IoU after motion compensation
    score: torch.Tensor   # [M] min(score_a, score_b)
    shift: torch.Tensor   # [M, 2] pixel shift applied to frame-a boxes (dx, dy)

    def __len__(self) -> int:
        return int(self.qa.numel())


def nms_keep(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
             score_thresh: float, nms_iou: float) -> torch.Tensor:
    """Indices of confident, non-duplicate queries (class-aware NMS)."""
    idx = torch.nonzero(scores >= score_thresh).flatten()
    if idx.numel() == 0:
        return idx
    keep = torchvision.ops.batched_nms(boxes[idx], scores[idx], labels[idx], nms_iou)
    return idx[keep]


def hungarian(cost: torch.Tensor):
    """scipy linear_sum_assignment on a CPU copy; returns (rows, cols) LongTensors."""
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)


def match_frames(
    boxes_a: torch.Tensor, scores_a: torch.Tensor, labels_a: torch.Tensor,
    boxes_b: torch.Tensor, scores_b: torch.Tensor, labels_b: torch.Tensor,
    shift_a: Optional[torch.Tensor] = None,
    score_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    nms_iou: float = 0.7,
    method: str = "hungarian",
) -> MatchResult:
    """boxes: [Q, 4] xyxy pixels; shift_a: [Q, 2] per-query (dx, dy) to apply to frame-a
    boxes before IoU (motion compensation), or None for no compensation."""
    dev = boxes_a.device
    ia = nms_keep(boxes_a, scores_a, labels_a, score_thresh, nms_iou)
    ib = nms_keep(boxes_b, scores_b, labels_b, score_thresh, nms_iou)
    empty = MatchResult(*(torch.zeros(0, dtype=torch.long, device=dev),) * 2,
                        torch.zeros(0, device=dev), torch.zeros(0, device=dev), torch.zeros(0, 2, device=dev))
    if ia.numel() == 0 or ib.numel() == 0:
        return empty
    shift = torch.zeros(boxes_a.shape[0], 2, device=dev, dtype=boxes_a.dtype) if shift_a is None else shift_a
    ba = boxes_a[ia] + shift[ia].repeat(1, 2)
    iou = torchvision.ops.box_iou(ba, boxes_b[ib])                   # [A, B]
    same = labels_a[ia][:, None] == labels_b[ib][None, :]
    iou = torch.where(same, iou, torch.zeros_like(iou))
    if method == "hungarian":
        r, c = hungarian(-iou)
        r, c = r.to(dev), c.to(dev)
    else:  # greedy by IoU
        flat = torch.argsort(iou.flatten(), descending=True)
        used_a, used_b, rs, cs = set(), set(), [], []
        for k in flat.tolist():
            i, j = divmod(k, iou.shape[1])
            if iou[i, j] < iou_thresh:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            rs.append(i)
            cs.append(j)
        r, c = torch.tensor(rs, dtype=torch.long, device=dev), torch.tensor(cs, dtype=torch.long, device=dev)
    ok = iou[r, c] >= iou_thresh
    r, c = r[ok], c[ok]
    qa, qb = ia[r], ib[c]
    return MatchResult(qa, qb, iou[r, c], torch.minimum(scores_a[qa], scores_b[qb]), shift[qa])
