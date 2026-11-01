"""Anti-collapse regularisers and diagnostics (P4.5).

Regularisers
  l2sp            — ||θ − θ0||² over trainable params (weight-space anchor to the
                    initial model; Li et al. 2018 "L2-SP").
  prediction_anchor — on the SAME unlabeled image: for student queries that overlap
                    (IoU ≥ τ) a confident box of the frozen initial model, penalise
                    KL(frozen class probs ‖ student class probs) and GIoU/L1 between the
                    boxes. Keeps classification and box statistics from drifting.
  entropy_floor   — hinge on the mean differential entropy of the edge distributions:
                    penalises sharpening below `h_min` nats (over-sharpening collapse).

Diagnostics (`diagnose`) — numbers logged every N steps on a fixed probe batch:
  mean H_diff per edge, fraction of edges with max prob > 0.9, mean box w/h vs the
  frozen model, mean |Δcentre| vs frozen for matched queries, #detections > 0.3,
  mean max score, KL(frozen‖student) on class probs.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision

from tcld.model.base import DistOutput
from tcld.tcl import grid
from tcld.tcl.sharpness import differential_entropy


def l2sp(params: Iterable[Tuple[str, torch.nn.Parameter]], init_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    tot = None
    for name, p in params:
        if not p.requires_grad or name not in init_state:
            continue
        d = (p - init_state[name].to(p)).pow(2).sum()
        tot = d if tot is None else tot + d
    return tot if tot is not None else torch.zeros(())


@torch.no_grad()
def _match_by_iou(boxes_s: torch.Tensor, boxes_f: torch.Tensor, scores_f: torch.Tensor,
                  score_thresh: float, iou_thresh: float):
    keep = scores_f >= score_thresh
    if keep.sum() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes_s.device), torch.zeros(0, dtype=torch.long, device=boxes_s.device)
    fi = torch.nonzero(keep).flatten()
    iou = torchvision.ops.box_iou(boxes_s, boxes_f[fi])     # [Qs, F]
    best, j = iou.max(-1)
    si = torch.nonzero(best >= iou_thresh).flatten()
    return si, fi[j[si]]


def prediction_anchor(
    student: DistOutput, frozen: DistOutput,
    score_thresh: float = 0.3, iou_thresh: float = 0.7, box_weight: float = 1.0, cls_weight: float = 1.0,
    index_aligned: bool = False, position_aligned: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Both outputs are for the same batch of images (normalized cxcywh boxes).

    index_aligned=True: backbone+encoder are frozen, so the top-k queries of student and
    frozen model are the SAME queries (same index) -> anchor the class logits of ALL
    queries index-wise (BCE to the frozen probabilities) and the boxes of queries the
    frozen model is confident about. This closes the confidence-inflation collapse seen
    with IoU-matched anchoring (FINDINGS.md 2026-10-03).
    position_aligned=True: encoder (or backbone) trainable, so the top-k sets differ ->
    pair queries by the anchor position they were selected from (DistOutput.topk_ind):
    common positions get the index-style anchor to the frozen decoder output; student
    queries at positions the frozen model did not select get a classification anchor to
    the frozen model's dense encoder logits at that position (so no query is left free).
    index_aligned=False and position_aligned=False: IoU-matched anchoring (only queries
    overlapping confident frozen boxes).
    """
    tot = torch.zeros((), device=student.logits.device)
    n = 0
    if position_aligned:
        assert student.topk_ind is not None and frozen.topk_ind is not None, "topk_ind missing (adapter hook)"
        cls_tot = box_tot = torch.zeros((), device=tot.device)
        n_common = n_conf = 0
        B, Q = student.logits.shape[:2]
        for b in range(B):
            s_ind, f_ind = student.topk_ind[b], frozen.topk_ind[b]
            in_f = torch.isin(s_ind, f_ind)
            s_pos = in_f.nonzero(as_tuple=True)[0]
            cls_b = torch.zeros((), device=tot.device)
            if s_pos.numel():
                common = s_ind[s_pos]
                f_pos = (f_ind.unsqueeze(1) == common.unsqueeze(0)).float().argmax(0)        # [M]
                pf = frozen.logits[b][f_pos].detach().sigmoid()
                cls_b = cls_b + F.binary_cross_entropy_with_logits(student.logits[b][s_pos], pf, reduction="none").mean() * s_pos.numel()
                conf = pf.max(-1).values >= score_thresh
                if conf.any():
                    bs = torchvision.ops.box_convert(student.boxes[b][s_pos][conf], "cxcywh", "xyxy")
                    bf = torchvision.ops.box_convert(frozen.boxes[b][f_pos][conf].detach(), "cxcywh", "xyxy")
                    giou = (1 - torchvision.ops.generalized_box_iou(bs, bf).diagonal()).mean()
                    l1 = (student.boxes[b][s_pos][conf] - frozen.boxes[b][f_pos][conf].detach()).abs().sum(-1).mean()
                    box_tot = box_tot + giou + l1
                    n_conf += 1
            rest = (~in_f).nonzero(as_tuple=True)[0]
            if rest.numel() and frozen.enc_logits is not None:
                pe = frozen.enc_logits[b][s_ind[rest]].detach().sigmoid()
                cls_b = cls_b + F.binary_cross_entropy_with_logits(student.logits[b][rest], pe, reduction="none").mean() * rest.numel()
            cls_tot = cls_tot + cls_b / Q
            n_common += int(s_pos.numel())
        loss = cls_weight * cls_tot / B + box_weight * box_tot / max(n_conf, 1)
        return loss, {"anchor_common_frac": n_common / float(B * Q), "anchor_images_with_matches": float(n_conf)}
    if index_aligned:
        pf = frozen.logits.detach().sigmoid()                                  # [B, Q, C]
        cls = F.binary_cross_entropy_with_logits(student.logits, pf, reduction="none").mean()
        conf = pf.max(-1).values >= score_thresh                               # [B, Q]
        if conf.any():
            bs = torchvision.ops.box_convert(student.boxes[conf], "cxcywh", "xyxy")
            bf = torchvision.ops.box_convert(frozen.boxes.detach()[conf], "cxcywh", "xyxy")
            giou = (1 - torchvision.ops.generalized_box_iou(bs, bf).diagonal()).mean()
            l1 = (student.boxes[conf] - frozen.boxes.detach()[conf]).abs().sum(-1).mean()
        else:
            giou = l1 = torch.zeros((), device=tot.device)
        return cls_weight * cls + box_weight * (giou + l1), {"anchor_conf_queries": float(conf.sum())}
    for b in range(student.logits.shape[0]):
        bs = torchvision.ops.box_convert(student.boxes[b], "cxcywh", "xyxy")
        bf = torchvision.ops.box_convert(frozen.boxes[b].detach(), "cxcywh", "xyxy")
        sf = frozen.logits[b].detach().sigmoid().max(-1).values
        si, fi = _match_by_iou(bs.detach(), bf, sf, score_thresh, iou_thresh)
        if si.numel() == 0:
            continue
        # classification: BCE-style KL between sigmoid probs (multi-label)
        pf = frozen.logits[b][fi].detach().sigmoid()
        ls = student.logits[b][si]
        cls = F.binary_cross_entropy_with_logits(ls, pf, reduction="none").mean()   # mean over classes: O(0.1)
        giou = (1 - torchvision.ops.generalized_box_iou(bs[si], bf[fi]).diagonal()).mean()
        l1 = (student.boxes[b][si] - frozen.boxes[b][fi].detach()).abs().sum(-1).mean()
        tot = tot + cls_weight * cls + box_weight * (giou + l1)
        n += 1
    return tot / max(n, 1), {"anchor_images_with_matches": float(n)}


def entropy_floor(pos: torch.Tensor, probs: torch.Tensor, h_min: float) -> torch.Tensor:
    """Hinge: mean over (query, edge) of max(0, h_min - H_diff)."""
    h = differential_entropy(pos, probs)
    return F.relu(h_min - h).mean()


@torch.no_grad()
def diagnose(
    adapter, student: DistOutput, frozen: Optional[DistOutput], image_size: Tuple[int, int],
    score_thresh: float = 0.3,
) -> Dict[str, float]:
    probs = student.probs()
    pos = grid.bin_positions(student.ref_points, adapter.bin_values(), student.reg_scale, image_size)
    h = differential_entropy(pos, probs)                           # [B, Q, 4]
    scores = student.logits.sigmoid().max(-1).values              # [B, Q]
    conf = scores >= score_thresh
    out = {
        "H_diff_mean_conf": float(h[conf].mean()) if conf.any() else float("nan"),
        "H_diff_mean_all": float(h.mean()),
        "frac_maxprob_gt_0.9": float((probs.max(-1).values > 0.9).float().mean()),
        "n_det_conf": float(conf.sum()) / student.logits.shape[0],
        "score_mean_top10": float(scores.topk(10, dim=-1).values.mean()),
        "box_w_mean_conf": float(student.boxes[..., 2][conf].mean()) if conf.any() else float("nan"),
        "box_h_mean_conf": float(student.boxes[..., 3][conf].mean()) if conf.any() else float("nan"),
    }
    if frozen is not None:
        fs = frozen.logits.sigmoid().max(-1).values
        fconf = fs >= score_thresh
        out["frozen_n_det_conf"] = float(fconf.sum()) / frozen.logits.shape[0]
        out["frozen_box_w_mean_conf"] = float(frozen.boxes[..., 2][fconf].mean()) if fconf.any() else float("nan")
        out["frozen_box_h_mean_conf"] = float(frozen.boxes[..., 3][fconf].mean()) if fconf.any() else float("nan")
        dc, n = 0.0, 0
        for b in range(student.logits.shape[0]):
            bs = torchvision.ops.box_convert(student.boxes[b], "cxcywh", "xyxy")
            bf = torchvision.ops.box_convert(frozen.boxes[b], "cxcywh", "xyxy")
            si, fi = _match_by_iou(bs, bf, fs[b], score_thresh, 0.5)
            if si.numel():
                dc += float((student.boxes[b][si][:, :2] - frozen.boxes[b][fi][:, :2]).norm(dim=-1).mean()) * image_size[0]
                n += 1
        out["dcentre_vs_frozen_px"] = dc / n if n else float("nan")
    return out
