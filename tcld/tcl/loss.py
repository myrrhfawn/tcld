"""Temporal consistency loss on edge distributions (P4.4).

Inputs are per matched pair m (frame a = t, frame b = t + Δ), per edge e:
  corners_a, corners_b : logits [M, 4, R+1]
  pos_a, pos_b         : absolute pixel positions of bins [M, 4, R+1] (pos_a already
                         motion-compensated into frame-b coordinates)
  weights              : optional [M] pair weights (confidence, motion magnitude ...)

Modes
  asym  : the sharper distribution (lower differential entropy in pixels) is the
          target (stop-grad); the blurrier side is pulled towards it. Direction is
          decided per pair and edge.
  sym   : Jensen-Shannon between both (both sides receive gradient).
  ema   : `target_*` come from an EMA / frozen teacher on frame a (caller supplies
          corners/pos of the teacher); only the student side (frame b) is trained.
          Sharpness becomes a weight, not a direction.

Divergence
  kl : KL(target ‖ student) after resampling the target onto the student grid
       (grid.resample, piecewise-uniform CDF interpolation). Temperature T applies
       to the student logits (and the target logits before resampling).
  w1 : 1-D Wasserstein between the two point-mass distributions in pixels; no
       resampling needed, bounded gradient when supports barely overlap.

Pairs whose resampled target keeps less than `min_kept` of its mass on the student
support are dropped (reported in the stats).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tcld.tcl import grid
from tcld.tcl.sharpness import differential_entropy


@dataclass
class TCLConfig:
    mode: str = "asym"          # asym | sym | ema
    divergence: str = "kl"      # kl | w1
    temperature: float = 1.0
    min_kept: float = 0.8       # drop pairs whose target mass outside student support > 1 - min_kept
    sharp_margin: float = 0.0   # nats: |H_a - H_b| must exceed this to pick a direction (asym)
    w1_scale: float = 1.0       # W1 is in pixels; divide to bring to O(1)
    w1_cap: float = 32.0        # clip per-edge W1 (pixels) before scaling: one wrong match must not dominate
    renormalize: bool = True
    size_ref: Optional[float] = None   # scale-invariant W1: multiply per-edge W1 by size_ref / box extent along the edge axis
    size_norm_down_only: bool = True   # only down-weight objects larger than size_ref (factor clipped at 1)
    min_denom_pairs: float = 0.0       # loss denominator = max(sum of pair weights, this): few pairs no longer get the full weight


@dataclass
class TCLOutput:
    loss: torch.Tensor
    stats: Dict[str, float] = field(default_factory=dict)


def _probs(logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.softmax(logits / T, dim=-1)


def _kl_to_student(target_probs: torch.Tensor, student_logits: torch.Tensor, T: float) -> torch.Tensor:
    """KL(target ‖ student) per (pair, edge). target_probs detached by the caller."""
    log_q = F.log_softmax(student_logits / T, dim=-1)
    log_p = target_probs.clamp_min(1e-12).log()
    return (target_probs * (log_p - log_q)).sum(-1) * (T * T)


def _directional(
    tgt_logits: torch.Tensor, tgt_pos: torch.Tensor,
    stu_logits: torch.Tensor, stu_pos: torch.Tensor, cfg: TCLConfig,
    edge_factor: Optional[torch.Tensor] = None,
):
    """Loss for 'tgt teaches stu' on every (pair, edge). Returns (loss [M,4], kept [M,4])."""
    with torch.no_grad():
        tgt_probs = _probs(tgt_logits, cfg.temperature)
    if cfg.divergence == "kl":
        tgt_on_stu, kept = grid.resample(tgt_pos, tgt_probs, stu_pos, renormalize=cfg.renormalize)
        loss = _kl_to_student(tgt_on_stu.detach(), stu_logits, cfg.temperature)
    elif cfg.divergence == "w1":
        stu_probs = _probs(stu_logits, cfg.temperature)
        loss = grid.wasserstein1(tgt_pos, tgt_probs, stu_pos, stu_probs)
        if edge_factor is not None:
            loss = loss * edge_factor
        loss = loss.clamp_max(cfg.w1_cap) / cfg.w1_scale
        kept = torch.ones_like(loss)
    else:
        raise ValueError(cfg.divergence)
    return loss, kept


def edge_factor(extent: Optional[torch.Tensor], cfg: TCLConfig) -> Optional[torch.Tensor]:
    """Per-edge multiplier size_ref / extent (W1 in units of the box size). D-FINE bins scale
    with the box, so pixel W1 grows linearly with object size and large objects dominate."""
    if cfg.size_ref is None or extent is None:
        return None
    f = cfg.size_ref / extent.detach().clamp_min(1.0)
    return f.clamp_max(1.0) if cfg.size_norm_down_only else f


def tcl_loss(
    corners_a: torch.Tensor, pos_a: torch.Tensor,
    corners_b: torch.Tensor, pos_b: torch.Tensor,
    cfg: TCLConfig,
    weights: Optional[torch.Tensor] = None,
    target_corners: Optional[torch.Tensor] = None,
    target_pos: Optional[torch.Tensor] = None,
    extent: Optional[torch.Tensor] = None,
) -> TCLOutput:
    """extent: optional [M, 4] box extent in pixels along each edge's axis (w, h, w, h); used with cfg.size_ref."""
    M = corners_a.shape[0]
    dev = corners_a.device
    if M == 0:
        return TCLOutput(torch.zeros((), device=dev, requires_grad=True), {"n_pairs": 0, "n_used": 0})
    w = torch.ones(M, device=dev) if weights is None else weights
    stats: Dict[str, float] = {"n_pairs": float(M)}
    ef = edge_factor(extent, cfg)
    if ef is not None:
        stats["size_factor_mean"] = float(ef.mean())

    if cfg.mode == "ema":
        assert target_corners is not None and target_pos is not None, "ema mode needs teacher outputs"
        loss_e, kept = _directional(target_corners.detach(), target_pos, corners_b, pos_b, cfg, ef)
        use = kept >= cfg.min_kept
    elif cfg.mode == "sym":
        # JS: each side towards the mixture; implemented as 0.5*(KL(a→b-grid‖b)+KL(b→a-grid‖a))
        l_ab, k_ab = _directional(corners_a, pos_a, corners_b, pos_b, cfg, ef)
        l_ba, k_ba = _directional(corners_b, pos_b, corners_a, pos_a, cfg, ef)
        loss_e = 0.5 * (l_ab + l_ba)
        kept = torch.minimum(k_ab, k_ba)
        use = kept >= cfg.min_kept
    elif cfg.mode == "asym":
        with torch.no_grad():
            h_a = differential_entropy(pos_a, _probs(corners_a, 1.0))
            h_b = differential_entropy(pos_b, _probs(corners_b, 1.0))
            a_teaches = h_a < h_b - cfg.sharp_margin
            b_teaches = h_b < h_a - cfg.sharp_margin
        l_ab, k_ab = _directional(corners_a, pos_a, corners_b, pos_b, cfg, ef)   # a (sharp) -> b
        l_ba, k_ba = _directional(corners_b, pos_b, corners_a, pos_a, cfg, ef)   # b (sharp) -> a
        loss_e = torch.where(a_teaches, l_ab, torch.where(b_teaches, l_ba, torch.zeros_like(l_ab)))
        kept = torch.where(a_teaches, k_ab, k_ba)
        use = (a_teaches | b_teaches) & (kept >= cfg.min_kept)
        stats["frac_a_teaches"] = float(a_teaches.float().mean())
        stats["frac_undecided"] = float((~(a_teaches | b_teaches)).float().mean())
    else:
        raise ValueError(cfg.mode)

    use_f = use.float()
    # torch.where (not multiplication) so that inf/NaN in dropped entries cannot leak
    loss_e = torch.where(use, loss_e, torch.zeros_like(loss_e))
    denom = (use_f * w[:, None]).sum().clamp_min(max(1.0, 4.0 * cfg.min_denom_pairs))   # 4 edges per pair
    loss = (loss_e * w[:, None]).sum() / denom
    stats.update({
        "n_used": float(use_f.sum()),
        "frac_dropped_support": float((kept < cfg.min_kept).float().mean()),
        "loss_mean_used": float(loss_e.detach().sum() / use_f.sum().clamp_min(1.0)),
        "kept_mean": float(kept.mean()),
    })
    return TCLOutput(loss, stats)


# ----------------------------------------------------------------------------- translation-invariant target
def size_distribution(pos: torch.Tensor, probs: torch.Tensor):
    """Distributions of box WIDTH (right - left) and HEIGHT (bottom - top) as point masses.

    pos, probs: [M, 4, R+1] (edges: left, top, right, bottom), absolute pixel positions of
    bins. Edges are modelled as independent, so the size distribution is the convolution:
    values v[i,j] = pos_r[i] - pos_l[j] with mass p_r[i] * p_l[j]  ->  [M, (R+1)^2].
    Translation of the object cancels exactly, so no motion estimate is needed.
    Returns (values [M, 2, K], masses [M, 2, K]) for (width, height)."""
    pl, pt, pr, pb = pos.unbind(1)
    ql, qt, qr, qb = probs.unbind(1)
    w_val = (pr[:, :, None] - pl[:, None, :]).flatten(1)
    w_mass = (qr[:, :, None] * ql[:, None, :]).flatten(1)
    h_val = (pb[:, :, None] - pt[:, None, :]).flatten(1)
    h_mass = (qb[:, :, None] * qt[:, None, :]).flatten(1)
    return torch.stack([w_val, h_val], 1), torch.stack([w_mass, h_mass], 1)


def size_consistency_loss(
    corners_a: torch.Tensor, pos_a: torch.Tensor, corners_b: torch.Tensor, pos_b: torch.Tensor,
    cfg: TCLConfig, weights: Optional[torch.Tensor] = None, mode: str = "sym",
    extent: Optional[torch.Tensor] = None,
) -> TCLOutput:
    """W1 between the width/height distributions of the same object in frames a and b.
    pos_* are the UNCOMPENSATED bin positions (translation cancels). mode: sym (stop-grad
    each way, teacher-free) | ema (a is a detached target)."""
    M = corners_a.shape[0]
    if M == 0:
        return TCLOutput(torch.zeros((), device=corners_a.device, requires_grad=True), {"n_pairs": 0})
    pa = _probs(corners_a, cfg.temperature)
    pb = _probs(corners_b, cfg.temperature)
    va, ma = size_distribution(pos_a.detach(), pa)
    vb, mb = size_distribution(pos_b.detach(), pb)
    if mode == "sym":
        l_ab = grid.wasserstein1(va, ma.detach(), vb, mb)   # a teaches b
        l_ba = grid.wasserstein1(vb, mb.detach(), va, ma)   # b teaches a
        loss_e = 0.5 * (l_ab + l_ba)
    else:
        loss_e = grid.wasserstein1(va, ma.detach(), vb, mb)
    ef = edge_factor(extent, cfg)
    if ef is not None:
        loss_e = loss_e * ef[:, :2]                            # (w, h)
    loss_e = loss_e.clamp_max(cfg.w1_cap) / cfg.w1_scale       # [M, 2]
    w = torch.ones(M, device=corners_a.device) if weights is None else weights
    loss = (loss_e * w[:, None]).sum() / (w[:, None].expand_as(loss_e)).sum().clamp_min(1.0)
    return TCLOutput(loss, {"n_pairs": float(M), "size_w1_mean": float(loss_e.detach().mean() * cfg.w1_scale)})
