"""Geometry of D-FINE edge distributions and transport between bin grids (P3.1 / P4.1).

D-FINE predicts, for each query and each of the 4 edges (left, top, right, bottom),
a softmax distribution p over R+1 bins. Bin n carries the value W(n) (the
"weighting function"), and the edge offset is the expectation Σ p_n W(n). The
absolute edge position follows distance2bbox (external/D-FINE/src/zoo/dfine/dfine_utils.py):

    left   = cx - (0.5 s + W(n)) * w / s
    top    = cy - (0.5 s + W(n)) * h / s
    right  = cx + (0.5 s + W(n)) * w / s
    bottom = cy + (0.5 s + W(n)) * h / s

with (cx, cy, w, h) = ref_points_initial (normalized) and s = reg_scale. Hence every
query has its *own* grid of absolute bin positions, and distributions of two
queries (e.g. the same object in two frames) can only be compared after mapping
both to absolute coordinates. This module provides:

* `bin_positions`   — absolute (pixel) position of every bin, [..., 4, R+1];
* `cell_edges`      — boundaries of the cell around each bin (midpoints), [..., 4, R+2];
* `resample`        — move a distribution from one grid to another via CDF
                      interpolation (differentiable w.r.t. the source probabilities);
* `wasserstein1`    — W1 between two distributions on different grids (pixels);
* `mean_var`        — mean / variance of the edge position in pixels.

All functions are shape-agnostic over leading dims and keep gradients only through
probabilities; positions are treated as constants (detached) unless stated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# edge index -> (sign, axis): left/top go backwards from the centre, right/bottom forwards
_SIGN = torch.tensor([-1.0, -1.0, 1.0, 1.0])


def bin_positions(
    ref_points: torch.Tensor,
    bin_values: torch.Tensor,
    reg_scale: torch.Tensor | float,
    image_size: Tuple[float, float],
) -> torch.Tensor:
    """Absolute pixel position of each bin.

    ref_points: [..., 4] normalized cxcywh (D-FINE `ref_points` / ref_points_initial)
    bin_values: [R+1] W(n) from `weighting_function`
    image_size: (W, H) pixels
    returns:    [..., 4, R+1] positions along x (edges 0, 2) or y (edges 1, 3).
    Positions are monotone in n: decreasing for left/top, increasing for right/bottom.
    """
    s = float(abs(reg_scale)) if not torch.is_tensor(reg_scale) else reg_scale.abs().float()
    W, H = image_size
    cx, cy, w, h = ref_points.unbind(-1)
    scale_x, scale_y = W, H
    centre = torch.stack([cx * scale_x, cy * scale_y, cx * scale_x, cy * scale_y], -1)  # [...,4]
    size = torch.stack([w * scale_x, h * scale_y, w * scale_x, h * scale_y], -1)         # [...,4]
    sign = _SIGN.to(ref_points)
    offs = (0.5 * s + bin_values.to(ref_points))  # [R+1]
    pos = centre[..., None] + sign[..., None] * offs * (size / s)[..., None]
    return pos


def cell_edges(pos: torch.Tensor) -> torch.Tensor:
    """Cell boundaries around point-mass bins: midpoints between neighbours, with the
    two outer boundaries mirrored from the first/last gaps. Returned sorted ascending
    regardless of the orientation of `pos`. pos [..., R+1] -> edges [..., R+2]."""
    pos_sorted, _ = torch.sort(pos, dim=-1)
    mid = 0.5 * (pos_sorted[..., 1:] + pos_sorted[..., :-1])
    first = pos_sorted[..., :1] - (mid[..., :1] - pos_sorted[..., :1])
    last = pos_sorted[..., -1:] + (pos_sorted[..., -1:] - mid[..., -1:])
    return torch.cat([first, mid, last], dim=-1)


def _sort_with_probs(pos: torch.Tensor, probs: torch.Tensor):
    order = torch.argsort(pos, dim=-1)
    return torch.gather(pos, -1, order), torch.gather(probs, -1, order)


def cdf_at(
    x: torch.Tensor, edges: torch.Tensor, probs_sorted: torch.Tensor
) -> torch.Tensor:
    """Piecewise-linear CDF (uniform density inside each cell) evaluated at x.
    edges [..., R+2] ascending, probs_sorted [..., R+1] matching cell order, x [..., K]."""
    cum = torch.cumsum(probs_sorted, dim=-1)
    cum = torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1)  # [..., R+2] CDF at edges
    lo, hi = edges[..., :-1], edges[..., 1:]                           # cell bounds [..., R+1]
    width = (hi - lo).clamp_min(1e-12)
    # fraction of each cell covered by x: clamp((x - lo)/width, 0, 1)
    frac = ((x[..., :, None] - lo[..., None, :]) / width[..., None, :]).clamp(0.0, 1.0)  # [...,K,R+1]
    return (frac * probs_sorted[..., None, :]).sum(-1) + 0.0 * cum[..., :1]


def resample(
    src_pos: torch.Tensor,
    src_probs: torch.Tensor,
    dst_pos: torch.Tensor,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transport a distribution given on grid `src_pos` onto grid `dst_pos`.

    The source is modelled as piecewise-uniform density over its cells; the mass that
    falls into each destination cell is obtained from CDF differences. Gradients flow
    through `src_probs` only (positions are detached).

    src_pos, dst_pos: [..., R+1] (same leading dims; destination may have its own
                      orientation — handled by sorting internally)
    src_probs:        [..., R+1]
    returns (dst_probs [..., R+1] in the ORIGINAL order of dst_pos, kept_mass [...])
      kept_mass = fraction of source mass inside the destination support (before
      renormalization). Callers should drop pairs with low kept_mass.
    """
    src_pos = src_pos.detach()
    dst_pos_d = dst_pos.detach()
    sp, spr = _sort_with_probs(src_pos, src_probs)
    s_edges = cell_edges(sp)
    d_order = torch.argsort(dst_pos_d, dim=-1)
    d_edges = cell_edges(dst_pos_d)  # ascending
    cdf = cdf_at(d_edges, s_edges, spr)  # [..., R+2]
    masses_sorted = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)  # [..., R+1] ascending order
    kept = masses_sorted.sum(-1)
    if renormalize:
        masses_sorted = masses_sorted / kept.clamp_min(1e-12)[..., None]
    # undo the destination sort
    inv = torch.empty_like(d_order)
    inv.scatter_(-1, d_order, torch.arange(d_order.shape[-1], device=d_order.device).expand_as(d_order))
    masses = torch.gather(masses_sorted, -1, inv)
    return masses, kept


def wasserstein1(
    pos_a: torch.Tensor, probs_a: torch.Tensor, pos_b: torch.Tensor, probs_b: torch.Tensor
) -> torch.Tensor:
    """1-D Wasserstein distance (pixels) between two point-mass distributions on
    possibly different grids: W1 = ∫ |F_a(x) - F_b(x)| dx over the union of supports.
    Differentiable w.r.t. both probability vectors."""
    pa, pra = _sort_with_probs(pos_a.detach(), probs_a)
    pb, prb = _sort_with_probs(pos_b.detach(), probs_b)
    x = torch.cat([pa, pb], dim=-1)
    x, _ = torch.sort(x, dim=-1)  # [..., 2(R+1)]
    # step CDFs evaluated at union points (right-continuous): F(x) = Σ p_i [pos_i <= x]
    Fa = ((pa[..., None, :] <= x[..., :, None]).to(probs_a) * pra[..., None, :]).sum(-1)
    Fb = ((pb[..., None, :] <= x[..., :, None]).to(probs_b) * prb[..., None, :]).sum(-1)
    dx = x[..., 1:] - x[..., :-1]
    return ((Fa - Fb).abs()[..., :-1] * dx).sum(-1)


def mean_var(pos: torch.Tensor, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance (pixels²) of the edge position under the point-mass view."""
    mean = (probs * pos).sum(-1)
    var = (probs * pos**2).sum(-1) - mean**2
    return mean, var.clamp_min(0.0)


def shift_positions(
    pos: torch.Tensor,
    shift: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    centre: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Motion compensation on absolute edge positions.
    pos   [..., 4, R+1]; shift [..., 2] (dx, dy) in pixels applied to x-edges (0,2) and
    y-edges (1,3); scale [..., 2] (sx, sy) optional, about `centre` [..., 2]."""
    dx, dy = shift[..., 0], shift[..., 1]
    d = torch.stack([dx, dy, dx, dy], -1)[..., None]
    out = pos + d
    if scale is not None:
        assert centre is not None, "scale needs a centre"
        sx, sy = scale[..., 0], scale[..., 1]
        s = torch.stack([sx, sy, sx, sy], -1)[..., None]
        c = torch.stack([centre[..., 0], centre[..., 1], centre[..., 0], centre[..., 1]], -1)[..., None] + d
        out = c + (out - c) * s
    return out
