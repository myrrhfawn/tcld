"""Sharpness of an edge distribution in *pixel* units (P3.2).

Shannon entropy over bins is NOT comparable between queries: D-FINE bins are
non-uniform (dense near the anchor edge, coarse in the tails) and their pixel width
scales with the anchor box size. Use the differential entropy of the
piecewise-uniform density instead:

    H_diff = H_bins + Σ_n p_n log(width_n)     [nats, width in pixels]

or the standard deviation of the edge position in pixels (`grid.mean_var`).
"""

from __future__ import annotations

import torch

from tcld.tcl.grid import cell_edges, mean_var


def shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(probs * probs.clamp_min(eps).log()).sum(-1)


def cell_widths(pos: torch.Tensor) -> torch.Tensor:
    """Width (pixels) of the cell of each bin, in the ORIGINAL bin order of `pos`."""
    order = torch.argsort(pos, dim=-1)
    edges = cell_edges(pos)
    w_sorted = edges[..., 1:] - edges[..., :-1]
    inv = torch.empty_like(order)
    inv.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
    return torch.gather(w_sorted, -1, inv)


def differential_entropy(pos: torch.Tensor, probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """H_diff in nats for pixel-scale cells. [..., R+1] -> [...]."""
    w = cell_widths(pos.detach()).clamp_min(eps)
    return shannon_entropy(probs, eps) + (probs * w.log()).sum(-1)


def pixel_std(pos: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    _, var = mean_var(pos.detach(), probs)
    return var.sqrt()


def max_prob(probs: torch.Tensor) -> torch.Tensor:
    return probs.max(-1).values


def sharper_mask(
    h_a: torch.Tensor, h_b: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    """True where distribution A is sharper (lower H_diff) than B by at least `margin`."""
    return h_a < (h_b - margin)
