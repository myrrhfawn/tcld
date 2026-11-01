"""AdaBN-style recalibration of BatchNorm statistics on frames of the target camera.

No labels and no gradients: the running mean / variance of the BatchNorm layers in the
chosen parts (backbone, encoder) are re-estimated from target-domain images. Used as an
optional first step of per-camera adaptation. The frozen anchor model receives the same
statistics so that the student and the anchor select the same top-k queries (the
index-aligned anchor relies on identical features).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from tcld.model.dfine_adapter import DFINEAdapter

_BN = nn.modules.batchnorm._BatchNorm
DEFAULT_PARTS: Sequence[str] = ("backbone", "encoder")


def bn_modules(adapter: DFINEAdapter, parts: Sequence[str] = DEFAULT_PARTS) -> List[_BN]:
    out: List[_BN] = []
    for part in parts:
        out += [m for m in getattr(adapter.model, part).modules() if isinstance(m, _BN)]
    return out


@torch.no_grad()
def recalibrate(adapter: DFINEAdapter, batches: Iterable[torch.Tensor], parts: Sequence[str] = DEFAULT_PARTS,
                max_batches: int = 10 ** 9, prior: float = 0.0) -> int:
    """Re-estimate the BatchNorm running statistics of `parts` as the cumulative average
    over `batches` (image tensors [B, 3, H, W]) and blend them with the source statistics:
    stat = prior * source + (1 - prior) * target. prior = 0 replaces them entirely (plain
    AdaBN), which breaks a single-scene detector; see FINDINGS 2026-10-08. Returns the
    number of batches used. Train/eval state of the model is restored afterwards."""
    bns = bn_modules(adapter, parts)
    was_training = adapter.model.training
    adapter.model.eval()
    saved = [(m.momentum, m.running_mean.clone(), m.running_var.clone()) for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative moving average
        m.train()
    n = 0
    for imgs in batches:
        adapter.model(imgs)
        n += 1
        if n >= max_batches:
            break
    for m, (mom, mu, var) in zip(bns, saved):
        m.momentum = mom
        m.eval()
        if prior > 0:
            m.running_mean.mul_(1 - prior).add_(mu, alpha=prior)
            m.running_var.mul_(1 - prior).add_(var, alpha=prior)
    adapter.model.train(was_training)
    if was_training:
        adapter.set_norm_eval()
    return n


@torch.no_grad()
def copy_bn_stats(src: DFINEAdapter, dst: DFINEAdapter, parts: Sequence[str] = DEFAULT_PARTS) -> int:
    """Copy running statistics of every BatchNorm in `parts` from src to dst (same architecture)."""
    a, b = bn_modules(src, parts), bn_modules(dst, parts)
    assert len(a) == len(b), (len(a), len(b))
    for ma, mb in zip(a, b):
        mb.running_mean.copy_(ma.running_mean)
        mb.running_var.copy_(ma.running_var)
        if ma.num_batches_tracked is not None and mb.num_batches_tracked is not None:
            mb.num_batches_tracked.copy_(ma.num_batches_tracked)
    return len(a)
