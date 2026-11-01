"""Test-time adaptation baselines adapted to D-FINE (P5, "modern methods").

Tent (Wang et al., ICLR 2021): minimize the prediction entropy on unlabeled target frames,
updating only the affine parameters of normalization layers (LayerNorm / BatchNorm) inside
the arm's trainable parts. D-FINE has sigmoid class scores, so the entropy is the binary
entropy of each candidate query's top-class score (queries with score >= score_min).

ActMAD (Mirza et al., CVPR 2023): align the first two moments of intermediate activations
on target batches to moments stored from source (COCO) images, L1 loss, all trainable
parameters updated. Spatial maps (encoder outputs) are aligned location-wise as in the
paper; decoder hidden states have no fixed spatial layout (top-k queries), so they are
aligned channel-wise over queries. Only layers inside the trainable parts get a loss term
that can change parameters; layers before them still contribute nothing (no gradient).
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from tcld.model.base import DistOutput
from tcld.model.dfine_adapter import DFINEAdapter

NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)


def restrict_to_norm_params(adapter: DFINEAdapter) -> int:
    """Keep requires_grad only on affine params of normalization layers (Tent). Returns #params."""
    keep = set()
    for m in adapter.model.modules():
        if isinstance(m, NORM_TYPES):
            for p in m.parameters(recurse=False):
                keep.add(id(p))
    n = 0
    for p in adapter.model.parameters():
        if p.requires_grad and id(p) not in keep:
            p.requires_grad_(False)
        n += p.numel() if p.requires_grad else 0
    return n


def tent_loss(out: DistOutput, score_min: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    s = out.logits.float().sigmoid().max(-1).values           # [B, Q]
    mask = s >= score_min
    s = s.clamp(1e-6, 1 - 1e-6)
    h = -(s * s.log() + (1 - s) * (1 - s).log())
    n = int(mask.sum())
    loss = (h * mask).sum() / max(n, 1)
    return loss, {"loss_total": float(loss.detach()), "n_matched": float(n)}


class ActivationTap:
    """Forward hooks on the encoder outputs (spatial, location-wise moments) and on every
    decoder layer (hidden states [B, Q, D], channel-wise moments)."""

    def __init__(self, adapter: DFINEAdapter):
        self.acts: Dict[str, torch.Tensor] = {}
        m = adapter.model
        self.handles = [m.encoder.register_forward_hook(self._enc)]
        for k, layer in enumerate(m.decoder.decoder.layers):
            self.handles.append(layer.register_forward_hook(lambda mod, i, o, k=k: self._dec(k, o)))

    def _enc(self, mod, inp, out):
        for j, t in enumerate(out):
            self.acts[f"enc{j}"] = t

    def _dec(self, k, out):
        self.acts[f"dec{k}"] = out[0] if isinstance(out, tuple) else out

    def clear(self):
        self.acts = {}

    def remove(self):
        for h in self.handles:
            h.remove()

    @staticmethod
    def moments(name: str, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t.float()
        if name.startswith("enc"):          # [B, C, H, W] -> location-wise over batch
            return t.mean(0), t.var(0, unbiased=False)
        x = t.reshape(-1, t.shape[-1])      # [B*Q, D] -> channel-wise
        return x.mean(0), x.var(0, unbiased=False)


def _coco_images(n: int, input_size, seed: int = 0) -> List[torch.Tensor]:
    root = os.environ.get("TCLD_DATA", "data")
    files = sorted(glob.glob(f"{root}/coco/val2017/*.jpg"))
    if not files:
        raise FileNotFoundError(f"no COCO val2017 images under {root}/coco/val2017")
    random.Random(seed).shuffle(files)
    imgs = []
    for f in files[:n]:
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (input_size[0], input_size[1]), interpolation=cv2.INTER_LINEAR)
        imgs.append(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0)
    return imgs


@torch.no_grad()
def source_moments(adapter: DFINEAdapter, input_size, n_images: int = 512, batch: int = 8,
                   cache: Path | None = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Mean / variance of tapped activations over COCO val2017 images (same forward mode as
    adaptation: train mode, BatchNorm eval, no denoising). Cached on disk per model/size."""
    if cache is not None and cache.exists():
        d = torch.load(cache, map_location=adapter.device)
        return {k: (v[0], v[1]) for k, v in d.items()}
    imgs = _coco_images(n_images, input_size)
    tap = ActivationTap(adapter)
    s1: Dict[str, torch.Tensor] = {}
    s2: Dict[str, torch.Tensor] = {}
    cnt: Dict[str, int] = {}
    for i in range(0, len(imgs), batch):
        x = torch.stack(imgs[i:i + batch]).to(adapter.device)
        tap.clear()
        adapter.forward_distributions(x)
        for k, t in tap.acts.items():
            t = t.float()
            if k.startswith("enc"):
                w, a, b = t.shape[0], t.sum(0), (t * t).sum(0)
            else:
                x2 = t.reshape(-1, t.shape[-1])
                w, a, b = x2.shape[0], x2.sum(0), (x2 * x2).sum(0)
            s1[k] = s1.get(k, 0) + a
            s2[k] = s2.get(k, 0) + b
            cnt[k] = cnt.get(k, 0) + w
    tap.remove()
    out = {}
    for k in s1:
        mu = s1[k] / cnt[k]
        out[k] = (mu, (s2[k] / cnt[k] - mu * mu).clamp_min(0))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: torch.stack([v[0], v[1]]) for k, v in out.items()}, cache)
    return out


class ActMADCriterion:
    def __init__(self, adapter: DFINEAdapter, src: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        self.adapter = adapter
        self.src = src
        self.tap = ActivationTap(adapter)

    def __call__(self, images: torch.Tensor, amp: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.tap.clear()
        with torch.autocast("cuda", enabled=amp):
            self.adapter.forward_distributions(images)
        loss = torch.zeros((), device=images.device)
        stats = {}
        for k, t in self.tap.acts.items():
            if not t.requires_grad:
                continue
            mu, var = ActivationTap.moments(k, t)
            l = (mu - self.src[k][0]).abs().mean() + (var - self.src[k][1]).abs().mean()
            loss = loss + l
            stats[f"actmad_{k}"] = float(l.detach())
        stats.update({"loss_total": float(loss.detach()), "n_matched": float(len(stats))})
        return loss, stats
