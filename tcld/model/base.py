"""Detector-agnostic interface for detectors that predict discrete edge distributions.

D-FINE is the first implementation (tcld/model/dfine_adapter.py). GFL/DEIM/... can
plug in later (ai-plan.md P7.2) by implementing the same contract.

Conventions
-----------
* Boxes are normalized cxcywh in [0, 1] relative to the network input.
* `corners` are raw logits over bins: [B, Q, 4, R+1] for edges (left, top, right, bottom).
* `ref_points` are normalized cxcywh anchors the bins are defined against: [B, Q, 4].
* Absolute position of bin n for an edge is detector-specific; the adapter exposes
  `bin_edges_abs(...)` so loss code never hard-codes the weighting function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import torch


@dataclass
class DistOutput:
    logits: torch.Tensor          # [B, Q, C]
    boxes: torch.Tensor           # [B, Q, 4] normalized cxcywh
    corners: torch.Tensor         # [B, Q, 4, R+1] logits over bins
    ref_points: torch.Tensor      # [B, Q, 4] normalized cxcywh (initial reference)
    up: torch.Tensor              # scalar tensor
    reg_scale: torch.Tensor       # scalar tensor
    aux: List["DistOutput"] = field(default_factory=list)  # shallower decoder layers, if requested

    @property
    def reg_max(self) -> int:
        return self.corners.shape[-1] - 1

    def probs(self) -> torch.Tensor:
        return torch.softmax(self.corners, dim=-1)


class DistributionalDetector:
    """Minimal contract used by tcld.tcl and tcld.train."""

    def forward_distributions(
        self, images: torch.Tensor, all_layers: bool = False
    ) -> DistOutput:  # pragma: no cover - interface
        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, orig_sizes: torch.Tensor
    ) -> List[dict]:  # pragma: no cover - interface
        """Post-processed detections in original image pixels: list of
        {labels, boxes (xyxy), scores}."""
        raise NotImplementedError

    def bin_values(self) -> torch.Tensor:  # pragma: no cover - interface
        """W(n): [R+1] bin values in the detector's normalized edge-distance units."""
        raise NotImplementedError

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:  # pragma: no cover
        raise NotImplementedError

    def freeze(self, parts: Optional[List[str]] = None) -> None:  # pragma: no cover
        raise NotImplementedError
