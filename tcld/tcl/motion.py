"""Motion compensation between frames a and b (P4.2).

Static camera: only objects move. We need, per object, the pixel shift (dx, dy)
that maps its edges in frame a to frame b (optionally a scale). Variants:

* `shift_from_boxes`   — difference of predicted box centres (cheap, but the loss can
                         game it: a drifted box drifts in both frames → zero shift).
                         Kept for ablation only.
* `shift_from_flow`    — median dense optical flow (Farneback, cv2) inside the frame-a
                         box; independent of the detector, so the loss cannot game it.
                         This is the default for TCLD.
* `shift_from_frozen`  — centre displacement of boxes predicted by a frozen copy of
                         the initial model (also un-gameable); provided by the caller
                         as two box sets.

All functions return shifts in the pixel space of the boxes they are given.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def shift_from_boxes(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """[M, 4] xyxy each (already paired) -> [M, 2] centre displacement."""
    ca = (boxes_a[:, :2] + boxes_a[:, 2:]) / 2
    cb = (boxes_b[:, :2] + boxes_b[:, 2:]) / 2
    return cb - ca


def scale_from_boxes(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """[M, 2] (sx, sy) size ratios b / a."""
    wa = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(eps)
    wb = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(eps)
    return wb / wa


def dense_flow(frame_a: np.ndarray, frame_b: np.ndarray, downscale: int = 1) -> np.ndarray:
    """Farneback flow a->b on grayscale frames (HxWx3 uint8 RGB). Returns [H, W, 2]
    float32 in the ORIGINAL pixel units (flow computed at reduced size is rescaled)."""
    import cv2

    ga = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    if downscale > 1:
        ga = cv2.resize(ga, (ga.shape[1] // downscale, ga.shape[0] // downscale), interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, (gb.shape[1] // downscale, gb.shape[0] // downscale), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, pyr_scale=0.5, levels=4, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    if downscale > 1:
        flow = cv2.resize(flow, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_LINEAR) * downscale
    return flow.astype(np.float32)


def shift_from_flow(flow: np.ndarray, boxes_a: torch.Tensor, shrink: float = 0.2,
                    min_pixels: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Median flow inside each (shrunk) box. Returns (shift [M, 2], valid [M]) where
    valid is False for boxes too small / outside the image."""
    H, W = flow.shape[:2]
    out = torch.zeros(boxes_a.shape[0], 2, dtype=torch.float32)
    valid = torch.zeros(boxes_a.shape[0], dtype=torch.bool)
    b = boxes_a.detach().cpu().numpy()
    for i, (x1, y1, x2, y2) in enumerate(b):
        w, h = x2 - x1, y2 - y1
        xs, xe = int(round(x1 + shrink * w)), int(round(x2 - shrink * w))
        ys, ye = int(round(y1 + shrink * h)), int(round(y2 - shrink * h))
        xs, xe = max(xs, 0), min(xe, W)
        ys, ye = max(ys, 0), min(ye, H)
        if xe - xs < 1 or ye - ys < 1 or (xe - xs) * (ye - ys) < min_pixels:
            continue
        patch = flow[ys:ye, xs:xe].reshape(-1, 2)
        med = np.median(patch, axis=0)
        out[i] = torch.from_numpy(med.astype(np.float32))
        valid[i] = True
    return out.to(boxes_a.device), valid.to(boxes_a.device)


def motion_gate(shift: torch.Tensor, min_px: float = 0.0, max_px: Optional[float] = None) -> torch.Tensor:
    """Boolean mask of pairs whose motion magnitude is within [min_px, max_px]."""
    mag = shift.norm(dim=-1)
    ok = mag >= min_px
    if max_px is not None:
        ok &= mag <= max_px
    return ok


class RaftFlow:
    """Dense flow with torchvision RAFT (small by default). Frames HxWx3 uint8 RGB ->
    [H, W, 2] float32 pixels. Weights are cached under $TORCH_HOME (set to /ckpt/torch
    in the container by scripts that use it)."""

    def __init__(self, device: str = "cuda", small: bool = True, iters: int = 12):
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        if small:
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.model = self.model.to(device).eval()
        self.device = device
        self.iters = iters

    @torch.no_grad()
    def __call__(self, frame_a: np.ndarray, frame_b: np.ndarray, max_side: int = 1024) -> np.ndarray:
        import torch.nn.functional as F

        H, W = frame_a.shape[:2]
        scale = min(1.0, max_side / max(H, W))
        h, w = int(round(H * scale / 8)) * 8, int(round(W * scale / 8)) * 8
        a = torch.from_numpy(frame_a).permute(2, 0, 1).float()[None].to(self.device) / 255
        b = torch.from_numpy(frame_b).permute(2, 0, 1).float()[None].to(self.device) / 255
        a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False) * 2 - 1
        b = F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False) * 2 - 1
        flow = self.model(a, b, num_flow_updates=self.iters)[-1]          # [1, 2, h, w]
        flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        flow[:, 0] *= W / w
        flow[:, 1] *= H / h
        return flow[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
