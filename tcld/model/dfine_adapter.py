"""Thin wrapper around the pinned D-FINE implementation.

What it adds on top of upstream:
* build model + postprocessor from a YAML config and load a checkpoint without
  touching upstream code;
* `distribution_mode()` — a context in which the model returns `pred_corners`,
  `ref_points`, `up`, `reg_scale` (upstream only returns them in train mode) while
  BatchNorm stays in eval and denoising (which requires GT targets) is disabled;
* `forward_distributions()` — DistOutput with optional per-layer aux outputs;
* helpers to freeze parts of the network and to get the bin weighting W(n).

Upstream facts this relies on (external/D-FINE/src/zoo/dfine/dfine_decoder.py):
* train-mode output dict has keys pred_logits, pred_boxes, pred_corners, ref_points,
  up, reg_scale, aux_outputs (list, one per shallower layer).
* `ref_points` is `ref_points_initial` — the same anchor for every decoder layer.
* `decoder.num_denoising > 0` with `targets=None` crashes in train mode.
* Every decoder layer's box = distance2bbox(ref_points, Integral(corners, W), reg_scale).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator, List, Optional, Union

import torch
import torch.nn as nn

from tcld.dfine_import import ensure_dfine_importable
from tcld.model.base import DistOutput, DistributionalDetector
from tcld.paths import ckpt_dir, model_config

ensure_dfine_importable()

from src.core import YAMLConfig  # noqa: E402
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function  # noqa: E402

PathLike = Union[str, Path]


class DFINEAdapter(nn.Module, DistributionalDetector):
    def __init__(self, config_path: PathLike, checkpoint: Optional[PathLike] = None):
        super().__init__()
        cfg = YAMLConfig(str(config_path))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        self.cfg = cfg
        self.model: nn.Module = cfg.model
        self.postprocessor: nn.Module = cfg.postprocessor
        self.tcld_meta: dict = dict(cfg.yaml_cfg.get("tcld", {}))
        self.checkpoint_path: Optional[Path] = None
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    # ------------------------------------------------------------------ build
    def load_checkpoint(self, path: PathLike) -> None:
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "ema" in state:
            state = state["ema"]["module"]
        elif "model" in state:
            state = state["model"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint {path} does not match the model: "
                f"missing={missing[:5]}... unexpected={unexpected[:5]}..."
            )
        self.checkpoint_path = path

    @property
    def decoder(self) -> nn.Module:
        return self.model.decoder

    @property
    def reg_max(self) -> int:
        return int(self.decoder.reg_max)

    @property
    def num_classes(self) -> int:
        return int(self.decoder.num_classes)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # --------------------------------------------------------------- freezing
    def freeze(self, parts: Optional[List[str]] = None) -> None:
        """parts ⊆ {backbone, encoder, decoder}. Frozen params get requires_grad=False."""
        for part in parts or []:
            module = getattr(self.model, part)
            for p in module.parameters():
                p.requires_grad = False

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.model.parameters() if p.requires_grad)

    def set_norm_eval(self) -> None:
        """Keep BatchNorm running stats fixed (relevant for D-FINE-S whose backbone
        BN is not frozen upstream). LayerNorm has no train/eval difference."""
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    # ------------------------------------------------------- distribution mode
    @contextlib.contextmanager
    def distribution_mode(self, denoising: bool = False):
        """Model in train() so the decoder emits corner distributions, but with
        BatchNorm in eval and denoising disabled unless explicitly requested."""
        was_training = self.model.training
        prev_dn = self.decoder.num_denoising
        self.model.train()
        self.set_norm_eval()
        if not denoising:
            self.decoder.num_denoising = 0
        try:
            yield self
        finally:
            self.decoder.num_denoising = prev_dn
            self.model.train(was_training)
            if was_training:
                self.set_norm_eval()

    def forward_distributions(
        self,
        images: torch.Tensor,
        all_layers: bool = False,
        targets: Optional[list] = None,
    ) -> DistOutput:
        """Run the detector and return edge distributions of the final decoder
        layer (and shallower layers if `all_layers`). Gradients flow unless the
        caller wraps this in torch.no_grad()."""
        with self.distribution_mode(denoising=targets is not None):
            out = self.model(images, targets=targets)
        return self._to_dist_output(out, all_layers=all_layers)

    def _to_dist_output(self, out: dict, all_layers: bool) -> DistOutput:
        R = self.reg_max
        B, Q = out["pred_corners"].shape[:2]
        main = DistOutput(
            logits=out["pred_logits"],
            boxes=out["pred_boxes"],
            corners=out["pred_corners"].reshape(B, Q, 4, R + 1),
            ref_points=out["ref_points"],
            up=out["up"],
            reg_scale=out["reg_scale"],
        )
        if all_layers:
            for aux in out.get("aux_outputs", []):
                main.aux.append(
                    DistOutput(
                        logits=aux["pred_logits"],
                        boxes=aux["pred_boxes"],
                        corners=aux["pred_corners"].reshape(B, Q, 4, R + 1),
                        ref_points=aux["ref_points"],
                        up=out["up"],
                        reg_scale=out["reg_scale"],
                    )
                )
        return main

    # ------------------------------------------------------------- inference
    @contextlib.contextmanager
    def _params_frozen(self):
        """Temporarily set requires_grad=False on every parameter.

        Why: nn.MultiheadAttention takes its fused "fast path" only when no
        parameter requires grad (upstream evaluates an EMA copy whose params are
        all frozen). The fused and unfused kernels differ by ~1e-6, which the
        decoder's top-k query selection amplifies into different detections —
        measured as 0.25 AP on a 50-image COCO subset. Freezing during predict()
        makes our numbers bit-identical to upstream `train.py --test-only`.
        """
        grads = [p for p in self.model.parameters() if p.requires_grad]
        for p in grads:
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p in grads:
                p.requires_grad_(True)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, orig_sizes: torch.Tensor) -> List[dict]:
        """Standard eval-mode inference. `orig_sizes` is [B, 2] (w, h) in pixels.
        Returns a list of {labels, boxes(xyxy, px), scores} with COCO category ids
        when the postprocessor has remap_mscoco_category=True.
        Input must be exactly `eval_spatial_size` (640x640 upstream): eval mode uses
        anchors / positional embeddings cached for that size."""
        self.model.eval()
        with self._params_frozen():
            out = self.model(images)
        return self.postprocessor(out, orig_sizes)

    # ------------------------------------------------------------ bin geometry
    def bin_values(self) -> torch.Tensor:
        """W(n), shape [R+1], in normalized distance units (see distance2bbox)."""
        return weighting_function(self.reg_max, self.decoder.up, self.decoder.reg_scale)

    def boxes_from_corners(self, corners: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
        """Recompute normalized cxcywh boxes from corner logits — must equal
        `DistOutput.boxes` (used as a self-check in tests and smoke)."""
        probs = torch.softmax(corners, dim=-1)
        dist = (probs * self.bin_values().to(probs)).sum(-1)  # [B, Q, 4]
        return distance2bbox(ref_points, dist, self.decoder.reg_scale)


def load_dfine(
    model: str = "dfine_s",
    checkpoint: Optional[PathLike] = None,
    device: Union[str, torch.device] = "cpu",
    strict_checkpoint: bool = True,
) -> DFINEAdapter:
    """Build an adapter from configs/model/<model>.yml. If `checkpoint` is None, the
    checkpoint named in the config's `tcld.checkpoint` is looked up in $TCLD_CKPT.
    With strict_checkpoint=False a missing file yields a randomly initialized model
    (used by CPU unit tests)."""
    cfg_path = model_config(model)
    adapter = DFINEAdapter(cfg_path)
    if checkpoint is None:
        name = adapter.tcld_meta.get("checkpoint")
        checkpoint = ckpt_dir() / name if name else None
    if checkpoint is not None and Path(checkpoint).exists():
        adapter.load_checkpoint(checkpoint)
    elif strict_checkpoint:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run `make download-ckpt` "
            f"(downloads into {ckpt_dir()})."
        )
    adapter.to(device)
    adapter.model.eval()
    return adapter
