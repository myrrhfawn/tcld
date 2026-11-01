"""P5.1 baseline: self-training with hard pseudo-boxes (ST).

Pseudo-labels: confident detections (score >= thresh, class-aware NMS) of a teacher
(frozen initial model by default, or the EMA model) on each unlabeled frame,
converted to D-FINE targets {labels: logit indices, boxes: normalized cxcywh}.
The student is trained with the upstream DFINECriterion (VFL + L1/GIoU + FGL + DDF)
with denoising queries enabled, exactly like supervised fine-tuning. Both frames
of a pair are used as independent images (no temporal term).

Mean-Teacher variant (`student_aug` set, teacher = EMA): the teacher labels the clean
view, the student is trained on a strongly augmented view (photometric jitter + random
horizontal flip, pseudo-boxes flipped accordingly), as in Mean Teacher / Unbiased Teacher.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import random

import torch
import torchvision

from tcld.data.pairs import apply_photometric, photometric_params

from tcld.model.base import DistOutput
from tcld.model.dfine_adapter import DFINEAdapter


@torch.no_grad()
def pseudo_targets(
    teacher_out: DistOutput, score_thresh: float = 0.5, nms_iou: float = 0.7,
    restrict_labels: Optional[List[int]] = None, max_boxes: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    """One target dict per image. Labels are D-FINE logit indices (0..79)."""
    targets = []
    probs = teacher_out.logits.sigmoid()                      # [B, Q, C]
    for b in range(probs.shape[0]):
        p = probs[b]
        if restrict_labels is not None:
            mask = torch.zeros(p.shape[-1], dtype=torch.bool, device=p.device)
            mask[restrict_labels] = True
            p = torch.where(mask[None, :], p, torch.zeros_like(p))
        score, label = p.max(-1)
        keep = torch.nonzero(score >= score_thresh).flatten()
        boxes = teacher_out.boxes[b][keep]
        xyxy = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
        k = torchvision.ops.batched_nms(xyxy, score[keep], label[keep], nms_iou)[:max_boxes]
        targets.append({"labels": label[keep][k].long(), "boxes": boxes[k].float()})
    return targets


def augment_student_views(
    images: torch.Tensor, targets: List[Dict[str, torch.Tensor]], rng: random.Random,
    photometric: float = 0.0, hflip: bool = False,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Strong view for the student: per-image photometric jitter and random horizontal flip.
    Pseudo-boxes are normalized cxcywh, so a flip maps cx -> 1 - cx. Returns new tensors."""
    out_imgs, out_tgts = [], []
    for img, tgt in zip(images, targets):
        x = img
        boxes = tgt["boxes"]
        if photometric > 0:
            x = apply_photometric(x.float().clamp(0, 1), photometric_params(rng, photometric)).to(img.dtype)
        if hflip and rng.random() < 0.5:
            x = x.flip(-1)
            if boxes.numel():
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]
        out_imgs.append(x)
        out_tgts.append({"labels": tgt["labels"], "boxes": boxes})
    return torch.stack(out_imgs), out_tgts


class SelfTrainCriterion:
    """Wraps upstream DFINECriterion; call with the student adapter, images and teacher output."""

    def __init__(self, adapter: DFINEAdapter, score_thresh: float = 0.5, nms_iou: float = 0.7,
                 restrict_to_target_classes: bool = False, student_aug: Optional[Dict[str, float]] = None,
                 seed: int = 0):
        self.adapter = adapter
        self.criterion = adapter.cfg.criterion.to(adapter.device)
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.student_aug = student_aug or {}
        self.rng = random.Random(seed * 7_919 + 17)
        self.restrict: Optional[List[int]] = None
        if restrict_to_target_classes:
            from src.data.dataset import mscoco_label2category

            from tcld.data.common import COCO_TO_TARGET

            self.restrict = [i for i, c in mscoco_label2category.items() if c in COCO_TO_TARGET]

    def __call__(self, images: torch.Tensor, teacher_out: DistOutput, amp: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
        targets = pseudo_targets(teacher_out, self.score_thresh, self.nms_iou, self.restrict)
        n_pl = sum(len(t["labels"]) for t in targets)
        dev = images.device
        if n_pl == 0:
            return torch.zeros((), device=dev, requires_grad=True), {"n_pseudo": 0.0, "loss_total": 0.0}
        if self.student_aug:
            images, targets = augment_student_views(images, targets, self.rng,
                                                    float(self.student_aug.get("photometric", 0.0)),
                                                    bool(self.student_aug.get("hflip", False)))
        # train-mode forward WITH denoising (targets available) — the upstream path
        self.adapter.model.train()
        self.adapter.set_norm_eval()
        with torch.autocast("cuda", enabled=amp):
            out = self.adapter.model(images, targets=targets)
        with torch.autocast("cuda", enabled=False):
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            for key in ("aux_outputs", "dn_outputs", "enc_aux_outputs"):
                if key in out:
                    out[key] = [{k: (v.float() if torch.is_tensor(v) else v) for k, v in d.items()} for d in out[key]]
            for key in ("pre_outputs", "dn_pre_outputs"):
                if key in out:
                    out[key] = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out[key].items()}
            loss_dict = self.criterion(out, targets)
        loss = sum(loss_dict.values())
        stats = {f"st_{k}": float(v.detach()) for k, v in loss_dict.items()}
        stats.update({"n_pseudo": float(n_pl), "loss_total": float(loss.detach())})
        return loss, stats
