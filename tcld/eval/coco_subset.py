"""Minimal, dependency-light COCO evaluation used by the smoke test and as the
reference implementation for tcld.eval.scenes100_eval (P1.3).

Pipeline mirrors upstream val: PIL -> Resize(input_size) -> float tensor [0,1];
postprocessor rescales boxes to original (w, h); COCO category ids via
remap_mscoco_category (set in the upstream config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tcld.model.base import DistributionalDetector


class CocoImages(Dataset):
    def __init__(self, img_dir: Path, ann_file: Path, input_size: Tuple[int, int]):
        from pycocotools.coco import COCO

        self.coco = COCO(str(ann_file))
        self.ids: List[int] = sorted(self.coco.getImgIds())
        self.img_dir = Path(img_dir)
        self.input_size = input_size  # (h, w)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        info = self.coco.loadImgs(self.ids[i])[0]
        im = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        w, h = im.size
        im = TF.resize(im, list(self.input_size))
        x = TF.to_tensor(im)
        return x, torch.tensor([w, h]), info["id"]


def _collate(batch):
    xs, sizes, ids = zip(*batch)
    return torch.stack(xs), torch.stack(sizes), list(ids)


def load_image_batch(img_dir: Path, ann_file: Path, n: int, input_size=(640, 640)):
    ds = CocoImages(img_dir, ann_file, input_size)
    items = [ds[i] for i in range(min(n, len(ds)))]
    xs, sizes, _ = _collate(items)
    return xs, sizes


@torch.no_grad()
def evaluate_coco_subset(
    model: DistributionalDetector,
    img_dir: Path,
    ann_file: Path,
    device: str = "cuda",
    input_size: Tuple[int, int] = (640, 640),
    batch_size: int = 8,
    num_workers: int = 4,
    score_thresh: float = 0.0,
) -> Dict[str, float]:
    from pycocotools.cocoeval import COCOeval

    ds = CocoImages(img_dir, ann_file, input_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)
    dets = []
    for xs, sizes, ids in dl:
        res = model.predict(xs.to(device), sizes.to(device))
        for r, img_id in zip(res, ids):
            boxes = r["boxes"].cpu()
            scores = r["scores"].cpu()
            labels = r["labels"].cpu()
            keep = scores > score_thresh
            for b, s, lab in zip(boxes[keep], scores[keep], labels[keep]):
                x1, y1, x2, y2 = b.tolist()
                dets.append({
                    "image_id": int(img_id),
                    "category_id": int(lab),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                })
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "num_images": len(ds)}
    coco_dt = ds.coco.loadRes(dets)
    ev = COCOeval(ds.coco, coco_dt, "bbox")
    ev.params.imgIds = ds.ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {
        "AP": float(s[0]) * 100, "AP50": float(s[1]) * 100, "AP75": float(s[2]) * 100,
        "APs": float(s[3]) * 100, "APm": float(s[4]) * 100, "APl": float(s[5]) * 100,
        "num_images": len(ds), "num_dets": len(dets),
    }
