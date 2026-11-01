"""COCO train2017 replay subset (P1.6): N images that contain at least one
person / car / bus / truck, with their full 80-class annotations, plus a
COCO-minival-500 subset of val2017 for sanity checks.

Layout under $TCLD_DATA/coco:
  annotations/instances_train2017.json             (full, kept for re-sampling)
  annotations/instances_train2017_replay<N>.json   (subset, full COCO categories)
  annotations/instances_val2017_minival500.json
  train2017/<file>.jpg, val2017/<file>.jpg          (subset images only)

Usage: python scripts/download_coco_replay.py [--n 2000] [--minival 500] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from scripts.download_coco_subset import ANN_ZIP, download
from tcld.paths import data_dir

IMG_BASE = {"train2017": "http://images.cocodataset.org/train2017/", "val2017": "http://images.cocodataset.org/val2017/"}
TARGET_COCO_CATS = {1, 3, 6, 8}  # person, car, bus, truck


def ensure_annotations(root: Path) -> None:
    ann_dir = root / "annotations"
    need = [ann_dir / "instances_train2017.json", ann_dir / "instances_val2017.json"]
    if all(p.exists() for p in need):
        return
    zip_path = root / "annotations_trainval2017.zip"
    download(ANN_ZIP, zip_path, "annotations zip")
    with zipfile.ZipFile(zip_path) as z:
        for p in need:
            if not p.exists():
                z.extract(f"annotations/{p.name}", root)
    zip_path.unlink()


def subset(coco: dict, n: int, seed: int, require_target: bool) -> dict:
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    imgs = sorted(coco["images"], key=lambda im: im["id"])
    if require_target:
        imgs = [im for im in imgs if any(a["category_id"] in TARGET_COCO_CATS and not a.get("iscrowd", 0) for a in by_img.get(im["id"], []))]
    else:
        imgs = [im for im in imgs if im["id"] in by_img]
    chosen = random.Random(seed).sample(imgs, min(n, len(imgs)))
    ids = {im["id"] for im in chosen}
    return {
        "info": coco.get("info", {}), "licenses": coco.get("licenses", []),
        "categories": coco["categories"], "images": chosen,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
    }


def fetch_images(split: str, images: list, root: Path, workers: int = 8) -> None:
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [im for im in images if not (out_dir / im["file_name"]).exists()]
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(lambda im: download(IMG_BASE[split] + im["file_name"], out_dir / im["file_name"]), todo),
                  total=len(todo), desc=f"{split} images"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--minival", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = data_dir() / "coco"
    ensure_annotations(root)

    train = json.load(open(root / "annotations" / "instances_train2017.json"))
    rep = subset(train, args.n, args.seed, require_target=True)
    out = root / "annotations" / f"instances_train2017_replay{args.n}.json"
    json.dump(rep, open(out, "w"))
    print(f"replay: {len(rep['images'])} images, {len(rep['annotations'])} boxes -> {out}")
    fetch_images("train2017", rep["images"], root)

    val = json.load(open(root / "annotations" / "instances_val2017.json"))
    mv = subset(val, args.minival, args.seed, require_target=False)
    out = root / "annotations" / f"instances_val2017_minival{args.minival}.json"
    json.dump(mv, open(out, "w"))
    print(f"minival: {len(mv['images'])} images, {len(mv['annotations'])} boxes -> {out}")
    fetch_images("val2017", mv["images"], root)


if __name__ == "__main__":
    main()
