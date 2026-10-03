"""Download COCO val2017 annotations and a small fixed image subset for smoke tests.

Layout under $TCLD_DATA/coco:
  annotations/instances_val2017.json          (full, ~450 MB unpacked)
  annotations/instances_val2017_smoke50.json  (50 images, deterministic)
  val2017/<id>.jpg                            (only the subset images)

Usage: python scripts/download_coco_subset.py [--n 50] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from tcld.paths import data_dir

ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_BASE = "http://images.cocodataset.org/val2017/"


def download(url: str, dst: Path, desc: str | None = None) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dst.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = data_dir() / "coco"
    ann_dir = root / "annotations"
    full_ann = ann_dir / "instances_val2017.json"

    if not full_ann.exists():
        zip_path = root / "annotations_trainval2017.zip"
        download(ANN_ZIP, zip_path, "annotations zip")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("annotations/instances_val2017.json", root)
        zip_path.unlink()

    subset_ann = ann_dir / f"instances_val2017_smoke{args.n}.json"
    with open(full_ann) as f:
        coco = json.load(f)

    rng = random.Random(args.seed)
    images = sorted(coco["images"], key=lambda im: im["id"])
    # prefer images that actually have annotations
    with_ann = {a["image_id"] for a in coco["annotations"]}
    images = [im for im in images if im["id"] in with_ann]
    chosen = rng.sample(images, args.n)
    chosen_ids = {im["id"] for im in chosen}
    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": chosen,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in chosen_ids],
    }
    with open(subset_ann, "w") as f:
        json.dump(subset, f)
    print(f"wrote {subset_ann} ({len(chosen)} images, {len(subset['annotations'])} boxes)")

    img_dir = root / "val2017"
    for im in tqdm(chosen, desc="images"):
        download(IMG_BASE + im["file_name"], img_dir / im["file_name"])
    print(f"images in {img_dir}")


if __name__ == "__main__":
    main()
