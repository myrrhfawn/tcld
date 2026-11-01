"""VIRAT Video Dataset Release 2.0, ground cameras (data.kitware.com, collection VIRAT).

Layout under $TCLD_DATA/virat:
  ann/annotations/<clip>.viratdata.objects.txt   (object file: oid dur frame x y w h type)
  videos/**/<clip>.mp4

Clip id: VIRAT_S_SSSSCC_NN_start_end — the first 4 digits SSSS identify the fixed
camera / scene (11 scenes: 0000 0001 0002 0100 0101 0102 0400 0401 0500 0502 0503).
Object types: 1 person, 2 car, 3 vehicles (other), 4 object (carried), 5 bike.
Mapping to the 2-class target space: 1 -> person, {2, 3} -> vehicle, {4, 5} -> ignore.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tcld.data.common import FrameGT, VideoClip
from tcld.paths import data_dir

VIRAT_TYPE_TO_TARGET: Dict[int, int] = {1: 0, 2: 1, 3: 1}
VIRAT_IGNORE_TYPES = {4, 5}
# release-2 clips: VIRAT_S_SSSSCC_NN_start_end; a few release-1 clips: VIRAT_S_SSSSCC
_CLIP_RE = re.compile(r"VIRAT_S_(\d{4})(\d{2})(?:_(\d{2})_(\d{6})_(\d{6}))?")


def virat_root() -> Path:
    return data_dir() / "virat"


def parse_clip_id(clip_id: str) -> Dict[str, str]:
    m = _CLIP_RE.fullmatch(clip_id)
    if not m:
        raise ValueError(f"not a VIRAT clip id: {clip_id}")
    scene, group, seg, start, end = m.groups()
    return {"scene": scene, "group": group, "segment": seg or "", "start": start or "", "end": end or ""}


def load_objects(ann_path: Path) -> np.ndarray:
    """Object file -> int64 array [M, 8]: oid, duration, frame, x, y, w, h, type.
    A few release-2 files contain lines where several records were glued together
    (16, 32, ... 3896 columns) or truncated; glued lines are split back into records,
    anything else is dropped."""
    rows = []
    dropped = 0
    with open(ann_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) % 8 == 0:
                try:
                    vals = [int(float(v)) for v in parts]
                except ValueError:
                    dropped += 1
                    continue
                rows.extend(vals[i:i + 8] for i in range(0, len(vals), 8))
            else:
                dropped += 1
    if dropped:
        import warnings

        warnings.warn(f"{ann_path.name}: dropped {dropped} malformed line(s)")
    if not rows:
        return np.zeros((0, 8), np.int64)
    return np.asarray(rows, dtype=np.int64)


def objects_to_gt(arr: np.ndarray) -> Dict[int, FrameGT]:
    """Dense per-frame GT from an object array (frame indices are 0-based in VIRAT)."""
    gt: Dict[int, FrameGT] = {}
    if arr.shape[0] == 0:
        return gt
    order = np.argsort(arr[:, 2], kind="stable")
    arr = arr[order]
    frames, starts = np.unique(arr[:, 2], return_index=True)
    ends = list(starts[1:]) + [arr.shape[0]]
    for f, s, e in zip(frames.tolist(), starts.tolist(), ends):
        rows = arr[s:e]
        xyxy = np.stack([rows[:, 3], rows[:, 4], rows[:, 3] + rows[:, 5], rows[:, 4] + rows[:, 6]], 1).astype(np.float32)
        types = rows[:, 7]
        keep = np.isin(types, list(VIRAT_TYPE_TO_TARGET))
        ign = np.isin(types, list(VIRAT_IGNORE_TYPES))
        labels = np.array([VIRAT_TYPE_TO_TARGET[int(t)] for t in types[keep]], np.int64)
        gt[f] = FrameGT(
            boxes=xyxy[keep], labels=labels, track_ids=rows[keep, 0].astype(np.int64),
            ignore_boxes=xyxy[ign],
        )
    return gt


def _video_meta(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return n, fps, w, h


def list_clips(
    root: Optional[Path] = None, scenes: Optional[List[str]] = None, with_gt: bool = True,
    require_video: bool = True,
) -> List[VideoClip]:
    """Clips that have both an object annotation file and (optionally) a video."""
    root = root or virat_root()
    ann_dir = root / "ann" / "annotations"
    videos = {p.stem: p for p in (root / "videos").rglob("*.mp4")} if (root / "videos").exists() else {}
    clips: List[VideoClip] = []
    for ann in sorted(ann_dir.glob("*.viratdata.objects.txt")):
        if not ann.is_file():   # the release-2 zip contains a few directory entries with .txt names
            continue
        clip_id = ann.name.split(".")[0]
        meta = parse_clip_id(clip_id)
        if scenes and meta["scene"] not in scenes:
            continue
        vp = videos.get(clip_id)
        if vp is None and require_video:
            continue
        if vp is not None:
            n, fps, w, h = _video_meta(vp)
        else:
            n, fps, w, h = 0, 30.0, 0, 0
        gt = objects_to_gt(load_objects(ann)) if with_gt else None
        clips.append(VideoClip(clip_id, meta["scene"], vp, None, n, fps, w, h, gt))
    return clips
