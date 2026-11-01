"""MOTChallenge sequences (MOT17Det / MOT20) as VideoClips with dense person GT.

Layout under $TCLD_DATA/mot:
  MOT17Det/train/MOT17-02/{img1/000001.jpg, gt/gt.txt, seqinfo.ini}
  MOT20/train/MOT20-01/...

gt.txt columns: frame(1-based), id, x, y, w, h, conf(1 = evaluated, 0 = ignore),
class, visibility. Classes: 1 pedestrian, 2 person on vehicle, 7 static person
count as person; 8 distractor and 12 reflection become ignore regions; the rest
(vehicles, occluders) are dropped. Only entries with conf == 1 are positives.
Static-camera sequences: MOT17-02, -04, -09 (train) and all MOT20 train sequences.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tcld.data.common import FrameGT, VideoClip
from tcld.paths import data_dir

MOT_PERSON_CLASSES = {1, 2, 7}
MOT_IGNORE_CLASSES = {8, 12}
STATIC_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def mot_root() -> Path:
    return data_dir() / "mot"


def read_seqinfo(seq_dir: Path) -> Dict[str, float]:
    cp = configparser.ConfigParser()
    cp.read(seq_dir / "seqinfo.ini")
    s = cp["Sequence"]
    return {
        "fps": float(s.get("frameRate", 30)),
        "num_frames": int(s["seqLength"]),
        "width": int(s["imWidth"]),
        "height": int(s["imHeight"]),
        "img_dir": s.get("imDir", "img1"),
    }


def load_gt(gt_path: Path, min_visibility: float = 0.0) -> Dict[int, FrameGT]:
    """gt.txt -> {frame0: FrameGT}. Frames are converted to 0-based."""
    arr = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    if arr.size == 0:
        return {}
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    frames, starts = np.unique(arr[:, 0], return_index=True)
    ends = list(starts[1:]) + [arr.shape[0]]
    gt: Dict[int, FrameGT] = {}
    for f, s, e in zip(frames.astype(int).tolist(), starts.tolist(), ends):
        rows = arr[s:e]
        cls = rows[:, 7].astype(int) if rows.shape[1] > 7 else np.ones(len(rows), int)
        conf = rows[:, 6] if rows.shape[1] > 6 else np.ones(len(rows))
        vis = rows[:, 8] if rows.shape[1] > 8 else np.ones(len(rows))
        xyxy = np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1).astype(np.float32)
        pos = np.isin(cls, list(MOT_PERSON_CLASSES)) & (conf > 0) & (vis >= min_visibility)
        ign = (np.isin(cls, list(MOT_IGNORE_CLASSES)) | (np.isin(cls, list(MOT_PERSON_CLASSES)) & ~pos))
        gt[f - 1] = FrameGT(
            boxes=xyxy[pos], labels=np.zeros(int(pos.sum()), np.int64),
            track_ids=rows[pos, 1].astype(np.int64), ignore_boxes=xyxy[ign],
        )
    return gt


# Unlabeled MOTChallenge *test* sequences recorded by the same cameras as labeled train
# sequences (checked visually on the first frames). Used for the cross-recording protocol:
# adapt on the test recording, evaluate on the whole labeled train recording.
SAME_CAMERA_TEST = {"MOT17-02": "MOT17-01", "MOT17-04": "MOT17-03", "MOT20-01": "MOT20-07", "MOT20-03": "MOT20-04", "MOT20-05": "MOT20-04"}


def list_sequences(
    root: Optional[Path] = None, names: Optional[List[str]] = None, with_gt: bool = True, split: str = "auto"
) -> List[VideoClip]:
    """Each MOT sequence is one VideoClip whose camera == sequence name. `split` is
    'train', 'test' or 'auto' (train if present, else test); test sequences have no gt."""
    root = root or mot_root()
    names = names or STATIC_SEQUENCES
    clips: List[VideoClip] = []
    for name in names:
        ds = "MOT17Det" if name.startswith("MOT17") else "MOT20"
        candidates = [split] if split != "auto" else ["train", "test"]
        seq_dir = next((root / ds / sp / name for sp in candidates if (root / ds / sp / name).exists()), None)
        if seq_dir is None:
            continue
        info = read_seqinfo(seq_dir)
        gt_path = seq_dir / "gt" / "gt.txt"
        gt = load_gt(gt_path) if (with_gt and gt_path.exists()) else None
        clips.append(VideoClip(
            clip_id=name, camera=name, video_path=None, frames_dir=seq_dir / info["img_dir"],
            num_frames=info["num_frames"], fps=info["fps"], width=info["width"], height=info["height"], gt=gt,
        ))
    return clips
