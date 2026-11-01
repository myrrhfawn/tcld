"""Scenes100 (CVPR 2023) training frames as unlabeled per-camera clips.

The official servers are down; the frames used here are the last 9000 training frames
of every video (every 6th frame of the 30 fps stream, i.e. 30 minutes at 5 fps),
re-shared on Google Drive by a Chalmers thesis (see FINDINGS 2026-10-08). Layout:

    <TCLD_DATA>/scenes100/frames/<video id>/<8-digit original frame index>.jpg

Manual evaluation annotations (`manual_valid.zip`) are not available, so these clips
carry no GT: they serve adaptation and pair statistics only. `masks.json` /
`videos.json` come from the submodule `external/scenes100/scenes100/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from tcld.data.common import VideoClip
from tcld.paths import REPO_ROOT, data_dir


def frames_root() -> Path:
    return data_dir() / "scenes100" / "frames"


def video_meta(root: Optional[Path] = None) -> dict:
    meta_path = (root or REPO_ROOT) / "external" / "scenes100" / "scenes100" / "videos.json"
    if not meta_path.exists():
        return {}
    return {v["id"]: v for v in json.loads(meta_path.read_text())}


def list_cameras(ids: Optional[List[str]] = None, root: Optional[Path] = None) -> List[VideoClip]:
    """One clip per video id directory that holds frames (sorted by original frame index)."""
    root = root or frames_root()
    meta = video_meta()
    clips: List[VideoClip] = []
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    for d in dirs:
        vid = d.name
        if ids is not None and vid not in ids:
            continue
        files = sorted(d.glob("*.jpg"))
        if len(files) < 2:
            continue
        idx = [int(f.stem) for f in files[:2]]
        step = max(1, idx[1] - idx[0])
        src_fps = float(meta.get(vid, {}).get("video", {}).get("fps", 30.0))
        w = h = 0
        if vid in meta:
            w, h = int(meta[vid]["video"]["W"]), int(meta[vid]["video"]["H"])
        else:
            from PIL import Image

            with Image.open(files[0]) as im:
                w, h = im.size
        clips.append(VideoClip(clip_id=vid, camera=vid, video_path=None, frames_dir=d, frame_files=files,
                               num_frames=len(files), fps=src_fps / step, width=w, height=h, gt=None))
    return clips
