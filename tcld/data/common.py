"""Shared dataset primitives: the 2-class target label space, frame records and
video access.

Target classes follow Scenes100: 0 = person, 1 = vehicle (car + bus + truck).
COCO category ids that map onto them (D-FINE outputs COCO ids via
remap_mscoco_category): person=1; vehicle=3 (car), 6 (bus), 8 (truck).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

TARGET_CLASSES: List[str] = ["person", "vehicle"]
COCO_TO_TARGET: Dict[int, int] = {1: 0, 3: 1, 6: 1, 8: 1}
# COCO json category ids we emit for our own GT files (pycocotools wants ids >= 1)
TARGET_CAT_IDS: Dict[int, int] = {0: 1, 1: 2}


@dataclass
class FrameGT:
    """Ground truth of one frame. boxes are xyxy pixels; labels in {0, 1};
    track_ids align with boxes; ignore_boxes (xyxy) mark regions whose
    detections must not count as false positives (e.g. bikes, 'object')."""
    boxes: np.ndarray                       # [N, 4] float32
    labels: np.ndarray                      # [N] int64
    track_ids: np.ndarray                   # [N] int64
    ignore_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))


@dataclass
class VideoClip:
    """One continuous video (a clip of a fixed camera)."""
    clip_id: str
    camera: str                 # scene / camera id used for per-camera adaptation
    video_path: Optional[Path]  # mp4 (VIRAT) or None when frames are on disk (MOT)
    frames_dir: Optional[Path]  # directory of jpgs (MOT) or None
    num_frames: int
    fps: float
    width: int
    height: int
    gt: Optional[Dict[int, FrameGT]] = None   # frame index -> GT (dense) when available
    frame_files: Optional[List[Path]] = None  # explicit frame list (Scenes100); None = MOT numbering

    def frame_path(self, frame: int) -> Optional[Path]:
        if self.frame_files is not None:
            return self.frame_files[frame]
        if self.frames_dir is None:
            return None
        return self.frames_dir / f"{frame + 1:06d}.jpg"  # MOT numbering starts at 1


class FrameReader:
    """Sequential/random access to frames of a VideoClip as RGB uint8 arrays."""

    def __init__(self, clip: VideoClip):
        self.clip = clip
        self._cap = None
        self._next = 0

    def read(self, frame: int) -> np.ndarray:
        from tcld.data import corrupt

        return corrupt.apply(self._read_raw(frame), self.clip.camera, frame, corrupt.active())

    def _read_raw(self, frame: int) -> np.ndarray:
        if self.clip.frames_dir is not None:
            from PIL import Image

            return np.asarray(Image.open(self.clip.frame_path(frame)).convert("RGB"))
        import cv2

        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.clip.video_path))
            self._next = 0
        if frame != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self._next = frame
        ok, bgr = self._cap.read()
        if not ok:
            raise IndexError(f"{self.clip.clip_id}: cannot read frame {frame}")
        self._next = frame + 1
        return bgr[..., ::-1].copy()

    def iter_frames(self, start: int = 0, stop: Optional[int] = None, step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        stop = self.clip.num_frames if stop is None else min(stop, self.clip.num_frames)
        for f in range(start, stop, step):
            yield f, self.read(f)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def split_clips_by_camera(
    clips: List[VideoClip], eval_frac: float = 0.3, seed: int = 0, min_eval: int = 1
) -> Dict[str, Tuple[List[VideoClip], List[VideoClip]]]:
    """Deterministic per-camera split into (train_clips, eval_clips). Train clips are
    used unlabeled for adaptation; eval clips provide GT. The permutation is seeded
    per camera (seed + crc32(camera)), so a camera's split is identical whether it is
    listed alone or together with other cameras."""
    import zlib

    by_cam: Dict[str, List[VideoClip]] = {}
    for c in clips:
        by_cam.setdefault(c.camera, []).append(c)
    out = {}
    for cam in sorted(by_cam):
        cs = sorted(by_cam[cam], key=lambda c: c.clip_id)
        rng = np.random.RandomState((seed * 1_000_003 + zlib.crc32(cam.encode())) % (2**32))
        idx = rng.permutation(len(cs))
        n_eval = max(min_eval, int(round(eval_frac * len(cs)))) if len(cs) > 1 else 0
        eval_ids = set(idx[:n_eval].tolist())
        train = [c for i, c in enumerate(cs) if i not in eval_ids]
        ev = [c for i, c in enumerate(cs) if i in eval_ids]
        out[cam] = (train, ev)
    return out
