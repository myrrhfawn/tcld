"""Frame-pair sampling from unlabeled video (P1.5).

A `PairSampler` enumerates (clip, t, t + delta) triples deterministically from a
seed; `PairDataset` turns them into model-ready tensors. Augmentation is
photometric only and applied identically to both frames, so a pair never differs
by geometry we did not account for in motion compensation.

Frames are decoded on the fly (cv2 for mp4, PIL for jpg directories) — each
DataLoader worker keeps its own FrameReader per clip. Reading t and t+delta back
to back is cheap after the first seek.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tcld.data.common import FrameReader, VideoClip


@dataclass(frozen=True)
class PairSpec:
    clip_idx: int
    t: int
    delta: int


class PairSampler:
    """Deterministic list of pairs.

    base_stride: distance (frames) between consecutive anchor frames t within a clip;
    deltas:      candidate deltas, one drawn per anchor (uniform) with the sampler seed;
    max_pairs_per_clip: optional cap (random subset, seeded).
    """

    def __init__(
        self,
        clips: Sequence[VideoClip],
        deltas: Sequence[int] = (1, 2, 5),
        base_stride: int = 30,
        max_pairs_per_clip: Optional[int] = None,
        seed: int = 0,
    ):
        self.clips = list(clips)
        self.deltas = list(deltas)
        self.base_stride = base_stride
        self.seed = seed
        rng = random.Random(seed)
        self.pairs: List[PairSpec] = []
        for ci, clip in enumerate(self.clips):
            max_d = max(self.deltas)
            anchors = list(range(0, max(clip.num_frames - max_d, 1), base_stride))
            specs = [PairSpec(ci, t, rng.choice(self.deltas)) for t in anchors]
            if max_pairs_per_clip is not None and len(specs) > max_pairs_per_clip:
                specs = rng.sample(specs, max_pairs_per_clip)
                specs.sort(key=lambda s: s.t)
            self.pairs.extend(specs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> PairSpec:
        return self.pairs[i]


def _resize_rgb(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize HxWx3 uint8 to (h, w) with bilinear interpolation (matches upstream
    val transform: plain Resize to a square, no letterbox)."""
    import cv2

    h, w = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def photometric_params(rng: random.Random, strength: float) -> Dict[str, float]:
    """Brightness/contrast/saturation factors shared by both frames of a pair."""
    s = strength
    return {
        "brightness": rng.uniform(1 - s, 1 + s),
        "contrast": rng.uniform(1 - s, 1 + s),
        "saturation": rng.uniform(1 - s, 1 + s),
    }


def apply_photometric(x: torch.Tensor, p: Dict[str, float]) -> torch.Tensor:
    """x: [3, H, W] float in [0, 1]."""
    import torchvision.transforms.functional as TF

    x = TF.adjust_brightness(x, p["brightness"])
    x = TF.adjust_contrast(x, p["contrast"])
    x = TF.adjust_saturation(x, p["saturation"])
    return x.clamp(0, 1)


class PairDataset(Dataset):
    """Yields dict(img_a, img_b [3,h,w] float, meta) for each PairSpec.
    `meta` carries clip_id, camera, t, delta, orig_size (w, h) so losses can map
    network coordinates back to pixels."""

    def __init__(
        self,
        sampler: PairSampler,
        input_size: Tuple[int, int] = (640, 640),
        photometric: float = 0.0,
        seed: int = 0,
        flow: bool = False,
        flow_downscale: int = 2,
        independent_views: bool = False,
        n_context: int = 1,
        context_stride: int = 2,
    ):
        """flow=True adds dense Farneback flow a->b computed on the RESIZED frames
        (so it lives in network-input pixel space, like bin positions computed with
        image_size=input_size); stored as [h, w, 2] float16."""
        self.sampler = sampler
        self.input_size = input_size
        self.photometric = photometric
        self.seed = seed
        self.flow = flow
        self.flow_downscale = flow_downscale
        self.independent_views = independent_views
        self.n_context = n_context          # frames t-(k-1)s ... t used as teacher context (temporal ensembling)
        self.context_stride = context_stride
        self._readers: Dict[int, FrameReader] = {}

    def __len__(self) -> int:
        return len(self.sampler)

    def __getstate__(self):
        # open cv2.VideoCapture handles are per-process; never pickle them (spawn/forkserver)
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers = {}

    def _reader(self, clip_idx: int) -> FrameReader:
        r = self._readers.get(clip_idx)
        if r is None:
            r = FrameReader(self.sampler.clips[clip_idx])
            self._readers[clip_idx] = r
        return r

    def load_pair(self, spec: PairSpec) -> Tuple[np.ndarray, np.ndarray]:
        reader = self._reader(spec.clip_idx)
        return reader.read(spec.t), reader.read(spec.t + spec.delta)

    def __getitem__(self, i: int) -> dict:
        spec = self.sampler[i]
        clip = self.sampler.clips[spec.clip_idx]
        a, b = self.load_pair(spec)
        h0, w0 = a.shape[:2]
        a_r, b_r = _resize_rgb(a, self.input_size), _resize_rgb(b, self.input_size)
        item = {}
        if self.flow:
            from tcld.tcl.motion import dense_flow

            item["flow"] = torch.from_numpy(dense_flow(a_r, b_r, downscale=self.flow_downscale)).half()
        a = torch.from_numpy(a_r).permute(2, 0, 1).float() / 255
        b = torch.from_numpy(b_r).permute(2, 0, 1).float() / 255
        if self.photometric > 0:
            rng = random.Random(self.seed * 1_000_003 + i)
            p = photometric_params(rng, self.photometric)
            if self.independent_views:   # same-frame two-view baseline: different photometric per view
                p2 = photometric_params(rng, self.photometric)
                a, b = apply_photometric(a, p), apply_photometric(b, p2)
            else:
                a, b = apply_photometric(a, p), apply_photometric(b, p)
        if self.n_context > 1:
            reader = self._reader(spec.clip_idx)
            ctx = []
            for j in range(self.n_context - 1, 0, -1):
                f = max(spec.t - j * self.context_stride, 0)
                c = torch.from_numpy(_resize_rgb(reader.read(f), self.input_size)).permute(2, 0, 1).float() / 255
                ctx.append(c)
            ctx.append(a)
            item["img_ctx"] = torch.stack(ctx)          # [k, 3, h, w], last = frame t
        item.update({
            "img_a": a, "img_b": b,
            "meta": {"clip_id": clip.clip_id, "camera": clip.camera, "t": spec.t,
                     "delta": spec.delta, "orig_size": (w0, h0), "pair_index": i},
        })
        return item


def worker_init(worker_id: int) -> None:
    """Use as DataLoader(worker_init_fn=worker_init): OpenCV's internal thread pool and
    torch intra-op threads deadlock after fork; keep workers single-threaded."""
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def collate_pairs(batch: List[dict]) -> dict:
    out = {
        "img_a": torch.stack([b["img_a"] for b in batch]),
        "img_b": torch.stack([b["img_b"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
    if "flow" in batch[0]:
        out["flow"] = torch.stack([b["flow"] for b in batch])
    if "img_ctx" in batch[0]:
        out["img_ctx"] = torch.stack([b["img_ctx"] for b in batch])   # [B, k, 3, h, w]
    return out
