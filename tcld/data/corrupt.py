"""Synthetic shooting-condition shifts applied to every decoded frame (adaptation, frozen model, flow and evaluation alike).

Selected by the TCLD_CORRUPT environment variable (set by run_adapt from the experiment's `corruption` key):
  night  - low light: gamma 2.2 darkening to ~30 % brightness, sensor noise, JPEG q=30
  fog    - homogeneous haze (transmission 0.45 towards light grey) + mild blur
  jpeg   - heavy compression at reduced resolution (1/3 scale, JPEG q=15, upscaled back)
  nightjpeg - night followed by jpeg (stacked shift: low-light, compressed CCTV)
Noise is seeded by (clip, frame), so a frame looks the same every time it is read.
"""

from __future__ import annotations

import io
import os
import zlib

import numpy as np

KINDS = ("night", "fog", "jpeg", "nightjpeg")


def active() -> str:
    k = os.environ.get("TCLD_CORRUPT", "").strip()
    if k and k not in KINDS:
        raise ValueError(f"unknown TCLD_CORRUPT={k!r}, expected one of {KINDS}")
    return k


def _jpeg(rgb: np.ndarray, q: int) -> np.ndarray:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def apply(rgb: np.ndarray, key: str, frame: int, kind: str) -> np.ndarray:
    if not kind:
        return rgb
    rng = np.random.default_rng((zlib.crc32(key.encode()) * 1_000_003 + frame) % (2**32))
    x = rgb.astype(np.float32) / 255.0
    if kind == "nightjpeg":
        return apply(apply(rgb, key, frame, "night"), key, frame, "jpeg")
    if kind == "night":
        x = 0.3 * x ** 2.2
        x = x + rng.normal(0.0, 0.03, x.shape).astype(np.float32) + rng.normal(0.0, 0.02, x.shape).astype(np.float32) * np.sqrt(x)
        return _jpeg((np.clip(x, 0, 1) * 255).astype(np.uint8), 30)
    if kind == "fog":
        import cv2

        x = cv2.GaussianBlur(x, (0, 0), 1.5)
        x = 0.45 * x + 0.55 * 0.8
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if kind == "jpeg":
        import cv2

        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        return cv2.resize(_jpeg(small, 15), (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(kind)
