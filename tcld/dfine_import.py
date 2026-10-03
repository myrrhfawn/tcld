"""Make the pinned D-FINE submodule importable as the top-level package `src`.

D-FINE is not pip-installable; its code expects `import src...` with the repo
root on sys.path. Call `ensure_dfine_importable()` before importing from `src`.
"""

from __future__ import annotations

import sys

from tcld.paths import DFINE_ROOT


def ensure_dfine_importable() -> None:
    root = str(DFINE_ROOT)
    if not (DFINE_ROOT / "src").is_dir():
        raise RuntimeError(
            f"D-FINE submodule not found at {DFINE_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
