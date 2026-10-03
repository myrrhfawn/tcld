"""Repository and volume paths.

Inside Docker: TCLD_DATA=/data, TCLD_RUNS=/runs, TCLD_CKPT=/ckpt.
On the host without .env they default to ./data, ./runs, ./checkpoints (gitignored).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DFINE_ROOT = REPO_ROOT / "external" / "D-FINE"
CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    p = Path(raw).expanduser() if raw else default
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def data_dir() -> Path:
    return _env_path("TCLD_DATA", REPO_ROOT / "data")


def runs_dir() -> Path:
    return _env_path("TCLD_RUNS", REPO_ROOT / "runs")


def ckpt_dir() -> Path:
    return _env_path("TCLD_CKPT", REPO_ROOT / "checkpoints")


def model_config(name: str) -> Path:
    """`dfine_s` -> configs/model/dfine_s.yml (or a path passed through)."""
    p = Path(name)
    if p.suffix == ".yml" and p.exists():
        return p
    return CONFIGS / "model" / f"{name}.yml"


def profile_config(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".yml" and p.exists():
        return p
    return CONFIGS / "profiles" / f"{name}.yml"
