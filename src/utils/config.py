"""Lightweight helpers for locating the repo root and loading YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root directory (the parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Relative paths are resolved against the project root so callers can pass
    paths like ``"configs/scenario_rl.yaml"`` without worrying about cwd.
    """
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with p.open("r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_digital_twin_paths() -> dict[str, Any]:
    """Load ``configs/digital_twin_paths.yaml``."""
    return load_yaml("configs/digital_twin_paths.yaml")


def load_rl_scenario() -> dict[str, Any]:
    """Load ``configs/scenario_rl.yaml``."""
    return load_yaml("configs/scenario_rl.yaml")
