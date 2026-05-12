"""Verify the local setup before training RL controllers.

Run this after cloning the repo and pulling artifacts from S3 to confirm:
  * configs load cleanly,
  * expected digital-twin artifact directories exist under ``data/external/``,
  * the ``UpfDigitalTwin`` dependency is importable.

Missing artifacts produce helpful warnings, not hard failures, so this script
can be run on a fresh checkout before any S3 sync has happened. Only invalid
config files cause a non-zero exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python scripts/check_setup.py`` from anywhere by putting the repo
# root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import (  # noqa: E402
    load_digital_twin_paths,
    load_rl_scenario,
    project_root,
)


def _ok(msg: str) -> None:
    print(f"[ OK ]  {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN]  {msg}")


def _info(msg: str) -> None:
    print(f"[INFO]  {msg}")


def check_configs() -> tuple[dict, dict]:
    """Load both YAML configs. Hard-fail if either cannot be parsed."""
    try:
        dt_paths = load_digital_twin_paths()
        _ok("Loaded configs/digital_twin_paths.yaml")
    except Exception as e:
        print(f"[FAIL]  Could not load configs/digital_twin_paths.yaml: {e}")
        raise

    try:
        rl_cfg = load_rl_scenario()
        _ok("Loaded configs/scenario_rl.yaml")
    except Exception as e:
        print(f"[FAIL]  Could not load configs/scenario_rl.yaml: {e}")
        raise

    return dt_paths, rl_cfg


def check_artifact_dirs(dt_paths: dict) -> None:
    """Warn (don't fail) if expected artifact directories are missing."""
    root = project_root()
    expected = [
        dt_paths.get("traffic_forecaster", {}).get(
            "dir", "data/external/traffic_forecaster"
        ),
        dt_paths.get("profiling_twin", {}).get(
            "dir", "data/external/profiling_twin"
        ),
    ]
    for rel in expected:
        path = root / rel
        if path.is_dir():
            _ok(f"Artifact directory present: {rel}")
        else:
            _warn(
                f"Artifact directory missing: {rel} "
                f"(pull from S3 into data/external/ before training)"
            )


def check_digital_twin_import() -> None:
    """Best-effort import of the digital-twin dependency."""
    try:
        from upf_digital_twin import DigitalTwin  # type: ignore  # noqa: F401

        _ok("Imported DigitalTwin from upf_digital_twin")
        return
    except Exception as e:
        _warn(f"Could not import upf_digital_twin.DigitalTwin: {e}")

    _info(
        "Install the digital twin into the active environment:\n"
        "          pip install -r requirements.txt\n"
        "        (this picks up git+https://github.com/Shima-Af/UpfDigitalTwin.git)\n"
        "        or, for local development:\n"
        "          pip install -e /path/to/UpfDigitalTwin"
    )


def main() -> int:
    print(f"Repository root: {project_root()}")
    print("-" * 60)

    dt_paths, _rl_cfg = check_configs()
    print("-" * 60)

    check_artifact_dirs(dt_paths)
    print("-" * 60)

    check_digital_twin_import()
    print("-" * 60)

    _info("Setup check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
