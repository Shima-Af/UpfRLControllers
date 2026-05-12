"""Copy non-DVC digital-twin artifacts into data/external/.

A handful of files that the env and the digital twin need are NOT tracked
by either upstream pipeline's DVC remote, so `dvc pull` cannot fetch them:

  * traffic_forecaster/predictions_test.npy  (pipeline writes it but
    UpfTrafficForecaster's `dvc.yaml` doesn't declare it as an output)
  * traffic_forecaster/targets_test.npy      (same)
  * traffic_forecaster/forecast_eval_summary.json
    (metric file in upstream git with `cache: false`)
  * profiling_twin/switching_costs.yaml      (hand-authored)
  * profiling_twin/params.yaml               (hand-authored config snapshot)

This script copies them from a peer directory laid out the same way as
`UpfRLControllers/data/external/` (the default is /home/ubuntu/UPF_NDT/data/external).

Existing files are left alone unless --force is passed. The script is
idempotent and never deletes anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Files that must be bootstrapped because they are not in any DVC remote.
# Each entry is (relative path under data/external/, ).
BOOTSTRAP_FILES = [
    "traffic_forecaster/predictions_test.npy",
    "traffic_forecaster/targets_test.npy",
    "traffic_forecaster/forecast_eval_summary.json",
    "profiling_twin/switching_costs.yaml",
    "profiling_twin/params.yaml",
]

DEFAULT_SOURCE = Path("/home/ubuntu/UPF_NDT/data/external")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Directory laid out like data/external/ "
            f"(default: {DEFAULT_SOURCE})"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist in data/external/.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dest_root = repo_root / "data" / "external"
    src_root: Path = args.source.resolve()

    if not src_root.is_dir():
        print(f"[FAIL] source directory does not exist: {src_root}")
        print(
            "       Point --source at a sibling repo's data/external/ tree, "
            "or populate that path first."
        )
        return 1

    print(f"Source:      {src_root}")
    print(f"Destination: {dest_root}")
    print("-" * 60)

    copied = 0
    skipped = 0
    missing = 0
    for rel in BOOTSTRAP_FILES:
        src = src_root / rel
        dst = dest_root / rel
        if not src.is_file():
            print(f"[MISS] {rel}: not found at {src}")
            missing += 1
            continue
        if dst.is_file() and not args.force:
            print(f"[SKIP] {rel}: already present (use --force to overwrite)")
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[ OK ] {rel}: copied ({src.stat().st_size} bytes)")
        copied += 1

    print("-" * 60)
    print(f"Done. copied={copied} skipped={skipped} missing={missing}")
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
