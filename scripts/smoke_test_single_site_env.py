"""Smoke test for SingleSiteUPFEnv: reset + 10 random actions, prints summary.

Designed to be safe on a fresh checkout: if traffic / profiling artifacts are
not yet present under ``data/external/``, the script prints a helpful warning
and exits 0 rather than crashing.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"[ OK ]  {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN]  {msg}")


def _info(msg: str) -> None:
    print(f"[INFO]  {msg}")


def main() -> int:
    try:
        import numpy as np
    except Exception as e:  # numpy is a hard dep; surface clearly if it isn't there.
        print(f"[FAIL]  numpy import failed: {e}")
        return 1

    try:
        from src.envs.single_site_upf_env import SingleSiteUPFEnv
    except ModuleNotFoundError as e:
        if "upf_digital_twin" in str(e):
            _warn(f"UpfDigitalTwin not importable: {e}")
            _info(
                "Install it via `pip install -r requirements.txt` "
                "or `pip install -e /path/to/UpfDigitalTwin`."
            )
            return 0
        print(f"[FAIL]  Could not import SingleSiteUPFEnv: {e}")
        return 1
    except Exception as e:
        print(f"[FAIL]  Could not import SingleSiteUPFEnv: {e}")
        traceback.print_exc()
        return 1

    try:
        env = SingleSiteUPFEnv(cluster_idx=0, horizon_idx=0)
    except FileNotFoundError as e:
        _warn(f"Missing data artifact: {e}")
        _info(
            "Pull traffic_forecaster and profiling_twin artifacts from S3 "
            "into data/external/ before running the smoke test."
        )
        return 0
    except Exception as e:
        print(f"[FAIL]  Could not construct SingleSiteUPFEnv: {e}")
        traceback.print_exc()
        return 1

    _ok(f"Constructed SingleSiteUPFEnv for cluster_idx={env.cluster_idx}")
    print(f"        observation_space: {env.observation_space}")
    print(f"        action_space:      {env.action_space}")
    print(f"        episode length N:  {env._N}")
    print(f"        alpha (load scale):{env._alpha}")
    print("-" * 60)

    obs, info = env.reset(seed=42)
    _ok(f"reset() -> obs shape={obs.shape}, dtype={obs.dtype}")
    print(f"        reset info keys: {sorted(info.keys())}")
    print("-" * 60)

    rng = np.random.default_rng(42)
    last_info: dict = {}
    rewards: list[float] = []
    for step in range(10):
        action = int(rng.integers(0, 2))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        last_info = info
        print(
            f"  step={step:2d}  a={action} ({info['requested_upf']:>4s}) "
            f"-> sel={info['selected_upf']:>4s}  "
            f"r={reward:+.5f}  P={info['power_watts']:6.2f}W  "
            f"sw={info['switching_energy_wh']:.4f}Wh  "
            f"load={info['actual_load_gbps']:.4f}Gbps  "
            f"safe={info['is_safe']}"
        )
        if terminated or truncated:
            print(f"  episode ended at step {step} (terminated={terminated}, truncated={truncated})")
            break

    print("-" * 60)
    _ok(f"10-step roll yielded mean reward {sum(rewards)/len(rewards):+.5f}")
    print(f"        info keys: {sorted(last_info.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
