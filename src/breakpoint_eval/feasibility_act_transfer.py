"""Run a short unmodified upstream FP32 eval for ACT ALOHA transfer.

This is a feasibility probe, not the Breakpoint CLI. It downloads one pinned
checkpoint revision and invokes LeRobot's eval.py from the card-pinned commit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from breakpoint_eval.candidate_map import LEROBOT_CARD_COMMIT
from breakpoint_eval.stack import ensure_lerobot_checkout, repo_root_from_package

LOGGER = logging.getLogger("breakpoint_eval.feasibility")

# Card-trained stack: all three candidate cards cite this LeRobot commit.
LEROBOT_COMMIT = LEROBOT_CARD_COMMIT

# Immutable Hub revision compatible with LeRobot@3c0a209 config schema.
# Current Hub main (ba73b276...) moved device/use_amp into policy config for
# LeRobot PR #812 and fails to parse on the card-pinned commit.
ACT_TRANSFER_REPO = "lerobot/act_aloha_sim_transfer_cube_human"
ACT_TRANSFER_REVISION = "9e93545744656e57d20f8b2d631d5d72544002ce"

ENV_TYPE = "aloha"
ENV_TASK = "AlohaTransferCube-v0"


@dataclass(frozen=True)
class FeasibilityRequest:
    repo_root: Path
    n_episodes: int
    batch_size: int
    device: str
    seed: int
    use_amp: bool


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def download_checkpoint(cache_dir: Path) -> Path:
    """Download the ACT transfer checkpoint at a pinned revision."""
    path = snapshot_download(
        repo_id=ACT_TRANSFER_REPO,
        revision=ACT_TRANSFER_REVISION,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def write_versions(
    path: Path, lerobot_root: Path, checkpoint_dir: Path, req: FeasibilityRequest
) -> None:
    import gym_aloha
    import gymnasium
    import torch

    request = asdict(req)
    request["repo_root"] = str(req.repo_root)
    payload = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gymnasium": gymnasium.__version__,
        "gym_aloha": getattr(gym_aloha, "__version__", "unknown"),
        "lerobot_commit": LEROBOT_COMMIT,
        "lerobot_root": str(lerobot_root),
        "checkpoint_repo": ACT_TRANSFER_REPO,
        "checkpoint_revision": ACT_TRANSFER_REVISION,
        "checkpoint_dir": str(checkpoint_dir),
        "request": request,
        "requested_device": req.device,
        "use_amp": req.use_amp,
        "env_type": ENV_TYPE,
        "env_task": ENV_TASK,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_upstream_eval(
    *,
    lerobot_root: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    req: FeasibilityRequest,
) -> Path:
    """Invoke unmodified LeRobot eval.py (FP32 when use_amp=false)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_py = lerobot_root / "lerobot" / "scripts" / "eval.py"
    if not eval_py.is_file():
        raise FileNotFoundError(f"Missing upstream eval script: {eval_py}")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(lerobot_root) if not existing else f"{lerobot_root}{os.pathsep}{existing}"
    )
    # Prefer GLFW on macOS. Direct MUJOCO_GL=glfw still breaks mujoco 2.3.7 CGL
    # import on dyld-cache OpenGL hosts; upstream eval.py does not run
    # prepare_mujoco_macos_gl (breakpoint sim_env does).
    env.setdefault("MUJOCO_GL", "glfw")

    cmd = [
        sys.executable,
        str(eval_py),
        f"--policy.path={checkpoint_dir}",
        f"--output_dir={output_dir}",
        f"--env.type={ENV_TYPE}",
        f"--env.task={ENV_TASK}",
        f"--eval.n_episodes={req.n_episodes}",
        f"--eval.batch_size={req.batch_size}",
        f"--device={req.device}",
        f"--use_amp={str(req.use_amp).lower()}",
        f"--seed={req.seed}",
    ]
    _run(cmd, cwd=req.repo_root, env=env)
    return output_dir / "eval_info.json"


def parse_args(argv: list[str] | None = None) -> FeasibilityRequest:
    repo_root = repo_root_from_package()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cuda", "cpu", "mps"],
        help="Card used cuda; this host defaults to mps when available.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Card eval used --use_amp=false (FP32).",
    )
    args = parser.parse_args(argv)
    if args.n_episodes < 1:
        raise SystemExit("--n-episodes must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    return FeasibilityRequest(
        repo_root=args.repo_root.resolve(),
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        use_amp=args.use_amp,
    )


def write_gate_status(path: Path, **fields: object) -> None:
    path.write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    req = parse_args(argv)
    artifacts = req.repo_root / "artifacts" / "feasibility_act_transfer"
    artifacts.mkdir(parents=True, exist_ok=True)
    gate_path = artifacts / "gate_status.json"

    lerobot_root = ensure_lerobot_checkout(req.repo_root)
    checkpoint_dir = download_checkpoint(req.repo_root / ".cache" / "huggingface")
    write_versions(artifacts / "versions.json", lerobot_root, checkpoint_dir, req)

    output_dir = artifacts / "upstream_eval"
    try:
        eval_info_path = run_upstream_eval(
            lerobot_root=lerobot_root,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            req=req,
        )
    except subprocess.CalledProcessError as exc:
        write_gate_status(
            gate_path,
            loaded="unknown_in_this_process",
            ran=False,
            reproduced_baseline=False,
            checkpoint_repo=ACT_TRANSFER_REPO,
            checkpoint_revision=ACT_TRANSFER_REVISION,
            lerobot_commit=LEROBOT_COMMIT,
            returncode=exc.returncode,
            command=exc.cmd,
            note=(
                "Upstream eval.py failed. Checkpoint download succeeded. "
                "See terminal traceback; common block on macOS is MuJoCo OpenGL/CGL."
            ),
        )
        LOGGER.exception("Upstream eval failed; gate left open (ran=false)")
        return exc.returncode or 1

    LOGGER.info("Wrote versions to %s", artifacts / "versions.json")
    LOGGER.info("Upstream eval_info: %s", eval_info_path)
    aggregated: object | None = None
    if eval_info_path.is_file():
        info = json.loads(eval_info_path.read_text(encoding="utf-8"))
        aggregated = info.get("aggregated", info)
        LOGGER.info("Aggregated: %s", json.dumps(aggregated, indent=2))
    write_gate_status(
        gate_path,
        loaded=True,
        ran=True,
        reproduced_baseline=False,
        checkpoint_repo=ACT_TRANSFER_REPO,
        checkpoint_revision=ACT_TRANSFER_REVISION,
        lerobot_commit=LEROBOT_COMMIT,
        eval_info=str(eval_info_path),
        aggregated=aggregated,
        note="Short upstream eval finished; card-scale reproduction not claimed.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
