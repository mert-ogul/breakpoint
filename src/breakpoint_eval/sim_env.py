"""LeRobot gym environment construction for breakpoint run.

Module import is free of gymnasium/torch/LeRobot so callers that only need
``preprocess_observation`` (and unit tests that monkeypatch it) can load under
a ``[dev]`` install without the simulator stack.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Any

import numpy as np

from breakpoint_eval.candidate_map import CandidateSpec

LOGGER = logging.getLogger("breakpoint_eval.sim_env")

# mujoco 2.3.7 Darwin CGL uses ctypes.CDLL on OpenGL.framework; that path is a
# broken symlink when the binary lives only in the dyld shared cache (macOS 14+).
# Import mujoco with GL disabled, then point dm_control at GLFW (verified here).
_MUJOCO_DISABLE = frozenset({"disable", "disabled", "off", "false", "0"})
_MUJOCO_GLFW_ALIASES = frozenset({"", "glfw", "cgl", "enable", "enabled", "on", "true", "1"})


def prepare_mujoco_macos_gl() -> None:
    """Make MuJoCo + dm_control importable for rendering on Apple Silicon macOS.

    No-op on non-Darwin. Does not change preprocessing, action scaling, success
    semantics, or episode limits. Does not synthesize a fake environment.
    """
    if platform.system() != "Darwin":
        return
    if "mujoco" in sys.modules:
        return
    requested = os.environ.get("MUJOCO_GL", "glfw").strip().lower()
    if requested in _MUJOCO_DISABLE:
        return
    os.environ["MUJOCO_GL"] = "disable"
    import mujoco  # noqa: F401

    if requested in _MUJOCO_GLFW_ALIASES:
        if requested == "cgl":
            LOGGER.warning(
                "MUJOCO_GL=cgl is unreliable on this macOS (OpenGL.framework "
                "ctypes load); using glfw for dm_control rendering"
            )
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = requested
    LOGGER.info(
        "macOS MuJoCo prepare: imported mujoco %s with GL disabled, "
        "MUJOCO_GL=%s for dm_control",
        getattr(sys.modules["mujoco"], "__version__", "?"),
        os.environ["MUJOCO_GL"],
    )


def make_env_config(spec: CandidateSpec) -> Any:
    from lerobot.common.envs.factory import make_env_config as lerobot_make_env_config

    return lerobot_make_env_config(spec.env_type, task=spec.env_task)


def make_vector_env(spec: CandidateSpec, *, n_envs: int = 1) -> Any:
    prepare_mujoco_macos_gl()
    from lerobot.common.envs.factory import make_env

    if n_envs != 1:
        raise ValueError("breakpoint run currently supports n_envs=1 only")
    cfg = make_env_config(spec)
    return make_env(cfg, n_envs=n_envs, use_async_envs=False)


def control_period_s(spec: CandidateSpec) -> float:
    cfg = make_env_config(spec)
    return 1.0 / float(cfg.fps)


def preprocess_observation(observation: dict[str, np.ndarray]) -> dict[str, Any]:
    """Match LeRobot eval preprocessing (channel-first float32 images in [0, 1])."""
    from lerobot.common.envs.utils import preprocess_observation as lerobot_preprocess

    return lerobot_preprocess(observation)


def load_policy(spec: CandidateSpec, checkpoint_dir: str, device: str) -> Any:
    from lerobot.common.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    cfg.pretrained_path = checkpoint_dir
    env_cfg = make_env_config(spec)
    policy = make_policy(cfg, device=device, env_cfg=env_cfg)
    policy.eval()
    return policy
