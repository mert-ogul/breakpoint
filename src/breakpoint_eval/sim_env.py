"""LeRobot gym environment construction for breakpoint run.

Module import is free of gymnasium/torch/LeRobot so callers that only need
``preprocess_observation`` (and unit tests that monkeypatch it) can load under
a ``[dev]`` install without the simulator stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from breakpoint_eval.candidate_map import CandidateSpec


def make_env_config(spec: CandidateSpec) -> Any:
    from lerobot.common.envs.factory import make_env_config as lerobot_make_env_config

    return lerobot_make_env_config(spec.env_type, task=spec.env_task)


def make_vector_env(spec: CandidateSpec, *, n_envs: int = 1) -> Any:
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
