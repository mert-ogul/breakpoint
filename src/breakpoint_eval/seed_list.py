"""Seed list parsing without simulator / torch imports."""

from __future__ import annotations


def parse_seed_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("seed list is empty")
    return [int(p) for p in parts]


def build_seeds(*, n_episodes: int | None, seeds: list[int] | None, start_seed: int) -> list[int]:
    if seeds is not None and n_episodes is not None and len(seeds) != n_episodes:
        raise ValueError(f"--seeds length ({len(seeds)}) must match --n-episodes ({n_episodes})")
    if seeds is not None:
        return seeds
    if n_episodes is None:
        raise ValueError("Provide --n-episodes and/or --seeds")
    if n_episodes < 1:
        raise ValueError("--n-episodes must be >= 1")
    return [start_seed + i for i in range(n_episodes)]
