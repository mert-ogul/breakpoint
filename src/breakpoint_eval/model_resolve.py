"""Resolve CLI model identifiers to pinned candidate keys."""

from __future__ import annotations

from breakpoint_eval.candidate_map import CANDIDATES, CandidateSpec


def resolve_model_to_candidate(model: str) -> tuple[str, CandidateSpec]:
    """Accept candidate key or Hub repo id (e.g. lerobot/act_aloha_sim_transfer_cube_human)."""
    text = model.strip()
    if not text:
        raise ValueError("model identifier is empty")
    if text in CANDIDATES:
        return text, CANDIDATES[text]
    hub_matches = [name for name, spec in CANDIDATES.items() if spec.hub_repo == text]
    if len(hub_matches) == 1:
        name = hub_matches[0]
        return name, CANDIDATES[name]
    if len(hub_matches) > 1:
        raise ValueError(
            f"Hub repo {text!r} matches multiple candidates {hub_matches}; "
            "pass an explicit candidate key"
        )
    known = ", ".join(sorted(CANDIDATES))
    repos = ", ".join(sorted({s.hub_repo for s in CANDIDATES.values()}))
    raise ValueError(
        f"Unknown model {text!r}. Known candidate keys: {known}. Known Hub repos: {repos}."
    )
