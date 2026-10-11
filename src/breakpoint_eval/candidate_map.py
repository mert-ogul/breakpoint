"""Direct model-to-environment mapping for feasibility candidates.

Not a discovery framework: each entry is an explicit, reviewed pairing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSpec:
    hub_repo: str
    # Revision chosen for LeRobot@3c0a209 config schema (no policy.device/use_amp).
    hub_revision: str
    # Current Hub main tip observed during the probe (may require newer LeRobot).
    hub_main_revision: str
    policy_type: str
    env_type: str
    env_task: str
    sim_package: str
    sim_package_pin: str
    card_success_rate_pct: float | None
    card_n_episodes: int
    notes: str


# Card-trained LeRobot commit cited by all three model cards.
LEROBOT_CARD_COMMIT = "3c0a209f9fac4d2a57617e686a7f2a2309144ba2"

CANDIDATES: dict[str, CandidateSpec] = {
    "act_aloha_transfer": CandidateSpec(
        hub_repo="lerobot/act_aloha_sim_transfer_cube_human",
        hub_revision="9e93545744656e57d20f8b2d631d5d72544002ce",
        hub_main_revision="ba73b2766f1371cdc133ca4efb97eb090d744625",
        policy_type="act",
        env_type="aloha",
        env_task="AlohaTransferCube-v0",
        sim_package="gym-aloha",
        sim_package_pin="0.1.1",
        card_success_rate_pct=83.0,
        card_n_episodes=500,
        notes=(
            "Success when reward==4 (left gripper holds cube off table). "
            "Obs: observation.images.top (3,480,640), observation.state (14). "
            "Action chunk 100; n_obs_steps 1; MEAN_STD norms."
        ),
    ),
    "act_aloha_insertion": CandidateSpec(
        hub_repo="lerobot/act_aloha_sim_insertion_human",
        hub_revision="2bd896d066e7a2c65c4c945d065e7f0bb1e389ca",
        hub_main_revision="33259aa86eb45fdf85350280044a33d9d50e40c3",
        policy_type="act",
        env_type="aloha",
        env_task="AlohaInsertion-v0",
        sim_package="gym-aloha",
        sim_package_pin="0.1.1",
        card_success_rate_pct=20.6,
        card_n_episodes=500,
        notes=(
            "Success when reward==4 (peg contacts pin). "
            "Same ACT obs/action layout as transfer. Eval not run in this probe."
        ),
    ),
    "diffusion_pusht": CandidateSpec(
        hub_repo="lerobot/diffusion_pusht",
        hub_revision="202c8695947a458d8ce08ce1c0219f2588d60b33",
        hub_main_revision="84a7c23178445c6bbf7e1a884ff497017910f653",
        policy_type="diffusion",
        env_type="pusht",
        env_task="PushT-v0",
        sim_package="gym-pusht",
        sim_package_pin="0.1.5",
        card_success_rate_pct=65.4,
        card_n_episodes=500,
        notes=(
            "Success when max goal coverage >= 0.95. "
            "Obs: observation.image (3,96,96), observation.state (2); n_obs_steps 2; "
            "horizon 16; n_action_steps 8; action in [0,512]^2. Eval not run."
        ),
    ),
}
