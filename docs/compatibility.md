# Compatibility baseline

Status date: 2026-10-12 (local probe on Apple M3, macOS 14.6).

Breakpoint evaluates pretrained policies in simulation. This document records one
explicit Python / LeRobot / simulator stack and what was actually verified for
three Hub candidates. Gates are distinguished as:

| Gate | Meaning |
| --- | --- |
| **loaded** | Checkpoint config + weights load through LeRobot's policy factory; a forward/`select_action` call succeeds |
| **ran** | Unmodified upstream `lerobot/scripts/eval.py` completes at least one full episode |
| **reproduced baseline** | Upstream FP32 eval at card scale (or a pre-registered short protocol) matches the card metric within documented tolerance |

Nothing below invents success rates for candidates that were not evaluated.

## Tested stack (single pin set)

| Component | Pin | Source |
| --- | --- | --- |
| Python | 3.11.14 | `requires-python = ">=3.11,<3.13"` (LeRobot card pin needs `<3.13`) |
| LeRobot | `3c0a209f9fac4d2a57617e686a7f2a2309144ba2` | Cited on all three model cards; vendor checkout via feasibility script |
| torch / torchvision | 2.6.0 / 0.21.0 | `poetry.lock` at that LeRobot commit |
| gymnasium | 0.29.1 | Exact pin in LeRobot `pyproject.toml` |
| gym-aloha | 0.1.1 | `poetry.lock` at card commit (`mujoco>=2.3.7,<3`) |
| mujoco | 2.3.7 | Resolved with gym-aloha 0.1.1 |
| dm-control | 1.0.14 | Resolved with gym-aloha 0.1.1 |
| gym-pusht | 0.1.5 | `poetry.lock` at card commit; required for Diffusion PushT |
| draccus | git `9b690730ca108930519f48cc5dead72a72fd27cb` | LeRobot `poetry.lock` resolved_reference |

Lockfile: [`requirements-feasibility.txt`](../requirements-feasibility.txt).

Do **not** assume `pip install lerobot` (latest) reproduces these cards. Hub issues
[#2605](https://github.com/huggingface/lerobot/issues/2605) and
[#4168](https://github.com/huggingface/lerobot/issues/4168) report much lower
ALOHA ACT success on current LeRobot than the cards (≈50–66% vs 83% transfer).

Do **not** use a generic Transformers Hub loader. Load through LeRobot
`PreTrainedConfig.from_pretrained` + `make_policy` / `lerobot/scripts/eval.py`.

### Hub main vs card-compatible revisions

On 2025-03-06 the three Hub repos gained `device` / `use_amp` inside policy
`config.json` (LeRobot PR [#812](https://github.com/huggingface/lerobot/pull/812)).
That schema **fails** on LeRobot@`3c0a209` with:

`DecodingError: The fields device, use_amp are not valid for ACTConfig`

(also tracked upstream as [#850](https://github.com/huggingface/lerobot/issues/850)).

This baseline therefore pins **pre-refactor Hub revisions** that still share
weights with current main for the ACT transfer repo (config-only commit on main).

## Candidate support matrix

Direct mapping lives in `src/breakpoint_eval/candidate_map.py`.

| Candidate | Hub revision (pinned) | Env | Policy | Card metric | loaded | ran | reproduced | camera / L=0 | ONNX FP32 | device profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT ALOHA transfer | `9e93545744656e57d20f8b2d631d5d72544002ce` | `aloha` / `AlohaTransferCube-v0` | ACT | 83.0% / 500 ep | **yes** | **no** (GL block) | **no** | **yes** | **yes** | **yes** (ORT) |
| ACT ALOHA insertion | `2bd896d066e7a2c65c4c945d065e7f0bb1e389ca` | `aloha` / `AlohaInsertion-v0` | ACT | 20.6% / 500 ep | **yes** | **no** (GL block) | **no** | **yes** | **yes** (shared ACT export) | not attempted |
| Diffusion PushT | `202c8695947a458d8ce08ce1c0219f2588d60b33` | `pusht` / `PushT-v0` | diffusion | 65.4% / 500 ep | **yes** | not attempted | **no** | **yes** | **unsupported** | **unsupported** (ORT); PyTorch full-denoise path available |

Machine-readable rows: `src/breakpoint_eval/support_status.py`. Do **not** pool
success rates across different tasks into one policy ranking. Do **not** copy
ACT latency profiles onto Diffusion Policy or another checkpoint.

Pre-stress gates for every supported policy: native baseline load, documented
environment success semantics, state reset, camera degradation at raw pixels,
and zero-latency equivalence. Stress sweeps require those gates.

Current Hub `main` tips observed during the probe (require newer LeRobot for config parse):

- `lerobot/act_aloha_sim_transfer_cube_human` → `ba73b2766f1371cdc133ca4efb97eb090d744625`
- `lerobot/act_aloha_sim_insertion_human` → `33259aa86eb45fdf85350280044a33d9d50e40c3`
- `lerobot/diffusion_pusht` → `84a7c23178445c6bbf7e1a884ff497017910f653`

## ACT transfer: verified interfaces

From Hub `config.json` / `train_config.json` at the pinned revision and LeRobot
env code at `3c0a209`:

| Item | Value |
| --- | --- |
| Policy type | `act` |
| Observation keys | `observation.images.top` `(3, 480, 640)` float32 in `[0, 1]` after `preprocess_observation`; `observation.state` `(14,)` |
| Env raw pixels | `pixels/top` uint8 HWC `(480, 640, 3)`; `agent_pos` `(14,)` |
| Action | `(14,)` joint targets; MEAN_STD normalized in policy |
| Chunk / history | `chunk_size=100`, `n_action_steps=100`, `n_obs_steps=1`, `temporal_ensemble_coeff=null` |
| Vision | ResNet18 ImageNet weights, VAE on |
| Success predicate | `gym_aloha` reward == 4: left gripper contacts cube and cube not contacting table |
| Episode length | 400 steps @ 50 fps |
| Card eval command shape | `python lerobot/scripts/eval.py --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=500 --eval.batch_size=50 --device=cuda --use_amp=false` |

Insertion uses the same ACT layout; success is reward == 4 when peg contacts pin.
PushT (card / `gym-pusht`): image `(3,96,96)`, state `(2,)`, `n_obs_steps=2`,
horizon 16 / `n_action_steps=8`, success if coverage ≥ 0.95; actions in `[0, 512]^2`.

## Evidence (ACT transfer)

### loaded — yes

Command (policy factory, no env):

```bash
# after feasibility env install; PYTHONPATH includes vendor/lerobot@3c0a209
python -c "..."  # see artifacts/feasibility_act_transfer/load_evidence.json
```

Result recorded in `artifacts/feasibility_act_transfer/load_evidence.json`:

- `ACTPolicy`, device `mps`, 51 613 672 parameters
- `select_action` → shape `(1, 14)`, dtype float32
- Checkpoint revision `9e93545744656e57d20f8b2d631d5d72544002ce`

Upstream eval **parsed** the same checkpoint and printed the full
`EvalPipelineConfig` (policy features, env task, `use_amp: False`) before env
construction.

### ran — no (blocked)

Upstream command invoked by the feasibility script:

```bash
cd /path/to/breakpoint
source .venv/bin/activate
export MUJOCO_GL=glfw   # or omit; mujoco 2.3.7 on macOS uses CGL
python -m breakpoint_eval.feasibility_act_transfer \
  --n-episodes 2 --batch-size 1 --device mps --no-use-amp
```

Equivalent direct upstream call:

```bash
export PYTHONPATH="$PWD/vendor/lerobot${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=glfw
python vendor/lerobot/lerobot/scripts/eval.py \
  --policy.path=$PWD/.cache/huggingface/models--lerobot--act_aloha_sim_transfer_cube_human/snapshots/9e93545744656e57d20f8b2d631d5d72544002ce \
  --output_dir=$PWD/artifacts/feasibility_act_transfer/upstream_eval \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=2 \
  --eval.batch_size=1 \
  --device=mps \
  --use_amp=false \
  --seed=1000
```

Exact block:

```text
OSError: dlopen(/System/Library/OpenGL.framework/OpenGL, 0x0006): tried:
  '/System/Library/OpenGL.framework/OpenGL' (no such file),
  '/System/Volumes/Preboot/Cryptexes/OS/System/Library/OpenGL.framework/OpenGL' (no such file),
  '/System/Library/OpenGL.framework/OpenGL' (no such file, not in dyld cache)
```

`MUJOCO_GL=osmesa` / `egl` are invalid on mujoco 2.3.7 macOS wheels. Env creation
never completed, so there is **no** episode outcome and **no** rollout video from
this host.

### reproduced baseline — no

Card claims 83% over 500 CUDA FP32 episodes. This probe did not finish any
episode. Gate remains open.

## Unresolved problems

1. **OpenGL / MuJoCo on this agent host** — mujoco 2.3.7 cannot create a GL
   context here; blocks `ran` and `reproduced` for ALOHA.
2. **Hub main vs card LeRobot** — current Hub configs need LeRobot ≥ PR #812;
   cards still advertise training at `3c0a209`. Baseline uses older Hub
   revisions for schema match.
3. **Modern LeRobot success regression** — open issues report card metrics not
   reproduced on latest LeRobot even when training/eval commands look the same.
4. **Insertion / PushT closed-loop** — both checkpoints **load**; MuJoCo GL still
   blocks ALOHA `ran` on some hosts. PushT needs `gym-pusht==0.1.5`; episode
   `ran` / card reproduction not claimed here.
5. **Diffusion ONNX** — full action generation is a multi-step denoising loop;
   single-shot ORT action-chunk export is **unsupported**. Profile full PyTorch
   `generate_actions` (or a labeled mixed UNet+scheduler runtime), never one
   UNet pass and never ACT timings.
6. **Device gap** — card used `cuda`; this host used `mps`/`cpu` for load probes.
   Reproduction should prefer CUDA to match the card, or document the effective
   device separately.

## Smallest next step (gate open)

On a machine where MuJoCo can open a GL context (typical interactive macOS
session or Linux with EGL/OSMesa):

1. Create the venv and install the freeze:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-feasibility.txt
uv pip install -e .
```

2. Run the short upstream FP32 eval (script clones LeRobot@`3c0a209` as needed):

```bash
python -m breakpoint_eval.feasibility_act_transfer \
  --n-episodes 2 --batch-size 1 --device cuda --no-use-amp
# on Apple GPU without CUDA:
#   --device mps
```

3. Confirm `artifacts/feasibility_act_transfer/upstream_eval/eval_info.json`
   exists, note per-episode `success`, and keep one video under `.../videos/`.
   Only then promote the ACT transfer row from **loaded** to **ran**.
