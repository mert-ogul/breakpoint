# Reproduce the ACT export validation

Pinned stack: LeRobot `3c0a209…`, ACT transfer Hub `9e935457…`,
`requirements-simulation.txt` + `requirements-onnx.txt`. Do not invent pins.

Seed pools (disjoint):

| Pool | Seeds |
| --- | --- |
| Calibration | `9000–9007` |
| Held-out export evaluation | `2000–2007` |
| Closed-loop diagnosis | `2100,2101` |

Corpus capture uses planned control steps `0,49,99,149,199,249,299,349` **where
reached** (no padding of early terminations).

INT4 and FP16 are **off** unless `--int4` / `--fp16` are passed. Unsupported INT8
is recorded as `status=errored|unsupported` with its own sidecar; no silent
recipe fallback.

## 1. Export FP32 + INT8 (no INT4 / FP16)

```bash
source .venv/bin/activate
export MUJOCO_GL=glfw   # prepare_mujoco_macos_gl also handles Darwin CGL break

breakpoint export \
  --candidate act_aloha_transfer \
  --device cpu \
  --calibration-seeds 9000,9001,9002,9003,9004,9005,9006,9007 \
  --eval-seeds 2000,2001,2002,2003,2004,2005,2006,2007 \
  --closed-loop-seeds 2100,2101 \
  --no-closed-loop \
  --output-dir outputs/v1/export-act-transfer
```

(`--no-closed-loop` keeps export focused on corpus + open-loop parity + INT8;
task/smoke outcomes come from diagnose below. INT4/FP16 stay off by default.)

Artifacts:

- `act_chunk_fp32.onnx`, optional `act_chunk_int8_qdq.onnx`
- `corpus_eval.npz` / `corpus_calibration.npz` (+ JSON sidecars)
- `export_manifest.json`
- `export_validation_int8.json` (INT8 identity/finite/shape/coverage; not FP32 hash)

## 2. Closed-loop diagnosis (binds to on-disk FP32 hash)

```bash
breakpoint diagnose-closed-loop \
  --candidate act_aloha_transfer \
  --export-dir outputs/v1/export-act-transfer \
  --output-dir outputs/v1/export-act-transfer-diagnosis \
  --seeds 2100,2101 \
  --onnx-name act_chunk_fp32.onnx \
  --device cpu
```

## 3. Write the FP32 export validation

```bash
breakpoint write-export-validation \
  --export-dir outputs/v1/export-act-transfer \
  --diagnosis outputs/v1/export-act-transfer-diagnosis/closed_loop_diagnosis.json \
  --output outputs/v1/export-act-transfer/export_validation.json
```

Refuses stale diagnosis (missing/`model_sha256` mismatch), missing
candidate/revision/recipe/corpus fields, non-finite/missing shape parity, and
incomplete task episodes. Independent trajectory equality is diagnostic only.

## 4. Two-model verification (FP32 then INT8)

```bash
# FP32 — requires export_validation.json bound to this graph hash
python - <<'PY'
from pathlib import Path
from breakpoint_eval.export_runtime import resolve_onnx_runtime
root = Path("outputs/v1/export-act-transfer")
fp32 = resolve_onnx_runtime(root / "act_chunk_fp32.onnx", expected_candidate="act_aloha_transfer")
print("fp32", fp32.variant_label, fp32.recipe_id, fp32.model_sha256[:12], fp32.export_validation.get("passed"))
PY

# INT8 — requires export_validation_int8.json (FP32 sidecar must not authorize INT8)
python - <<'PY'
from pathlib import Path
from breakpoint_eval.export_runtime import resolve_onnx_runtime
root = Path("outputs/v1/export-act-transfer")
int8_path = root / "act_chunk_int8_qdq.onnx"
if not int8_path.is_file():
    raise SystemExit("INT8 graph missing; see export_manifest variants status/detail")
rt = resolve_onnx_runtime(int8_path, expected_candidate="act_aloha_transfer")
print("int8", rt.variant_label, rt.recipe_id, rt.model_sha256[:12], rt.export_validation.get("passed"))
print("parent", (rt.export_validation.get("binding") or {}).get("parent_fp32_sha256", "")[:12])
PY
```

## 5. Run the deployment stress protocol

See [benchmark_protocol.md](benchmark_protocol.md) for `scripts/run_benchmark.py`, `scripts/profile_device.py`,
and `configs/act-transfer.json`. Prefer `--dry-run` before any long eval.

## 6. Package, report, and media for the completed full run

See [`benchmarks/act-transfer/deployment-stress/REPRODUCE.md`](../benchmarks/act-transfer/deployment-stress/REPRODUCE.md)
for `scripts/package_benchmark.py`, `scripts/render_benchmark.py`, and `scripts/render_benchmark_media.py`.

## Notes

- Do not start long stress sweeps from this document.
- Future stress seeds must stay outside the pools above.
- Reduced INT8 task performance is expected measurement, not an export numerical gate.
