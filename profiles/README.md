# Device latency profiles

Measured ONNX Runtime inference traces for Breakpoint latency replay.

- Layout: `profiles/<device_label>/…json`
- Create on the target device with `breakpoint profile` (no simulator install).
- `breakpoint run --device rpi5` resolves an unambiguous file under `profiles/rpi5/`.
- Prefer `--latency-profile PATH` when multiple variants exist.
- Do not invent timings for a missing profile.
