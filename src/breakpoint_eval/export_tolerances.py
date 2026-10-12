"""Declared ONNX FP32 parity tolerances (set before measuring results).

These bounds are engineering acceptance criteria for export-induced numeric
drift between PyTorch FP32 and ONNX Runtime FP32 on ACT action chunks in
environment units. They are not tuned after inspecting a particular run.

Quantized variants are compared for reporting only; they are not required to
meet these FP32 parity bounds. Success-rate degradation under quantization is
a measured result, not a gate on valid computation.
"""

from __future__ import annotations

# Per-element absolute / relative tolerances for allclose-style checks.
ONNX_FP32_PARITY_ATOL = 1e-4
ONNX_FP32_PARITY_RTOL = 1e-4

# Aggregate error caps on the held-out corpus (env action units).
ONNX_FP32_PARITY_MAX_ABS = 1e-4
ONNX_FP32_PARITY_MEAN_ABS = 1e-5
# Mean absolute error averaged over action dimensions (then over corpus).
ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS = 1e-4

# Closed-loop paired action agreement (same seeds, same control steps).
ONNX_FP32_CLOSED_LOOP_ATOL = 1e-4
ONNX_FP32_CLOSED_LOOP_RTOL = 1e-4
