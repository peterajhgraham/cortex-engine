"""Post-training INT8 quantization with calibration.

Implementation notes for Claude Code (Phase 2):
    - Per-channel weight quantization for linear layers (symmetric, signed INT8).
    - Per-tensor activation quantization with statistics collected from a
      calibration set of held-out trials.
    - SmoothQuant-style activation smoothing factor to reduce outliers.
    - Replace nn.Linear with QuantizedLinear in cortex/models/cortex.py via a
      conversion utility.
    - Custom dequantize-on-load if storing INT8 on disk.

Reference:
    - Xiao et al. 2023, SmoothQuant
    - GPTQ for the per-row weight quantization pattern
"""
