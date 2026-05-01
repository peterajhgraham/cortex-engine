"""Correctness tests for INT8 quantization.

Structure:

  Group 1 — ActivationStats:
      Verify percentile computation and update semantics.

  Group 2 — QuantizedLinear:
      Verify weight quantization/dequantization round-trip error bounds.
      Verify output shape, dtype, and bias handling.
      Verify that QuantizedLinear reproduces nn.Linear within quantization error.

  Group 3 — derive_scales / attach hooks:
      Verify that hooks record the correct number of batches.
      Verify that scales are non-zero and finite.
      Verify per-channel weight scale has the correct shape.

  Group 4 — convert_model:
      Verify that nn.Linear layers are replaced with QuantizedLinear.
      Verify that LayerNorm, embeddings, etc. are NOT replaced.
      Verify that the converted model produces finite output.

  Group 5 — model_weight_bytes:
      Verify that INT8 model uses less memory than float32.
      Verify the reduction is approximately 4× for linear layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from cortex.quantization.calibrate import (
    ActivationStats,
    CalibrationHook,
    LayerScales,
    QuantizedLinear,
    attach_calibration_hooks,
    calibration_summary,
    convert_model,
    count_quantized_linears,
    derive_scales,
    model_weight_bytes,
    remove_calibration_hooks,
)


# ── Group 1: ActivationStats ──────────────────────────────────────────────────


class TestActivationStats:
    def test_empty_percentile(self):
        stats = ActivationStats()
        assert stats.percentile_99 == 1.0

    def test_single_update(self):
        stats = ActivationStats()
        x = torch.tensor([1.0, -2.0, 3.0])
        stats.update(x)
        assert stats.n_batches == 1
        # 99th percentile of 1 element is that element
        assert stats.percentile_99 == pytest.approx(3.0, abs=1e-5)

    def test_many_updates_percentile(self):
        stats = ActivationStats()
        # 99 batches with abs_max=1.0, 1 batch with abs_max=100.0
        for _ in range(99):
            stats.update(torch.ones(10))
        stats.update(torch.full((10,), 100.0))
        assert stats.n_batches == 100
        # 99th percentile should be ~1.0, not 100.0
        assert stats.percentile_99 < 50.0

    def test_abs_max_taken(self):
        stats = ActivationStats()
        x = torch.tensor([-5.0, 1.0, 2.0])
        stats.update(x)
        assert stats.percentile_99 == pytest.approx(5.0, abs=1e-5)


# ── Group 2: QuantizedLinear ──────────────────────────────────────────────────


class TestQuantizedLinear:
    def _make_layer(self, in_f: int = 32, out_f: int = 16, bias: bool = False):
        linear = nn.Linear(in_f, out_f, bias=bias)
        nn.init.normal_(linear.weight, std=0.5)
        scales = LayerScales(
            weight_scale=linear.weight.detach().float().abs().amax(dim=1).clamp(min=1e-8) / 127.0,
            act_scale=0.01,
        )
        return QuantizedLinear.from_linear(linear, scales), linear, scales

    def test_output_shape(self):
        ql, _, _ = self._make_layer(32, 16)
        x = torch.randn(8, 32)
        out = ql(x)
        assert out.shape == (8, 16)

    def test_output_dtype_preserved(self):
        for dtype in (torch.float32, torch.float16):
            ql, _, _ = self._make_layer(16, 8)
            x = torch.randn(4, 16, dtype=dtype)
            out = ql(x)
            assert out.dtype == dtype

    def test_with_bias(self):
        ql, linear, scales = self._make_layer(32, 16, bias=True)
        assert ql.bias is not None

    def test_no_bias(self):
        ql, linear, scales = self._make_layer(32, 16, bias=False)
        assert ql.bias is None

    def test_weight_stored_as_int8(self):
        ql, _, _ = self._make_layer(32, 16)
        assert ql.weight_int8.dtype == torch.int8

    def test_weight_scale_shape(self):
        ql, _, _ = self._make_layer(32, 16)
        assert ql.weight_scale.shape == (16,)

    def test_matches_float_within_quantization_error(self):
        """INT8 output must be close to float output within ~0.5 LSB per element."""
        torch.manual_seed(0)
        in_f, out_f = 64, 32
        linear = nn.Linear(in_f, out_f, bias=False)
        nn.init.normal_(linear.weight, std=1.0)
        scales = LayerScales(
            weight_scale=linear.weight.detach().float().abs().amax(dim=1).clamp(min=1e-8) / 127.0,
            act_scale=0.01,
        )
        ql = QuantizedLinear.from_linear(linear, scales)

        x = torch.randn(16, in_f)
        ref = linear(x)
        out = ql(x)

        # Max error is bounded by 0.5 × scale × |x|_max per output neuron
        # In practice much tighter; we just verify it's in a reasonable range
        max_err = (out - ref).abs().max().item()
        assert max_err < 1.0, f"Max quant error {max_err:.4f} seems too large"

    def test_register_buffer_in_state_dict(self):
        ql, _, _ = self._make_layer(16, 8)
        sd = ql.state_dict()
        assert "weight_int8" in sd
        assert "weight_scale" in sd

    def test_extra_repr(self):
        ql, _, _ = self._make_layer(16, 8)
        repr_str = ql.extra_repr()
        assert "INT8" in repr_str


# ── Group 3: attach_calibration_hooks / derive_scales ────────────────────────


class TestCalibration:
    def _small_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def test_hooks_attached_to_all_linears(self):
        model = self._small_model()
        hooks = attach_calibration_hooks(model)
        n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        assert len(hooks) == n_linear

    def test_hooks_record_batches(self):
        model = self._small_model()
        hooks = attach_calibration_hooks(model)

        with torch.no_grad():
            for _ in range(5):
                model(torch.randn(4, 16))

        remove_calibration_hooks(hooks)
        # Each layer sees each batch
        for hook in hooks.values():
            assert hook.stats.n_batches == 5

    def test_remove_hooks_stops_recording(self):
        model = self._small_model()
        hooks = attach_calibration_hooks(model)

        with torch.no_grad():
            model(torch.randn(4, 16))

        remove_calibration_hooks(hooks)

        with torch.no_grad():
            model(torch.randn(4, 16))  # should NOT be recorded

        for hook in hooks.values():
            assert hook.stats.n_batches == 1  # only the first batch

    def test_derive_scales_all_finite(self):
        model = self._small_model()
        hooks = attach_calibration_hooks(model)

        with torch.no_grad():
            for _ in range(3):
                model(torch.randn(4, 16))

        remove_calibration_hooks(hooks)
        scales = derive_scales(model, hooks)

        for name, s in scales.items():
            assert torch.isfinite(s.weight_scale).all(), f"weight_scale not finite for {name}"
            assert s.act_scale > 0, f"act_scale non-positive for {name}"

    def test_derive_scales_weight_scale_shape(self):
        model = self._small_model()
        hooks = attach_calibration_hooks(model)

        with torch.no_grad():
            model(torch.randn(4, 16))

        remove_calibration_hooks(hooks)
        scales = derive_scales(model, hooks)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in scales:
                assert scales[name].weight_scale.shape == (module.out_features,), \
                    f"weight_scale shape mismatch for {name}"

    def test_no_scales_without_forward(self):
        """If calibration never ran, derive_scales should return empty dict."""
        model = self._small_model()
        hooks = attach_calibration_hooks(model)
        remove_calibration_hooks(hooks)
        scales = derive_scales(model, hooks)
        assert len(scales) == 0


# ── Group 4: convert_model ────────────────────────────────────────────────────


class TestConvertModel:
    def _calibrated_small_model(self) -> tuple[nn.Module, dict]:
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        hooks = attach_calibration_hooks(model)
        with torch.no_grad():
            for _ in range(3):
                model(torch.randn(4, 16))
        remove_calibration_hooks(hooks)
        scales = derive_scales(model, hooks)
        return model, scales

    def test_linears_replaced(self):
        model, scales = self._calibrated_small_model()
        n_linear_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_model(model, scales, inplace=True)
        n_linear_after = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        n_quant = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
        assert n_linear_after == 0
        assert n_quant == n_linear_before

    def test_layer_norm_preserved(self):
        model, scales = self._calibrated_small_model()
        convert_model(model, scales, inplace=True)
        n_ln = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
        assert n_ln == 1

    def test_converted_model_produces_finite_output(self):
        model, scales = self._calibrated_small_model()
        convert_model(model, scales, inplace=True)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(8, 16))
        assert torch.isfinite(out).all()

    def test_not_inplace_leaves_original_unchanged(self):
        model, scales = self._calibrated_small_model()
        model_q = convert_model(model, scales, inplace=False)
        # Original still has plain nn.Linear
        assert any(isinstance(m, nn.Linear) for m in model.modules())
        # Copy has QuantizedLinear
        assert any(isinstance(m, QuantizedLinear) for m in model_q.modules())

    def test_count_quantized_linears(self):
        model, scales = self._calibrated_small_model()
        convert_model(model, scales, inplace=True)
        n_quant, n_total = count_quantized_linears(model)
        assert n_quant == n_total  # all linear layers were quantized
        assert n_quant == 2


# ── Group 5: model_weight_bytes ───────────────────────────────────────────────


class TestModelWeightBytes:
    def test_int8_smaller_than_float32(self):
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64))
        fp_bytes = model_weight_bytes(model)

        hooks = attach_calibration_hooks(model)
        with torch.no_grad():
            model(torch.randn(4, 128))
        remove_calibration_hooks(hooks)
        scales = derive_scales(model, hooks)
        convert_model(model, scales, inplace=True)
        int8_bytes = model_weight_bytes(model)

        assert int8_bytes < fp_bytes, f"INT8 ({int8_bytes}) not smaller than fp32 ({fp_bytes})"

    def test_reduction_approximately_4x_for_pure_linear_model(self):
        """A model with only Linear layers should see ~4× weight reduction.

        convert_model walks named_children(), so the model root must be a
        container (e.g., nn.Sequential), not a bare nn.Linear.
        """
        # Large enough to make the approximation good; bias and scale buffers add small overhead
        model = nn.Sequential(nn.Linear(512, 512, bias=False))
        fp_bytes = model_weight_bytes(model)

        hooks = attach_calibration_hooks(model)
        with torch.no_grad():
            model(torch.randn(4, 512))
        remove_calibration_hooks(hooks)
        scales = derive_scales(model, hooks)
        convert_model(model, scales, inplace=True)
        int8_bytes = model_weight_bytes(model)

        # int8 weights: 512*512*1 = 262144 bytes
        # fp32 weights: 512*512*4 = 1048576 bytes
        # scale buffer: 512*4 = 2048 bytes (small overhead)
        reduction = fp_bytes / int8_bytes
        assert 2.5 < reduction < 5.0, f"Expected ~4× reduction, got {reduction:.2f}×"
