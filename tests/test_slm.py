# -*- coding: utf-8 -*-
"""Tests for the SLM pixel-effect simulation layer (fill factor)."""

import numpy as np
import pytest
import torch

from differential_shaping import params
from differential_shaping.simulation import (
    quantize_slm_phase,
    slm_far_field_intensity,
    slm_far_field_intensity_many,
    slm_field,
    slm_fill_mask,
)

SLM_UP = 4  # must match simulation.slm._SLM_UP (亚像素采样倍数)


# --------------------------------------------------------------------------- #
# phase quantization
# --------------------------------------------------------------------------- #
class TestQuantize:
    def test_shape_dtype_preserved(self):
        phase = torch.randn(params.N, params.N, dtype=torch.float32)
        q = quantize_slm_phase(phase)
        assert q.shape == (params.N, params.N)
        assert q.dtype == torch.float32

    def test_values_in_wrapped_range(self):
        phase = torch.linspace(-10, 10, 1000, dtype=torch.float32)
        q = quantize_slm_phase(phase)
        assert bool((q >= 0).all())
        # 最高灰度级对应的相位可精确等于 2*pi (满驱动), 故用 <=。
        assert bool((q <= 2 * torch.pi).all())

    def test_quantized_to_levels(self):
        phase = torch.linspace(0, 2 * torch.pi, 10000, dtype=torch.float32)
        for levels in (4, 256):
            q = quantize_slm_phase(phase, levels=levels)
            n_unique = torch.unique(q).numel()
            assert n_unique <= levels


# --------------------------------------------------------------------------- #
# fill-factor microstructure masks (visualization)
# --------------------------------------------------------------------------- #
class TestFillMask:
    def test_fine_grid_shape_dtype(self):
        mask = slm_fill_mask(0.8)
        assert mask.shape == (params.N * SLM_UP,) * 2
        assert mask.dtype == torch.float32
        assert bool((mask >= 0).all()) and bool((mask <= 1).all())

    def test_distinguishes_fill_factors(self):
        m60 = slm_fill_mask(0.60)
        m80 = slm_fill_mask(0.80)
        m100 = slm_fill_mask(1.00)
        # 细网格上掩模必须能分辨 60% / 80% / 100% 的死区差异。
        assert float((m60 - m80).abs().sum()) > 0.0
        assert float((m80 - m100).abs().sum()) > 0.0
        # 填充因子越大, 有效区占比越大。
        assert float(m60.sum()) < float(m80.sum()) < float(m100.sum())

    def test_full_fill_factor_is_uniform_inside_pupil(self):
        mask100 = slm_fill_mask(1.00)
        # FF=100%: 无死区, 掩模在圆形孔径内全为 1。 圆形面积占方形网格
        # 的比例为 pi/4 ~= 0.785, 故整体均值应接近该值。
        frac = float(mask100.mean())
        assert frac == pytest.approx(np.pi / 4.0, abs=1e-2)

    def test_downsample_to_grid_N(self):
        mask = slm_fill_mask(0.8, grid_N=params.N)
        assert mask.shape == (params.N, params.N)

    def test_invalid_fill_factor_raises(self):
        with pytest.raises(ValueError):
            slm_fill_mask(0.0)
        with pytest.raises(ValueError):
            slm_fill_mask(1.5)


# --------------------------------------------------------------------------- #
# far-field intensity (physics: fill-factor pixelation)
# --------------------------------------------------------------------------- #
class TestFarField:
    def _plane(self):
        return torch.zeros(params.N, params.N, dtype=torch.float32)

    def test_shape_dtype(self):
        I = slm_far_field_intensity(self._plane(), fill_factor=0.8)
        assert I.shape == (params.N * SLM_UP,) * 2
        assert I.dtype == torch.float32

    def test_peak_monotonic_in_fill_factor(self):
        """填充因子越大, 中央峰越强 (死区越少, 能量越集中)。"""
        peaks = {
            ff: float(slm_far_field_intensity(self._plane(), ff).max())
            for ff in (0.60, 0.80, 1.00)
        }
        assert peaks[0.60] < peaks[0.80] < peaks[1.00]

    def test_total_energy_monotonic_in_fill_factor(self):
        """被死区遮挡的总体能量损失: 总强度随填充因子单调上升。"""
        totals = {
            ff: float(
                slm_far_field_intensity(self._plane(), ff).sum()
            )
            for ff in (0.60, 0.80, 1.00)
        }
        assert totals[0.60] < totals[0.80] < totals[1.00]

    def test_plane_wave_peak_on_axis(self):
        I = slm_far_field_intensity(self._plane(), fill_factor=0.8)
        flat = I.flatten()
        idx = int(flat.argmax())
        cy, cx = idx // I.shape[1], idx % I.shape[1]
        assert abs(cy - I.shape[0] // 2) <= 0 and abs(cx - I.shape[1] // 2) <= 0

    def test_many_matches_individual(self):
        ffs = [0.60, 0.80, 1.00]
        many = slm_far_field_intensity_many(self._plane(), ffs)
        assert set(many.keys()) == set(ffs)
        for ff in ffs:
            single = slm_far_field_intensity(self._plane(), ff)
            assert bool(torch.equal(many[ff], single))

    def test_invalid_fill_factor_raises(self):
        with pytest.raises(ValueError):
            slm_far_field_intensity(self._plane(), 0.0)
        with pytest.raises(ValueError):
            slm_far_field_intensity(self._plane(), 2.0)


# --------------------------------------------------------------------------- #
# slm_field (explicit pixelated pupil field)
# --------------------------------------------------------------------------- #
class TestField:
    def test_slm_field_shape_complex(self):
        phase = torch.zeros(params.N, params.N, dtype=torch.float32)
        E = slm_field(phase, fill_factor=0.8)
        assert E.shape == (params.N * SLM_UP,) * 2
        assert torch.is_complex(E)
        # 有效区照度归一: |E| 仅取 0/1 (掩模 * 相位)。
        mag = E.abs()
        assert bool((mag == 0).any())  # 死区/孔径外为 0
        assert bool((mag == 1).any())  # 有效区为 1

    def test_slm_field_invalid_ff(self):
        phase = torch.zeros(params.N, params.N, dtype=torch.float32)
        with pytest.raises(ValueError):
            slm_field(phase, fill_factor=0.0)