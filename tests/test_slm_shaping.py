# -*- coding: utf-8 -*-
"""Tests for SLM-pixelated beam shaping (fill factor x algorithm)."""

import numpy as np
import pytest
import torch

from differential_shaping import params
from differential_shaping.optimization import (
    make_square_target,
    slm_forward_cropped,
    slm_shaping_forward,
    spgd_shaping_optimization,
    target_shaping_optimization,
)
from differential_shaping.optimization import shaping as shaping_mod
from differential_shaping.simulation import (
    generate_influence_functions,
    generate_turbulence_phase,
)


# --------------------------------------------------------------------------- #
# SLM 前向裁剪 (与理想前向同视场的 (N, N) 接口)
# --------------------------------------------------------------------------- #
class TestSlmForwardCropped:
    def test_crops_to_n_by_n(self):
        forward = slm_forward_cropped(0.8, quantize=False)
        phase = torch.zeros(params.N, params.N, dtype=torch.float32)
        I = forward(phase)
        assert I.shape == (params.N, params.N)
        assert I.dtype == torch.float32
        assert bool(torch.isfinite(I).all())

    def test_autograd_flows_through_crop(self):
        # 核心: Torch-GD 需要梯度穿过细网格 SLM 前向 + 中央裁剪。
        forward = slm_forward_cropped(0.6, quantize=False)
        phase = torch.zeros(params.N, params.N, dtype=torch.float32, requires_grad=True)
        target = make_square_target(half_width=6).to(dtype=torch.float32)
        I = forward(phase)
        I_n = I / (I.sum() + 1e-12)
        loss = torch.mean((I_n - target / (target.sum() + 1e-12)) ** 2)
        loss.backward()
        assert phase.grad is not None
        assert bool(torch.isfinite(phase.grad).all())
        assert float(phase.grad.abs().sum()) > 0.0

    def test_zero_phase_total_intensity_scales_with_fill_factor(self):
        # 物理校验: (裁剪窗内) 总强度随填充因子上升而上升 —— 死区遮挡效应。
        phase = torch.zeros(params.N, params.N, dtype=torch.float32)
        sums = [
            float(slm_forward_cropped(ff, quantize=False)(phase).sum())
            for ff in (0.60, 0.80, 1.00)
        ]
        assert sums[0] < sums[1] < sums[2]


# --------------------------------------------------------------------------- #
# 上下文管理器: 替换并恢复 shaping 模块级前向
# --------------------------------------------------------------------------- #
class TestSlmShapingForward:
    def test_patch_used_and_restored(self):
        original = shaping_mod.far_field_intensity_metric
        with slm_shaping_forward(0.8, quantize=False):
            assert shaping_mod.far_field_intensity_metric is not original
        assert shaping_mod.far_field_intensity_metric is original

    def test_spgd_runs_under_slm_forward(self):
        # 数值梯度方法不受可微性约束, 应能直接从 SLM 前向运行。
        turb_phase = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms,
            params.seed_phase,
        )
        inf_flat = generate_influence_functions().reshape(params.n_act, -1).numpy()
        target_np = make_square_target(half_width=6).numpy()
        with slm_shaping_forward(0.8, quantize=False):
            u, dm_u, loss_hist, eng_hist, I_np = spgd_shaping_optimization(
                turb_phase, inf_flat, target_np, max_iter=5, label="square"
            )
        assert np.isfinite(loss_hist).all()
        assert np.isfinite(eng_hist).all()
        assert I_np.shape == (params.N, params.N)
        # 只跑了 5 步, 能量不应有 NaN/超界。
        assert eng_hist[-1] >= 0.0 and eng_hist[-1] <= 1.0 + 1e-4


# --------------------------------------------------------------------------- #
# Torch-GD 在 SLM 前向下端到端可训练 (短迭代回归)
# --------------------------------------------------------------------------- #
class TestTorchGdUnderSlm:
    def test_short_run_loss_decreases(self):
        turb_phase = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms,
            params.seed_phase,
        )
        inf_flat = generate_influence_functions().reshape(params.n_act, -1).numpy()
        target_np = make_square_target(half_width=6).numpy()
        with slm_shaping_forward(0.8, quantize=False):
            u, dm_u, loss_hist, eng_hist, I_np = target_shaping_optimization(
                turb_phase, inf_flat, target_np, max_iter=10, lr=0.02,
                seed=params.seed_spgd, label="square",
            )
        assert np.isfinite(loss_hist).all()
        assert loss_hist[-1] < loss_hist[0]
        assert I_np.shape == (params.N, params.N)