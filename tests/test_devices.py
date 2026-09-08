# -*- coding: utf-8 -*-
"""Tests for the phase-device abstraction behind run.py ``--device``.

Covers ``optimization.devices`` (``device_forward`` / ``device_compute_metrics`` /
``device_scope``), the analytic ideal-device point branch, ``direct_phase``
shaping, and the run.py CLI wiring for dm / slm / ideal.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from differential_shaping import params
from differential_shaping.optimization import (
    DEVICE_CHOICES,
    device_compute_metrics,
    device_forward,
    device_scope,
    make_square_target,
    target_shaping_optimization,
    torch_gd_optimization,
)
from differential_shaping.optimization import (
    gs as gs_mod,
    hgd as hgd_mod,
    shaping as shaping_mod,
    spgd as spgd_mod,
    torch_gd as torch_gd_mod,
)
from differential_shaping.simulation import (
    generate_influence_functions,
    generate_turbulence_phase,
)
from differential_shaping.simulation.optics import (
    compute_metrics as ideal_compute_metrics,
    far_field_intensity_metric as ideal_forward,
)

ROOT = Path(__file__).resolve().parents[1]


def _turb_phase() -> torch.Tensor:
    return generate_turbulence_phase(
        params.N, params.pixel_size, params.r0, params.target_phase_rms,
        params.seed_phase,
    )


def _inf_flat() -> np.ndarray:
    return generate_influence_functions().reshape(params.n_act, -1).numpy()


# --------------------------------------------------------------------------- #
# device_forward: 选择与校验
# --------------------------------------------------------------------------- #
class TestDeviceForward:
    def test_choices(self):
        assert DEVICE_CHOICES == ("dm", "slm", "ideal")
        for device in ("dm", "ideal"):
            forward = device_forward(device)
            assert forward is ideal_forward
        forward = device_forward("slm", fill_factor=0.8)
        assert forward is not ideal_forward
        assert forward(torch.zeros(params.N, params.N)).shape == (params.N, params.N)

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError):
            device_forward("quantum-doom")

    def test_dm_metrics_match_ideal_exactly(self):
        # dm 设备 = 经典理想前向 -> 指标应与 optics.compute_metrics 完全一致。
        turb = _turb_phase()
        cm_dm = device_compute_metrics(device_forward("dm"))
        J_dm, SR_dm, I_dm = cm_dm(turb)
        J_id, SR_id, I_id = ideal_compute_metrics(turb)
        assert torch.equal(J_dm, J_id)
        assert torch.equal(SR_dm, SR_id)
        assert torch.equal(I_dm, I_id)


# --------------------------------------------------------------------------- #
# SLM 指标物理校验 (归一化约定: 与理想前向同口径)
# --------------------------------------------------------------------------- #
class TestSlmMetricsPhysics:
    def test_slm_ff1_flat_sr_is_unity(self):
        # 归一化后: FF=1.0 零相位 SLM 峰值应 (近似) 等于理想衍射极限峰值。
        turb = _turb_phase()
        cm_slm = device_compute_metrics(device_forward("slm", fill_factor=1.0))
        J, SR, I = cm_slm(torch.zeros_like(turb))
        assert 0.90 < float(SR) < 1.05
        assert float(J) < 1.0
        assert I.shape == (params.N, params.N)

    def test_slm_sr_decreases_with_fill_factor(self):
        # 死区遮挡: 填充因子越低, 零阶峰值能量越低 -> SR 单调下降。
        turb = _turb_phase()
        zero = torch.zeros_like(turb)
        srs = [
            float(device_compute_metrics(device_forward("slm", ff))(zero)[1])
            for ff in (0.60, 0.80, 1.00)
        ]
        assert srs[0] < srs[1] < srs[2]
        assert all(sr <= 1.05 for sr in srs)

    def test_slm_forward_sample_point_less_than_ff1(self):
        # 同一张像差相位: 0.8 填充因子的峰值能量低于 1.0。
        turb = _turb_phase()
        sr08 = float(device_compute_metrics(device_forward("slm", 0.8))(turb)[1])
        sr10 = float(device_compute_metrics(device_forward("slm", 1.0))(turb)[1])
        assert sr08 < sr10


# --------------------------------------------------------------------------- #
# device_scope: 替换并恢复各优化器模块级指标
# --------------------------------------------------------------------------- #
class TestDeviceScope:
    def test_slm_swaps_and_restores(self):
        saved = [
            (spgd_mod, "compute_metrics"),
            (hgd_mod, "compute_metrics"),
            (torch_gd_mod, "compute_metrics"),
            (torch_gd_mod, "far_field_intensity_metric"),
            (gs_mod, "compute_metrics"),
            (gs_mod, "far_field_intensity_metric"),
            (shaping_mod, "far_field_intensity_metric"),
        ]
        originals = [getattr(mod, name) for mod, name in saved]
        with device_scope("slm", fill_factor=0.8):
            assert all(
                getattr(mod, name) is not orig
                for (mod, name), orig in zip(saved, originals)
            )
            assert torch_gd_mod.far_field_intensity_metric is not ideal_forward
        for (mod, name), orig in zip(saved, originals):
            assert getattr(mod, name) is orig

    def test_dm_and_ideal_scope_are_noop(self):
        for device in ("dm", "ideal"):
            orig = shaping_mod.far_field_intensity_metric
            with device_scope(device):
                assert shaping_mod.far_field_intensity_metric is orig


# --------------------------------------------------------------------------- #
# Torch-GD 在 SLM 设备下的点校正 (短跑回归)
# --------------------------------------------------------------------------- #
class TestTorchGdUnderSlm:
    def test_short_point_run_improves_sr(self):
        turb = _turb_phase()
        inf_flat = _inf_flat()
        with device_scope("slm", fill_factor=0.8):
            u, dm_u, J_hist, SR_hist = torch_gd_optimization(
                turb, inf_flat, max_iter=8, lr=0.05, seed=params.seed_spgd,
            )
        assert np.isfinite(SR_hist).all()
        assert SR_hist[-1] > SR_hist[0]
        assert float(SR_hist[0]) < 0.95  # 带像差时起始 Strehl 必然较低

    def test_quantize_zeroes_gradient(self):
        # 量化 (round) 不可微 -> 梯度为零 -> 命令保持 0, SR 纹丝不动。
        turb = _turb_phase()
        inf_flat = _inf_flat()
        with device_scope("slm", fill_factor=0.8, quantize=True):
            u, dm_u, J_hist, SR_hist = torch_gd_optimization(
                turb, inf_flat, max_iter=3, lr=0.1, seed=params.seed_spgd,
            )
        assert float(np.abs(dm_u).max()) == 0.0
        assert float(SR_hist[0]) == float(SR_hist[-1])


# --------------------------------------------------------------------------- #
# direct_phase: 理想器件逐像素整形
# --------------------------------------------------------------------------- #
class TestDirectPhaseShaping:
    def test_smoke_and_energy_improves(self):
        turb = _turb_phase()
        inf_flat = _inf_flat()
        target = make_square_target(half_width=6).numpy()
        u, dm_u, loss_hist, eng_hist, I_np = target_shaping_optimization(
            turb, inf_flat, target, max_iter=8, lr=0.05, seed=params.seed_spgd,
            label="square", direct_phase=True,
        )
        assert u.shape == (params.N * params.N,)
        assert dm_u.shape == (params.N, params.N)
        assert np.isfinite(loss_hist).all()
        assert loss_hist[-1] < loss_hist[0]
        assert eng_hist[-1] >= eng_hist[0]
        assert I_np.shape == (params.N, params.N)


# --------------------------------------------------------------------------- #
# run.py CLI 集成 (动态导入, 避免 tests 目录 sys.path 依赖)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run_main():
    spec = importlib.util.spec_from_file_location("_run_cli", ROOT / "run.py")
    assert spec is not None and spec.loader is not None
    run_mod = importlib.util.module_from_spec(spec)
    sys.modules["_run_cli"] = run_mod
    spec.loader.exec_module(run_mod)
    return run_mod.main


@pytest.fixture(scope="module")
def cli_runner():
    from click.testing import CliRunner

    return CliRunner()


class TestRunCliDevices:
    def test_point_ideal_analytic(self, run_main, cli_runner, tmp_path):
        out = tmp_path / "fig.png"
        result = cli_runner.invoke(
            run_main, ["--shape", "point", "--device", "ideal", "--out", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_square_ideal_torch_gd(self, run_main, cli_runner, tmp_path):
        out = tmp_path / "fig2.png"
        result = cli_runner.invoke(
            run_main,
            [
                "--shape", "square", "--device", "ideal", "--algorithm", "torch-gd",
                "--max-iter", "6", "--out", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_slm_point_torch_gd(self, run_main, cli_runner, tmp_path):
        out = tmp_path / "fig3.png"
        result = cli_runner.invoke(
            run_main,
            [
                "--shape", "point", "--device", "slm", "--algorithm", "torch-gd",
                "--fill-factor", "0.8", "--max-iter", "6", "--out", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_ideal_shaping_requires_torch_gd(self, run_main, cli_runner):
        result = cli_runner.invoke(
            run_main,
            ["--shape", "square", "--device", "ideal", "--algorithm", "spgd"],
        )
        assert result.exit_code == 2

    def test_quantize_requires_slm(self, run_main, cli_runner):
        result = cli_runner.invoke(
            run_main, ["--shape", "point", "--device", "dm", "--quantize"]
        )
        assert result.exit_code == 2

    def test_slm_quantize_torch_gd_runs(self, run_main, cli_runner, tmp_path):
        # 量化 + torch-gd 会警告不收敛, 但管线应完整跑通 (梯度为零, 见
        # TestTorchGdUnderSlm.test_quantize_zeroes_gradient)。
        out = tmp_path / "fig4.png"
        result = cli_runner.invoke(
            run_main,
            [
                "--shape", "point", "--device", "slm", "--algorithm", "torch-gd",
                "--quantize", "--max-iter", "2", "--out", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()