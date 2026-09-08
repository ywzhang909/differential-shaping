# -*- coding: utf-8 -*-
"""End-to-end optimizer regression tests (numpy float64 boundary)."""

import numpy as np
import pytest
import torch

from differential_shaping import params
from differential_shaping.simulation import (
    generate_turbulence_phase,
    generate_influence_functions,
    compute_metrics,
)
from differential_shaping.optimization import (
    spgd_optimization,
    hgd_optimization,
    torch_gd_optimization,
    gs_optimization,
    gs_shaping_optimization,
    make_square_target,
    make_triangle_target,
    target_shaping_optimization,
    spgd_shaping_optimization,
    hgd_shaping_optimization,
    energy_in_target,
)


@pytest.fixture(scope="module")
def _shared():
    """Shared turbulence + DM (module-scoped to avoid recomputing)."""
    turb = generate_turbulence_phase(
        params.N,
        params.pixel_size,
        params.r0,
        params.target_phase_rms,
        params.seed_phase,
    )
    inf = generate_influence_functions()
    inf_flat = inf.reshape(params.n_act, -1)  # (n_act, N*N)
    J0, SR0, _ = compute_metrics(turb)
    return turb, inf_flat, float(J0), float(SR0)


def _assert_optimizer_result(res, n_act, N):
    u, dm_u, J_hist, SR_hist = res
    assert isinstance(u, np.ndarray) and u.dtype == np.float64
    assert isinstance(dm_u, np.ndarray) and dm_u.dtype == np.float64
    assert isinstance(J_hist, np.ndarray) and J_hist.dtype == np.float64
    assert isinstance(SR_hist, np.ndarray) and SR_hist.dtype == np.float64
    assert u.shape == (n_act,)
    assert dm_u.shape == (N, N)
    assert J_hist.shape == SR_hist.shape == (len(J_hist),)


class TestOptimizerRegression:
    def test_spgd_improves_strehl(self, _shared):
        turb, inf_flat, J0, SR0 = _shared
        res = spgd_optimization(turb, inf_flat)
        u, dm_u, J_hist, SR_hist = res
        _assert_optimizer_result(res, params.n_act, params.N)
        assert len(J_hist) == params.max_iter
        # SPGD should substantially correct the wavefront (SR >> initial)
        assert SR_hist[-1] > 0.7
        assert J_hist[-1] < J0
        assert SR_hist[-1] > SR0

    def test_hgd_improves_strehl(self, _shared):
        turb, inf_flat, J0, SR0 = _shared
        res = hgd_optimization(turb, inf_flat)
        u, dm_u, J_hist, SR_hist = res
        _assert_optimizer_result(res, params.n_act, params.N)
        assert len(J_hist) == params.max_iter
        assert SR_hist[-1] > 0.8
        assert J_hist[-1] < J0
        assert SR_hist[-1] > SR0

    def test_torch_gd_improves_strehl(self, _shared):
        turb, inf_flat, J0, SR0 = _shared
        res = torch_gd_optimization(
            turb, inf_flat, max_iter=300, lr=0.01, seed=params.seed_spgd
        )
        u, dm_u, J_hist, SR_hist = res
        _assert_optimizer_result(res, params.n_act, params.N)
        assert len(J_hist) == 300
        assert SR_hist[-1] > 0.8
        assert J_hist[-1] < J0
        assert SR_hist[-1] > SR0

    def test_optimizers_accept_both_numpy_and_torch(self, _shared):
        turb, inf_flat, J0, SR0 = _shared
        # torch tensors path (simulation layer is torch-canonical)
        res_t = torch_gd_optimization(turb, inf_flat, max_iter=20, lr=0.01)
        # numpy float64 path (back-compat)
        res_n = torch_gd_optimization(
            turb.detach().cpu().numpy(),
            inf_flat.detach().cpu().numpy(),
            max_iter=20,
            lr=0.01,
        )
        assert res_t[2].shape == res_n[2].shape == (20,)
        assert np.allclose(res_t[2], res_n[2], atol=1e-3)

    def test_snapshot_store_populated(self, _shared):
        turb, inf_flat, _, _ = _shared
        store = []
        frames = [0, 5, 10]
        torch_gd_optimization(
            turb,
            inf_flat,
            max_iter=12,
            seed=params.seed_spgd,
            snapshot_indices=frames,
            snapshot_store=store,
        )
        assert len(store) == len(frames)
        for it, u_snap in store:
            assert u_snap.shape == (params.n_act,)


class TestTargetShaping:
    def test_square_target_geometry(self):
        sq = make_square_target(half_width=6)
        assert sq.shape == (params.N, params.N)
        assert sq.dtype == torch.float32
        # a 13x13 square centred at (N//2, N//2)
        c = params.N // 2
        assert sq[c, c].item() == 1.0
        assert sq[c, c + 6].item() == 1.0 and sq[c, c + 7].item() == 0.0
        assert int(sq.sum().item()) == 13 * 13

    def test_triangle_target_geometry(self):
        tri = make_triangle_target(size=11, apex="up")
        assert tri.shape == (params.N, params.N)
        assert tri.dtype == torch.float32
        c = params.N // 2
        # apex-up triangle: top tip present, far outside (bottom corners) absent
        assert tri[c - 5, c].item() == 1.0  # apex (top)
        assert tri[c + 5, c].item() == 1.0  # base centre (bottom)
        assert tri[c + 5, c + 6].item() == 0.0  # outside triangle base half-width
        area = int(tri.sum().item())
        assert 0 < area < 11 * 11  # triangle is a proper subset of its box

    def test_energy_in_target_bounds(self):
        sq = make_square_target(half_width=6)
        # A uniform field has exactly area/total = 169/128^2 energy in target.
        uniform = torch.ones(params.N, params.N, dtype=torch.float32)
        e = energy_in_target(uniform, sq)
        assert abs(float(e) - 169 / (params.N * params.N)) < 1e-3

    def test_shaping_converges(self, _shared):
        turb, inf_flat, _, _ = _shared
        tgt = make_square_target(half_width=5)
        res = target_shaping_optimization(
            turb,
            inf_flat,
            tgt.numpy(),
            max_iter=120,
            lr=0.02,
            seed=params.seed_spgd,
            label="square",
        )
        u, dm_u, loss_hist, energy_hist, I_np = res
        assert isinstance(u, np.ndarray) and u.shape == (params.n_act,)
        assert dm_u.shape == (params.N, params.N)
        assert loss_hist.shape == energy_hist.shape == (120,)
        assert np.all(np.isfinite(I_np))
        # The differentiable shapber should push more conserved energy into the target.
        assert energy_hist[-1] > energy_hist[0] - 1e-6
        assert I_np.shape == (params.N, params.N)

    def test_shaping_snapshot_store(self, _shared):
        turb, inf_flat, _, _ = _shared
        tgt = make_triangle_target(size=9, apex="up")
        store = []
        target_shaping_optimization(
            turb,
            inf_flat,
            tgt.numpy(),
            max_iter=20,
            seed=params.seed_spgd,
            snapshot_indices=[0, 5, 19],
            snapshot_store=store,
            label="triangle",
        )
        assert len(store) == 3
        for it, u_snap in store:
            assert u_snap.shape == (params.n_act,) and u_snap.dtype == np.float64

    def test_spgd_shaping_returns_contract(self, _shared):
        turb, inf_flat, _, _ = _shared
        tgt = make_square_target(half_width=5).numpy()
        u, dm_u, loss_hist, energy_hist, I_np = spgd_shaping_optimization(
            turb,
            inf_flat,
            tgt,
            max_iter=120,
            label="square",
        )
        assert u.shape == (params.n_act,) and u.dtype == np.float64
        assert dm_u.shape == (params.N, params.N)
        assert loss_hist.shape == energy_hist.shape == (120,)
        assert np.all(np.isfinite(energy_hist)) and np.all(np.isfinite(I_np))
        # SPGD numeric-gradient shaping minimises the shared loss (non-increasing)
        # but is too weak to move the conserved energy (energy-in-target stays flat).
        assert loss_hist[-1] <= loss_hist[0] + 1e-12
        assert abs(energy_hist[-1] - energy_hist[0]) < 0.05

    def test_hgd_shaping_returns_contract(self, _shared):
        turb, inf_flat, _, _ = _shared
        tgt = make_triangle_target(size=9, apex="down").numpy()
        u, dm_u, loss_hist, energy_hist, I_np = hgd_shaping_optimization(
            turb,
            inf_flat,
            tgt,
            max_iter=120,
            label="triangle",
        )
        assert u.shape == (params.n_act,) and u.dtype == np.float64
        assert dm_u.shape == (params.N, params.N)
        assert loss_hist.shape == energy_hist.shape == (120,)
        assert np.all(np.isfinite(energy_hist)) and np.all(np.isfinite(I_np))
        assert loss_hist[-1] <= loss_hist[0] + 1e-12
        assert abs(energy_hist[-1] - energy_hist[0]) < 0.05


class TestGSOptimizer:
    """Gerchberg-Saxton point-correction (Strehl) regression tests."""

    def test_gs_improves_strehl(self, _shared):
        turb, inf_flat, J0, SR0 = _shared
        res = gs_optimization(turb, inf_flat)
        u, dm_u, J_hist, SR_hist = res
        _assert_optimizer_result(res, params.n_act, params.N)
        assert len(J_hist) == params.max_iter
        # GS computes a phase that focuses the spot; it should substantially
        # improve Strehl over the uncorrected (turbulent) wavefront.
        assert SR_hist[-1] > 0.7
        assert J_hist[-1] < J0
        assert SR_hist[-1] > SR0

    def test_gs_accepts_both_numpy_and_torch(self, _shared):
        turb, inf_flat, _, _ = _shared
        res_t = gs_optimization(turb, inf_flat, max_iter=20)
        res_n = gs_optimization(
            turb.detach().cpu().numpy(), inf_flat.detach().cpu().numpy(), max_iter=20
        )
        assert res_t[2].shape == res_n[2].shape == (20,)
        # Both input conventions should reach the same converged solution.
        assert np.allclose(res_t[2], res_n[2], atol=1e-3)

    def test_gs_snapshot_store_populated(self, _shared):
        turb, inf_flat, _, _ = _shared
        store = []
        frames = [0, 5, 10]
        gs_optimization(
            turb,
            inf_flat,
            max_iter=12,
            snapshot_indices=frames,
            snapshot_store=store,
        )
        assert len(store) == len(frames)
        for it, u_snap in store:
            assert u_snap.shape == (params.n_act,)
            assert u_snap.dtype == np.float64


class TestGSShaping:
    """Gerchberg-Saxton far-field beam-shaping regression tests."""

    def test_gs_shaping_returns_contract(self, _shared):
        turb, inf_flat, _, _ = _shared
        tgt = make_square_target(half_width=5).numpy()
        u, dm_u, loss_hist, energy_hist, I_np = gs_shaping_optimization(
            turb,
            inf_flat,
            tgt,
            max_iter=120,
            label="square",
        )
        assert u.shape == (params.n_act,) and u.dtype == np.float64
        assert dm_u.shape == (params.N, params.N)
        assert loss_hist.shape == energy_hist.shape == (120,)
        assert np.all(np.isfinite(energy_hist)) and np.all(np.isfinite(I_np))
        assert I_np.shape == (params.N, params.N)
        # GS imprints the target amplitude in the focal plane -> energy in the
        # target region should grow (or at least not drop) over the run.
        assert energy_hist[-1] > energy_hist[0] - 1e-6

    def test_gs_shaping_snapshot_store(self, _shared):
        turb, inf_flat, _, _ = _shared
        tgt = make_triangle_target(size=9, apex="up").numpy()
        store = []
        gs_shaping_optimization(
            turb,
            inf_flat,
            tgt,
            max_iter=20,
            snapshot_indices=[0, 5, 19],
            snapshot_store=store,
            label="triangle",
        )
        assert len(store) == 3
        for it, u_snap in store:
            assert u_snap.shape == (params.n_act,)
            assert u_snap.dtype == np.float64
