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
)


@pytest.fixture(scope="module")
def _shared():
    """Shared turbulence + DM (module-scoped to avoid recomputing)."""
    turb = generate_turbulence_phase(
        params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
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
        res = torch_gd_optimization(turb, inf_flat, max_iter=300, lr=0.01, seed=params.seed_spgd)
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
            turb.detach().cpu().numpy(), inf_flat.detach().cpu().numpy(), max_iter=20, lr=0.01
        )
        assert res_t[2].shape == res_n[2].shape == (20,)
        assert np.allclose(res_t[2], res_n[2], atol=1e-3)

    def test_snapshot_store_populated(self, _shared):
        turb, inf_flat, _, _ = _shared
        store = []
        frames = [0, 5, 10]
        torch_gd_optimization(
            turb, inf_flat, max_iter=12, seed=params.seed_spgd,
            snapshot_indices=frames, snapshot_store=store,
        )
        assert len(store) == len(frames)
        for it, u_snap in store:
            assert u_snap.shape == (params.n_act,)