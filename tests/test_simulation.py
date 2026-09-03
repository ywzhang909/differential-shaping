# -*- coding: utf-8 -*-
"""Tests for the canonical PyTorch physics layer (wavefront-sensorless AO)."""

import numpy as np
import pytest
import torch

from differential_shaping import params
from differential_shaping.simulation import (
    generate_turbulence_phase,
    generate_influence_functions,
    remove_piston,
    compute_metrics,
    far_field_intensity_metric,
    far_field_intensity_padded,
    crop_center,
    dm_surface,
    to_numpy,
    to_torch,
    pupil_mask,
    pupil_float_t,
    generate_actuator_grid,
)


# --------------------------------------------------------------------------- #
# pupil
# --------------------------------------------------------------------------- #
class TestPupil:
    def test_pupil_mask_shape_and_bool(self):
        assert pupil_mask.shape == (params.N, params.N)
        assert pupil_mask.dtype == torch.bool
        # circular aperture is non-empty, and there is an off-disc region
        assert pupil_mask.any() and (~pupil_mask).any()

    def test_pupil_float_dtype(self):
        assert pupil_float_t.shape == (params.N, params.N)
        assert pupil_float_t.dtype == torch.float32
        # aperture value 1.0 inside, 0.0 outside
        assert bool((pupil_float_t[pupil_mask] == 1.0).all())
        assert bool((pupil_float_t[~pupil_mask] == 0.0).all())


# --------------------------------------------------------------------------- #
# turbulence
# --------------------------------------------------------------------------- #
class TestTurbulence:
    def test_shape_dtype_device(self):
        tp = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
        )
        assert tp.shape == (params.N, params.N)
        assert tp.dtype == torch.float32
        assert tp.device.type == "cpu"

    def test_deterministic_given_seed(self):
        a = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms, 7
        )
        b = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms, 7
        )
        assert torch.equal(a, b)

    def test_pupil_rms_matches_target(self):
        tp = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
        )
        rms = float(tp[pupil_mask].std())
        assert rms == pytest.approx(params.target_phase_rms, abs=1e-3)

    def test_outside_pupil_is_zero(self):
        tp = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
        )
        assert bool((tp[~pupil_mask] == 0.0).all())

    def test_remove_piston(self):
        tp = generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
        )
        rp = remove_piston(tp)
        assert torch.equal(rp[~pupil_mask], torch.zeros_like(rp[~pupil_mask]))
        assert float(rp[pupil_mask].mean()) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# deformable mirror
# --------------------------------------------------------------------------- #
class TestDM:
    def test_actuator_grid_counts(self):
        ax, ay = generate_actuator_grid()
        assert ax.shape == (params.n_act,) and ay.shape == (params.n_act,)
        assert ax.dtype == torch.float32

    def test_influence_function_shape_and_norm(self):
        inf = generate_influence_functions()
        assert inf.shape == (params.n_act, params.N, params.N)
        assert inf.dtype == torch.float32
        assert bool(torch.isfinite(inf).all())
        # each influence function is peak-normalised to abs max = 1.0
        for i in range(params.n_act):
            assert float(inf[i].abs().max()) == pytest.approx(1.0, rel=1e-2)


# --------------------------------------------------------------------------- #
# optics (metrics + far field + differentiable DM surface)
# --------------------------------------------------------------------------- #
class TestOptics:
    def _turb(self):
        return generate_turbulence_phase(
            params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
        )

    def test_compute_metrics_types_and_ranges(self):
        tp = self._turb()
        J, SR, I = compute_metrics(tp)
        assert J.dtype == torch.float32
        assert SR.dtype == torch.float32
        assert I.shape == (params.N, params.N)
        # no correction at r0=60um / D=1.28mm: a degraded but sensible SR
        assert 0.0 < float(SR) < 1.0
        assert float(SR) > 0.1
        assert float(J) > 0.0

    def test_zero_dm_leaves_metrics_unchanged(self):
        tp = self._turb()
        inf = generate_influence_functions()
        inf_flat = inf.reshape(params.n_act, -1).t()  # (N*N, n_act) for dm_surface
        zero_u = torch.zeros(params.n_act, dtype=torch.float32)
        dm = dm_surface(zero_u, inf_flat)
        assert bool((dm == 0.0).all())
        J0, SR0, _ = compute_metrics(tp)
        J1, SR1, _ = compute_metrics(tp + dm)
        assert float(J0) == pytest.approx(float(J1), abs=1e-5)
        assert float(SR0) == pytest.approx(float(SR1), abs=1e-5)

    def test_far_field_padded_shape(self):
        tp = self._turb()
        Ipad = far_field_intensity_padded(tp)
        assert Ipad.shape == (params.N * params.pad_factor_show,) * 2

    def test_crop_center_extent(self):
        tp = self._turb()
        Ipad = far_field_intensity_padded(tp)
        crop, extent = crop_center(Ipad, params.spot_half_width_lamD)
        assert crop.shape == (
            2 * params.spot_half_width_lamD + 1,
            2 * params.spot_half_width_lamD + 1,
        )
        assert extent[-1] == pytest.approx(
            params.spot_half_width_lamD / params.pad_factor_show, rel=1e-2
        )

    def test_backprop_through_metrics_produces_finite_gradients(self):
        """The differentiable core: gradients of a Gaussian-weighted on-axis
        energy loss flow back to the DM command vector (the basis of Torch-GD)."""
        tp = self._turb()
        inf = generate_influence_functions()
        inf_flat = inf.reshape(params.n_act, -1).t()  # (N*N, n_act)
        u = torch.zeros(params.n_act, dtype=torch.float32, requires_grad=True)
        dm = dm_surface(u, inf_flat)
        # Gaussian-weighted on-axis energy (maximise -> minimise negative)
        E = pupil_float_t * torch.exp(1j * remove_piston(tp + dm))
        I = torch.abs(torch.fft.fftshift(torch.fft.fft2(E))) ** 2
        from differential_shaping.simulation import gaussian_window

        loss = -(gaussian_window * I).sum()
        loss.backward()
        assert u.grad is not None
        assert u.grad.shape == (params.n_act,)
        assert bool(torch.isfinite(u.grad).all())
        assert float(u.grad.abs().max()) > 0.0

    def test_numpy_torch_conversions(self):
        tp = self._turb()
        arr = to_numpy(tp)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        t2 = to_torch(arr, requires_grad=True)
        assert t2.dtype == torch.float32
        assert t2.requires_grad