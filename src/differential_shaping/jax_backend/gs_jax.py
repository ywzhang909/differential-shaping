# -*- coding: utf-8 -*-
"""
JAX / chromatix version of ``optimization.gs`` — Gerchberg-Saxton (GS).

Projection-based iterative phase computation, ported one-to-one from the torch
GS.  The only change is the backend: the forward/inverse transforms use
``chromatix.functional.fft`` (``fx.fft``) and the DM actuator projection uses
``jax.numpy`` least squares.

GS alternates pupil <-> focal plane, imposing an amplitude constraint at each
(step below); after each iteration the required pupil phase is wrapped back onto
the deformable mirror by a least-squares fit through the influence functions:

    dm_u = inf_flat^T @ pinv(inf_flat^T) @ (phi_gs - turb)

Two public entry points, mirroring the torch API exactly:

    gs_optimization            -> (u, dm_u, J_hist, SR_hist)   [point / Strehl]
    gs_shaping_optimization    -> (u, dm_u, loss_hist, energy_hist, I_np)  [square/triangle/...]

The shaping entry drives the focal amplitude toward ``sqrt(target)``; the
Airy point focus is the special case used by ``gs_optimization``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger

from differential_shaping.params import N, n_act
from differential_shaping.params import max_iter as _default_max_iter

from .optics_jax import (
    compute_metrics,
    far_field_field,
    far_field_intensity_metric,
    to_jax,
    to_numpy,
)
from .pupil_jax import I0_jax, pupil_float_jax
from .turbulence_jax import remove_piston

__all__ = ["gs_optimization", "gs_shaping_optimization", "recover_u_jax"]


# ---------------------------------------------------------------------------
# Core GS step (jax)
# ---------------------------------------------------------------------------
def _gs_refine_phase(phase: jax.Array, focal_amp: jax.Array) -> jax.Array:
    """One Gerchberg-Saxton iteration -> refined pupil phase.

    ``phase`` is the total pupil phase (turb + dm); ``focal_amp`` is the desired
    focal-plane amplitude.  Forwards to the focal plane, replaces the focal
    amplitude with the target (keeping the focal phase), inverts back, and
    returns the new pupil phase.
    """
    # Forward: pupil complex field -> focal field (chromatix fft, shifted).
    E = pupil_float_jax * jnp.exp(1j * remove_piston(phase))
    g = far_field_field(phase)
    phi = jnp.angle(g)
    # Focal-plane amplitude constraint.
    g_new = focal_amp * jnp.exp(1j * phi)
    # Inverse transform (fftshift then ifft2).
    E_new = jnp.fft.ifft2(jnp.fft.ifftshift(g_new))
    return jnp.angle(E_new)


def recover_u_jax(inf_flat_T: jax.Array, dm_u: jax.Array) -> jax.Array:
    """Recover actuator commands for a DM surface via least squares (JAX)."""
    A = inf_flat_T  # (N*N, n_act)
    pinv = jnp.linalg.pinv(A)  # (n_act, N*N)
    return pinv @ dm_u.reshape(-1)


def _project_onto_dm(
    target_dm: jax.Array, inf_flat_T: jax.Array, pinv: jax.Array
) -> jax.Array:
    """Least-squares fit of a desired DM surface onto the actuator basis."""
    clean = remove_piston(target_dm)
    u = pinv @ clean.reshape(-1)
    return (inf_flat_T @ u).reshape(N, N)


# Ideal Airy focal amplitude: mirrors the torch GS reference exactly.
#
# The torch GS uses ``focal_amp = sqrt(clamp(I0_t, min=0))`` where ``I0_t`` is
# the diffraction-limited focal *field* (``fftshift(fft2(pupil))``).  Here
# ``I0_jax`` is the same field, so we use ``sqrt(clamp(I0_jax, min=0))``.
# This is a *spatially-varying* focal-plane amplitude template (bright at
# the Airy core, tapering through the rings, ~0 in the far field), NOT the
# scalar ``max|I0|`` — which would force a constant amplitude everywhere and
# diverge.
_AIRY_AMP = jnp.abs(I0_jax).astype(jnp.float32)


def _precompute_dm_basis(inf_flat: jax.Array):
    """Return (inf_flat_T, pinv) for the DM projection (float32).

    ``inf_flat`` is (n_act, N*N); the transposed basis is (N*N, n_act) so that
    ``dm_surface = inf_flat_T @ u`` and ``u = pinv @ dm_surface.reshape(-1)``.
    """
    inf_flat_T = inf_flat.T  # (N*N, n_act)
    pinv = jnp.linalg.pinv(inf_flat_T)  # (n_act, N*N)
    return inf_flat_T, pinv


# ---------------------------------------------------------------------------
# Point correction (Strehl)
# ---------------------------------------------------------------------------
def gs_optimization(
    turb_phase,
    inf_flat,
    max_iter: int = _default_max_iter,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correct the wavefront to a point via Gerchberg-Saxton (JAX)."""
    turb = to_jax(turb_phase)
    inf_flat = to_jax(np.asarray(inf_flat, dtype=np.float64))  # (n_act, N*N)
    inf_flat_T, pinv = _precompute_dm_basis(inf_flat)

    dm_u = jnp.zeros((N, N), dtype=jnp.float32)

    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, np.zeros(n_act, dtype=np.float64)))

    for it in range(max_iter):
        phase = turb + dm_u
        phase_new = _gs_refine_phase(phase, focal_amp=_AIRY_AMP)
        dm_u = _project_onto_dm(phase_new - turb, inf_flat_T, pinv)

        J, SR, _ = compute_metrics(turb + dm_u)
        J_hist[it] = float(J)
        SR_hist[it] = float(SR)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(recover_u_jax(inf_flat_T, dm_u)).copy()))

        if it % 100 == 0 or it == max_iter - 1:
            logger.info(f"GS[jax] {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}")

    u_np = to_numpy(recover_u_jax(inf_flat_T, dm_u))
    dm_np = to_numpy(dm_u)
    return u_np, dm_np, J_hist, SR_hist


# ---------------------------------------------------------------------------
# Far-field beam shaping (square / triangle / ... )
# ---------------------------------------------------------------------------
def energy_in_target_jax(intensity: jax.Array, target: jax.Array) -> jax.Array:
    """Fraction of conserved far-field energy captured inside the target."""
    return (intensity * target).sum() / (intensity.sum() + 1e-12)


def gs_shaping_optimization(
    turb_phase,
    inf_flat,
    target,
    max_iter: int = _default_max_iter,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
    label: str | None = None,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the far-field spot into ``target`` via Gerchberg-Saxton (JAX)."""
    turb = to_jax(turb_phase)
    inf_flat = to_jax(np.asarray(inf_flat, dtype=np.float64))  # (n_act, N*N)
    inf_flat_T, pinv = _precompute_dm_basis(inf_flat)

    target_t = jnp.clip(to_jax(target), min=0.0)
    target_n = target_t / (target_t.sum() + 1e-12)
    focal_amp = jnp.sqrt(target_t)
    log_name = label or "gs-shaping"

    dm_u = jnp.zeros((N, N), dtype=jnp.float32)

    loss_hist = np.zeros(max_iter)
    energy_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, np.zeros(n_act, dtype=np.float64)))

    for it in range(max_iter):
        phase = turb + dm_u
        phase_new = _gs_refine_phase(phase, focal_amp=focal_amp)
        dm_u = _project_onto_dm(phase_new - turb, inf_flat_T, pinv)

        I_eval = far_field_intensity_metric(turb + dm_u)
        I_n = I_eval / (I_eval.sum() + 1e-12)
        energy_hist[it] = float(energy_in_target_jax(I_eval, target_t))
        loss_match = jnp.mean((I_n - target_n) ** 2)
        loss_hist[it] = float(
            loss_weight_match * loss_match - loss_weight_capture * energy_hist[it]
        )

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(recover_u_jax(inf_flat_T, dm_u)).copy()))

        if it % 200 == 0 or it == max_iter - 1:
            logger.info(
                f"GS-shaping[jax][{log_name}] {it:4d}: "
                f"loss={loss_hist[it]:.4e}, energy-in-target={energy_hist[it]:.3f}"
            )

    I_final = far_field_intensity_metric(turb + dm_u)
    u_np = to_numpy(recover_u_jax(inf_flat_T, dm_u))
    dm_np = to_numpy(dm_u)
    I_np = to_numpy(I_final / (I_final.sum() + 1e-12))

    return u_np, dm_np, loss_hist, energy_hist, I_np
