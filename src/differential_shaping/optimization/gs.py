# -*- coding: utf-8 -*-
"""
Gerchberg-Saxton (GS) phase-computation optimisers.

Unlike the gradient/perturbative methods in this package (SPGD, H-GD, Torch-GD),
the GS algorithm is a **projection-based** iterative scheme that directly
computes the pupil phase needed to produce a desired far-field intensity.

GS alternates between two planes, applying an amplitude constraint at each:

    1. **Forward**  ``E = pupil * exp(i * phase)``  ->  ``g = FFT(E)``
    2. **Focal constraint** -- replace the focal-plane *amplitude* with the
       target (``sqrt(target_intensity)`` for beam shaping; the ideal Airy
       amplitude for a focused ``point``), keeping the current focal phase.
    3. **Inverse**  ``E' = IFFT(g)``, extract the new pupil phase.
    4. **Pupil constraint** -- force unit amplitude inside the pupil, keep the
       extracted phase.

After each GS step the resulting pupil phase is wrapped back onto the
deformable mirror by a least-squares fit through the influence functions:

    dm_u  = pinv(inf_flat^T) applied to (phi_gs - turb)

so the public API matches ``spgd_optimization`` (point correction, 4-tuple)
and the shaping optimisers (5-tuple) for drop-in use by the run.py pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.fft
from loguru import logger

from differential_shaping.params import N, n_act
from differential_shaping.params import max_iter as _default_max_iter
from differential_shaping.simulation.optics import (
    compute_metrics,
    far_field_intensity_metric,
    remove_piston,
    to_numpy,
)
from differential_shaping.simulation.pupil import I0_t, pupil_float_t
from differential_shaping.optimization.shaping import energy_in_target

__all__ = ["gs_optimization", "gs_shaping_optimization"]

_DEVICE = torch.device("cpu")
_DTYPE = torch.float32


def _as_tensor(a) -> torch.Tensor:
    """Convert a numpy array or CPU torch tensor to a float32 CPU tensor."""
    if isinstance(a, torch.Tensor):
        return a.to(dtype=_DTYPE, device=_DEVICE)
    return torch.from_numpy(np.asarray(a, dtype=np.float64)).to(
        dtype=_DTYPE, device=_DEVICE
    )


# ---------------------------------------------------------------------------
# Core GS step
# ---------------------------------------------------------------------------
def _gs_refine_phase(phase: torch.Tensor, focal_amp: torch.Tensor) -> torch.Tensor:
    """One Gerchberg-Saxton iteration returning the refined pupil phase.

    Args:
        phase:      (N, N) current total pupil phase (rad).
        focal_amp:  (N, N) desired focal-plane *amplitude* target.

    Returns:
        (N, N) refined pupil phase (wrapped to ``[-pi, pi]``) such that
        ``pupil * exp(i * phase_new)`` produces ``focal_amp`` in the plane.
    """
    # Forward propagation to the focal plane.
    E = pupil_float_t * torch.exp(1j * remove_piston(phase))
    g = torch.fft.fftshift(torch.fft.fft2(E))

    phi = torch.angle(g)

    # Focal-plane amplitude constraint: imprint the target amplitude, keep the
    # current focal phase (the classic GS "replace amplitude" projection).
    g_new = focal_amp * torch.exp(1j * phi)

    # Inverse propagation back to the pupil, keep the phase.
    E_new = torch.fft.ifft2(torch.fft.ifftshift(g_new))
    return torch.angle(E_new)


def _project_onto_dm(
    target_dm: torch.Tensor,
    inf_flat_T: torch.Tensor,
    pinv: torch.Tensor,
) -> torch.Tensor:
    """Least-squares fit of a desired DM surface onto the actuator basis.

    ``u = pinv @ target.flatten()`` then ``dm_u = inf_flat_T @ u``.  The DM
    surface is piston-removed before the fit so the solution is well posed.

    Returns:
        dm_u : (N, N) actuators-projected DM surface.
    """
    clean = remove_piston(target_dm)
    u = pinv @ clean.reshape(-1)
    return (inf_flat_T @ u).reshape(N, N)


# The ideal Airy focal amplitude: the amplitude of the diffraction-limited
# ``|FFT(pupil)|`` spot that a flat wavefront produces.  A point-focus GS drives
# the aberrated field toward exactly this — the tightest spot the pupil allows.
_AIRY_AMP = torch.sqrt(torch.clamp(I0_t, min=0.0)).to(dtype=_DTYPE)


# ---------------------------------------------------------------------------
# Point correction (Strehl)
# ---------------------------------------------------------------------------
def gs_optimization(
    turb_phase: np.ndarray | torch.Tensor,
    inf_flat: np.ndarray | torch.Tensor,
    max_iter: int = _default_max_iter,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correct the wavefront to a point via Gerchberg-Saxton phase computation.

    GS iteratively shapes the focal field toward the ideal diffraction-limited
    (Airy) spot, extracts the required pupil phase and projects it onto the DM
    actuators.  Maximising the match to the Airy spot maximises on-axis energy,
    i.e. the Strehl ratio.  Returns the identical ``(u, dm_u, J_hist, SR_hist)``
    contract as the other point-correction optimisers.

    Args:
        turb_phase: (N, N) float64 turbulence phase (rad).
        inf_flat:   (n_act, N*N) flattened DM influence functions.
        max_iter:   number of GS iterations.
        snapshot_indices: iteration indices at which to record ``(iter, u)``.
        snapshot_store:   mutable list appended with ``(iter, u)`` snapshots.

    Returns:
        (u, dm_u, J_hist, SR_hist):
            u       (n_act,)  float64 final DM commands.
            dm_u    (N, N)    float64 final DM surface.
            J_hist  (max_iter,) float64 centroid-radius history.
            SR_hist (max_iter,) float64 Strehl-ratio history.
    """
    turb_t = _as_tensor(turb_phase)  # (N, N)
    inf_flat_t = _as_tensor(inf_flat)  # (n_act, N*N)
    inf_flat_T = inf_flat_t.t().contiguous()  # (N*N, n_act)
    pinv = torch.linalg.pinv(inf_flat_T)  # (n_act, N*N)

    u = torch.zeros(n_act, dtype=_DTYPE)
    dm_u = torch.zeros((N, N), dtype=_DTYPE)

    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, to_numpy(u).copy()))

    for it in range(max_iter):
        phase = turb_t + dm_u
        # GS step toward the ideal Airy spot (tight point focus).
        phase_new = _gs_refine_phase(phase, focal_amp=_AIRY_AMP)
        # Project the required DM surface onto the actuators.
        dm_u = _project_onto_dm(phase_new - turb_t, inf_flat_T, pinv)

        with torch.no_grad():
            J, SR, _ = compute_metrics(turb_t + dm_u)
            J_hist[it] = to_numpy(J)
            SR_hist[it] = to_numpy(SR)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, recover_u(inf_flat_T, dm_u)))

        if it % 100 == 0 or it == max_iter - 1:
            logger.info(f"GS {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}")

    # Recover the actuator commands from the final DM surface (least squares).
    u_np = recover_u(inf_flat_T, dm_u)
    dm_np = to_numpy(dm_u)

    return u_np, dm_np, J_hist, SR_hist


def recover_u(inf_flat_T: torch.Tensor, dm_u: torch.Tensor) -> np.ndarray:
    """Recover actuator commands for a DM surface via least squares (numpy)."""
    return to_numpy(torch.linalg.lstsq(inf_flat_T, dm_u.reshape(-1)).solution)


# ---------------------------------------------------------------------------
# Far-field beam shaping
# ---------------------------------------------------------------------------
def gs_shaping_optimization(
    turb_phase: np.ndarray | torch.Tensor,
    inf_flat: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    max_iter: int = _default_max_iter,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
    label: str | None = None,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the far-field spot into ``target`` via Gerchberg-Saxton.

    The focal-plane amplitude is driven toward ``sqrt(target)`` each GS step
    while conserving the total energy, then projected onto the DM actuators.
    Returns the same ``(u, dm_u, loss_hist, energy_hist, I_np)`` 5-tuple
    contract as the other shaping optimisers.

    Args:
        turb_phase: (N, N) float64 turbulence phase (rad).
        inf_flat:   (n_act, N*N) flattened DM influence functions.
        target:     (N, N) non-negative target intensity (numpy or torch).
        max_iter:   number of GS iterations.
        loss_weight_match / loss_weight_capture: reported-loss blend (kept for
                    API parity with the other shaping optimisers).
        label:      optional human-readable name for logging.
        snapshot_indices / snapshot_store: optional ``(iter, u)`` checkpointing.

    Returns:
        (u, dm_u, loss_hist, energy_hist, I_np):
            u          (n_act,)  float64 final DM commands.
            dm_u       (N, N)    float64 final DM surface.
            loss_hist  (max_iter,) float64 shaping-loss history.
            energy_hist (max_iter,) float64 energy-in-target fraction history.
            I_np       (N, N)    float64 final normalised far-field intensity.
    """
    turb_t = _as_tensor(turb_phase)
    inf_flat_t = _as_tensor(inf_flat)
    inf_flat_T = inf_flat_t.t().contiguous()
    pinv = torch.linalg.pinv(inf_flat_T)

    target_t = _as_tensor(target)
    target_t = torch.clamp(target_t, min=0.0)
    target_n = target_t / (target_t.sum() + 1e-12)
    # GS works on focal *amplitude* = sqrt of the target intensity.
    focal_amp = torch.sqrt(target_t)
    log_name = label or "gs-shaping"

    u = torch.zeros(n_act, dtype=_DTYPE)
    dm_u = torch.zeros((N, N), dtype=_DTYPE)

    loss_hist = np.zeros(max_iter)
    energy_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, to_numpy(u).copy()))

    for it in range(max_iter):
        phase = turb_t + dm_u
        phase_new = _gs_refine_phase(phase, focal_amp=focal_amp)
        dm_u = _project_onto_dm(phase_new - turb_t, inf_flat_T, pinv)

        with torch.no_grad():
            I_eval = far_field_intensity_metric(turb_t + dm_u)
            I_n = I_eval / (I_eval.sum() + 1e-12)
            energy_hist[it] = to_numpy(energy_in_target(I_eval, target_t))
            loss_match = torch.mean((I_n - target_n) ** 2)
            loss_hist[it] = to_numpy(
                loss_weight_match * loss_match - loss_weight_capture * energy_hist[it]
            )

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, recover_u(inf_flat_T, dm_u)))

        if it % 200 == 0 or it == max_iter - 1:
            logger.info(
                f"GS-shaping[{log_name}] {it:4d}: "
                f"loss={loss_hist[it]:.4e}, energy-in-target={energy_hist[it]:.3f}"
            )

    with torch.no_grad():
        dm_u_final = dm_u
        I_final = far_field_intensity_metric(turb_t + dm_u_final)

    u_np = recover_u(inf_flat_T, dm_u_final)
    dm_np = to_numpy(dm_u_final)
    I_np = to_numpy(I_final / (I_final.sum() + 1e-12))

    return u_np, dm_np, loss_hist, energy_hist, I_np
