"""
Stochastic Parallel Gradient Descent (SPGD) optimization.

Provides the two-sided perturbation estimator and the main SPGD loop
used by the run entry point.  The forward model (``compute_metrics``) is
evaluated in PyTorch on CPU (float32) throughout the hot loop for speed;
only snapshot copies and the final return values are materialised as
numpy float64 arrays.
"""

import numpy as np
import torch
from loguru import logger

from differential_shaping import params
from differential_shaping.params import (
    N,
    alpha_spgd,
    delta_amp,
    n_act,
    seed_spgd,
)
from differential_shaping.simulation.optics import compute_metrics, to_numpy

__all__ = ["spgd_optimization", "update_by_two_sided_perturbation"]

_DT = torch.float32  # working dtype for all torch arrays


def update_by_two_sided_perturbation(
    turb_phase: torch.Tensor,
    u: torch.Tensor,
    dm_u: torch.Tensor,
    delta_u: torch.Tensor,
    dm_delta: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize J with a two-sided perturbation estimate.

    Because dm_delta = DM(delta_u), the control update is scalar * delta_u,
    so the DM surface update is the same scalar * dm_delta.

    All inputs and outputs are float32 torch tensors.
    """
    delta = torch.max(torch.abs(delta_u))
    J_p, _, _ = compute_metrics(turb_phase + dm_u + dm_delta)
    J_m, _, _ = compute_metrics(turb_phase + dm_u - dm_delta)
    scalar = -alpha * (J_p - J_m) / (2 * delta**2 + 1e-30)
    u = u + scalar * delta_u
    dm_u = dm_u + scalar * dm_delta
    return u, dm_u


def spgd_optimization(
    turb_phase,
    inf_flat,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
    max_iter: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run SPGD with two-sided perturbation for max_iter steps.

    The inner loop operates entirely on float32 PyTorch tensors; only
    snapshot copies and the returned arrays are numpy float64.

    Parameters
    ----------
    turb_phase : torch.Tensor or np.ndarray
        Turbulence phase screen, shape ``(N, N)``.
    inf_flat : torch.Tensor or np.ndarray
        Flattened DM influence functions, shape ``(n_act, N*N)``.
    snapshot_indices : list[int], optional
        Iteration numbers at which to store a snapshot of the control vector.
    snapshot_store : list, optional
        Mutable list; ``(iteration, np.ndarray)`` tuples are appended.
    max_iter : int, optional
        Number of iterations.  Defaults to ``params.max_iter``.

    Returns
    -------
    u_np, dm_np, J_hist_np, SR_hist_np : np.ndarray (float64)
        ``u_np`` shape ``(n_act,)``, ``dm_np`` shape ``(N, N)``,
        ``J_hist_np`` and ``SR_hist_np`` shape ``(max_iter,)``.
    """
    if max_iter is None:
        max_iter = params.max_iter

    # --- normalise inputs to float32 torch tensors -----------------------
    if isinstance(turb_phase, np.ndarray):
        turb_phase = torch.from_numpy(np.asarray(turb_phase, dtype=np.float64)).to(
            dtype=_DT
        )
    turb_phase = turb_phase.to(dtype=_DT)

    if isinstance(inf_flat, np.ndarray):
        inf_flat_t = torch.from_numpy(np.asarray(inf_flat, dtype=np.float64)).to(
            dtype=_DT
        )
    else:
        inf_flat_t = inf_flat.to(dtype=_DT)

    # --- precompute perturbation DM surfaces (pure torch) ----------------
    gen = torch.Generator()
    gen.manual_seed(seed_spgd)
    patterns = (
        2 * torch.randint(0, 2, (max_iter, n_act), generator=gen).to(dtype=_DT) - 1
    )
    delta_patterns = patterns * delta_amp

    # (max_iter, n_act) @ (n_act, N*N) -> (max_iter, N*N) -> (max_iter, N, N)
    dm_deltas = (delta_patterns @ inf_flat_t).reshape(max_iter, N, N)

    u = torch.zeros(n_act, dtype=_DT)
    dm_u = torch.zeros((N, N), dtype=_DT)

    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, to_numpy(u).copy()))

    for it in range(max_iter):
        u, dm_u = update_by_two_sided_perturbation(
            turb_phase, u, dm_u, delta_patterns[it], dm_deltas[it], alpha_spgd
        )
        J, SR, _ = compute_metrics(turb_phase + dm_u)
        J_hist[it] = float(J)
        SR_hist[it] = float(SR)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(u).copy()))

        if it % 500 == 0:
            logger.info(f"SPGD {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}")

    return to_numpy(u), to_numpy(dm_u), J_hist, SR_hist
