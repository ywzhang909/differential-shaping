# -*- coding: utf-8 -*-
"""
Hadamard Gradient Descent (H-GD) optimization.

Uses Hadamard-encoded perturbation patterns for structured exploration
of the DM actuator space.

The inner loop operates entirely on float32 PyTorch tensors; only
snapshot copies and the returned arrays are numpy float64.
"""

import numpy as np
import torch
from loguru import logger
from scipy.linalg import hadamard

from ..params import delta_amp, alpha_hgd, max_iter, n_act, N
from ..simulation.optics import compute_metrics, to_numpy
from .spgd import update_by_two_sided_perturbation

__all__ = ["hgd_optimization"]

_DT = torch.float32  # working dtype for all torch arrays


def hgd_optimization(
    turb_phase,
    inf_flat,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run H-GD with two-sided perturbation for max_iter steps.

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

    Returns
    -------
    u_np, dm_np, J_hist_np, SR_hist_np : np.ndarray (float64)
        ``u_np`` shape ``(n_act,)``, ``dm_np`` shape ``(N, N)``,
        ``J_hist_np`` and ``SR_hist_np`` shape ``(max_iter,)``.
    """
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

    # --- Hadamard perturbation patterns (algorithmic, scipy) -------------
    H = hadamard(n_act, dtype=np.float64)
    # Skip the all-one column; it is mainly a piston-like common mode after
    # influence-function projection.
    patterns = H[:, 1:].T.copy()
    patterns_t = torch.from_numpy(patterns.astype(np.float32))

    delta_patterns = patterns_t * delta_amp
    # (n_act-1, n_act) @ (n_act, N*N) -> (n_act-1, N*N) -> (n_act-1, N, N)
    dm_deltas = (delta_patterns @ inf_flat_t).reshape(n_act - 1, N, N)

    u = torch.zeros(n_act, dtype=_DT)
    dm_u = torch.zeros((N, N), dtype=_DT)

    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, to_numpy(u).copy()))

    for it in range(max_iter):
        k = it % (n_act - 1)
        u, dm_u = update_by_two_sided_perturbation(
            turb_phase, u, dm_u, delta_patterns[k], dm_deltas[k], alpha_hgd
        )
        J, SR, _ = compute_metrics(turb_phase + dm_u)
        J_hist[it] = float(J)
        SR_hist[it] = float(SR)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(u).copy()))

        if it % 500 == 0:
            logger.info(
                f"H-GD {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}"
            )

    return to_numpy(u), to_numpy(dm_u), J_hist, SR_hist
