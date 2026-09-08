"""
PyTorch backpropagation-driven wavefront-sensorless optimiser (Torch-GD).

Unlike the finite-difference SPGD / H-GD methods, this optimiser differentiates
directly through the (differentiable) far-field forward model provided by
``simulation.optics`` and steers the DM actuator commands ``u`` with an
Adam optimiser against a Gaussian-weighted on-axis energy loss.

The public API mirrors ``spgd_optimization`` / ``hgd_optimization`` so that the
pipeline orchestrator can treat all three optimisers uniformly: it returns
``(u, dm_u, J_hist, SR_hist)`` where ``J_hist`` / ``SR_hist`` are the *numpy
metric* histories (identical definition to the numpy optimisers) so convergence
curves are directly comparable.
"""

import numpy as np
import torch
from loguru import logger

from differential_shaping.params import n_act
from differential_shaping.params import max_iter as _default_max_iter
from differential_shaping.simulation.optics import (
    compute_metrics,
    dm_surface,
    far_field_intensity_metric,
    gaussian_window,
    to_numpy,
    to_torch,
)
from differential_shaping.simulation.pupil import pupil_float_t

__all__ = ["torch_gd_optimization"]


def torch_gd_optimization(
    turb_phase: np.ndarray | torch.Tensor,
    inf_flat: np.ndarray | torch.Tensor,
    max_iter: int = _default_max_iter,
    lr: float = 0.01,
    seed: int = 42,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimise the spot spread (maximise on-axis energy) via autograd.

    Args:
        turb_phase: (N, N) float64 turbulence phase in radians.
        inf_flat:   (n_act, N*N) flattened DM influence functions.
        max_iter:   number of Adam steps (defaults to params.max_iter; typically
                    run with fewer steps than SPGD/H-GD since each step is one
                    full forward+backward).
        lr:         Adam learning rate.
        seed:       RNG seed for reproducible Adam trajectories.
        snapshot_indices: iteration indices at which to record ``(iter, u)``
                    snapshots into ``snapshot_store`` (for step visualisation).
        snapshot_store:   mutable list to append ``(iter, u)`` snapshots to.

    Returns:
        (u, dm_u, J_hist, SR_hist):
            u       (n_act,)  float64 final DM commands.
            dm_u    (N, N)    float64 final DM surface.
            J_hist  (max_iter,) float64 centroid-radius history (numpy metric).
            SR_hist (max_iter,) float64 Strehl-ratio history (numpy metric).
    """
    torch.manual_seed(seed)

    turb_t = to_torch(turb_phase)  # (N, N) constant graph input
    inf_flat_t = to_torch(inf_flat)  # (n_act, N*N)
    inf_flat_T = inf_flat_t.t().contiguous()  # (N*N, n_act)

    u = torch.zeros(n_act, requires_grad=True, device=pupil_float_t.device)

    optimizer = torch.optim.Adam([u], lr=lr)

    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, to_numpy(u).copy()))

    for it in range(max_iter):
        optimizer.zero_grad()
        dm_u = dm_surface(u, inf_flat_T)  # (N, N)
        phase = turb_t + dm_u
        intensity = far_field_intensity_metric(phase)  # differentiable forward
        # Maximise on-axis energy: dense smooth gradients over the Airy core.
        loss = -(intensity * gaussian_window).sum()
        loss.backward()
        optimizer.step()

        # Metric logging under no_grad (not part of the training graph).
        with torch.no_grad():
            phase_eval = turb_t + dm_surface(u, inf_flat_T)
            J, SR, _ = compute_metrics(phase_eval)
            J_hist[it] = to_numpy(J)
            SR_hist[it] = to_numpy(SR)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(u).copy()))

        if it % 100 == 0 or it == max_iter - 1:
            logger.info(
                f"Torch-GD {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}"
            )

    with torch.no_grad():
        dm_u_final = dm_surface(u, inf_flat_T)

    u_np = to_numpy(u)
    dm_np = to_numpy(dm_u_final)

    return u_np, dm_np, J_hist, SR_hist
