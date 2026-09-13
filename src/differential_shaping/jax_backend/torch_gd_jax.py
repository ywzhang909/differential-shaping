# -*- coding: utf-8 -*-
"""
JAX / chromatix version of ``optimization.torch_gd`` — backprop Torch-GD.

Differentiates directly through the differentiable far-field forward model
(``optics_jax.far_field_intensity_metric``) with ``jax.grad`` and steers the DM
actuator commands with Adam against a Gaussian-weighted on-axis energy loss.
Returns the identical ``(u, dm_u, J_hist, SR_hist)`` contract as SPGD / H-GD.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger

from differential_shaping.params import n_act
from differential_shaping.params import max_iter as _default_max_iter

from .optics_jax import (
    compute_metrics,
    dm_surface,
    far_field_intensity_metric,
    gaussian_window_jax,
    to_jax,
    to_numpy,
)
from .jax_adam import jax_adam_init, jax_adam_step

__all__ = ["torch_gd_optimization"]


def torch_gd_optimization(
    turb_phase,
    inf_flat,
    max_iter: int = _default_max_iter,
    lr: float = 0.01,
    seed: int = 42,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimise the spot spread (maximise on-axis energy) via jax.grad + Adam."""
    turb = to_jax(turb_phase)  # (N, N)
    inf_flat_t = to_jax(inf_flat)
    inf_flat_T = inf_flat_t.T  # (N*N, n_act), matches dm = inf_flat @ u

    u = jnp.zeros(n_act, dtype=jnp.float32)
    state = jax_adam_init((n_act,))

    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, np.zeros(n_act, dtype=np.float64)))

    def _loss(u: jax.Array) -> jax.Array:
        dm_u = dm_surface(u, inf_flat_T)
        phase = turb + dm_u
        intensity = far_field_intensity_metric(phase)
        return -(intensity * gaussian_window_jax).sum()

    grad_fn = jax.grad(_loss)

    for it in range(max_iter):
        grad = grad_fn(u)
        u = jax_adam_step(state, u, grad, lr)

        phase_eval = turb + dm_surface(u, inf_flat_T)
        J, SR, _ = compute_metrics(phase_eval)
        J_hist[it] = float(J)
        SR_hist[it] = float(SR)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(u).copy()))

        if it % 100 == 0 or it == max_iter - 1:
            logger.info(
                f"Torch-GD[jax] {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}"
            )

    u_np = to_numpy(u)
    dm_np = to_numpy(dm_surface(u, inf_flat_T))
    return u_np, dm_np, J_hist, SR_hist
