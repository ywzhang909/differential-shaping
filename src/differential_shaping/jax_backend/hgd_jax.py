# -*- coding: utf-8 -*-
"""
JAX / chromatix version of ``optimization.hgd`` — Hadamard Gradient Descent.

Uses orthogonal Hadamard-row perturbation patterns (cycled) instead of random
SPGD patterns.  Same two-sided estimator as SPGD; the perturbation structure is
the only difference.  Supports both point correction and target shaping.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import hadamard
import jax.numpy as jnp
from loguru import logger

from differential_shaping.params import (
    alpha_hgd,
    delta_amp,
    n_act,
    max_iter as _default_max_iter,
)
from differential_shaping.params import N

from .optics_jax import compute_metrics, dm_surface, to_jax, to_numpy

__all__ = ["hgd_optimization", "hgd_shaping_optimization"]


def _hadamard_patterns(delta_amp_: float) -> tuple[jax.Array, jax.Array]:
    """Return (delta_patterns, dm_deltas) cycling (n_act-1) Hadamard rows."""
    H = hadamard(n_act, dtype=np.float64)
    patterns = H[:, 1:].T.copy()  # (n_act-1, n_act), skip all-one column
    return jnp.asarray(patterns, dtype=jnp.float32) * delta_amp_


def hgd_optimization(
    turb_phase,
    inf_flat,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
    max_iter: int = _default_max_iter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """H-GD point correction (J = centroid radius)."""
    from .spgd_jax import _spgd_loop

    turb = to_jax(turb_phase)
    inf_flat_t = to_jax(inf_flat)

    delta_patterns = _hadamard_patterns(delta_amp)  # (n_act-1, n_act)
    dm_deltas_full = (delta_patterns @ inf_flat_t).reshape(n_act - 1, N, N)

    if max_iter <= n_act - 1:
        delta_used = delta_patterns[:max_iter]
        dm_used = dm_deltas_full[:max_iter]
    else:
        reps = (max_iter + n_act - 2) // (n_act - 1)
        delta_used = jnp.repeat(delta_patterns, reps, axis=0)[:max_iter]
        dm_used = jnp.repeat(dm_deltas_full, reps, axis=0)[:max_iter]

    def objective(phase):
        J, _, _ = compute_metrics(phase)
        return J

    return _spgd_loop(
        turb,
        inf_flat_t,
        delta_used,
        dm_used,
        alpha_hgd,
        max_iter,
        objective,
        snapshot_indices,
        snapshot_store,
    )


def hgd_shaping_optimization(
    turb_phase,
    inf_flat,
    target,
    max_iter: int = _default_max_iter,
    alpha: float = alpha_hgd,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
    label: str | None = None,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """H-GD shaping (target-loss objective)."""
    from .spgd_jax import _spgd_loop
    from .shaping_jax import shaping_metric_jax
    from .optics_jax import far_field_intensity_metric

    turb = to_jax(turb_phase)
    inf_flat_t = to_jax(inf_flat)
    target_t = jnp.clip(to_jax(target), min=0.0)
    target_n = target_t / (target_t.sum() + 1e-12)

    delta_patterns = _hadamard_patterns(delta_amp)
    dm_deltas_full = (delta_patterns @ inf_flat_t).reshape(n_act - 1, N, N)

    if max_iter <= n_act - 1:
        delta_used = delta_patterns[:max_iter]
        dm_used = dm_deltas_full[:max_iter]
    else:
        reps = (max_iter + n_act - 2) // (n_act - 1)
        delta_used = jnp.repeat(delta_patterns, reps, axis=0)[:max_iter]
        dm_used = jnp.repeat(dm_deltas_full, reps, axis=0)[:max_iter]

    def objective(phase):
        loss, _, _ = shaping_metric_jax(
            phase, target_t, target_n, loss_weight_match, loss_weight_capture
        )
        return loss

    def metric(phase):
        loss, energy, _ = shaping_metric_jax(
            phase, target_t, target_n, loss_weight_match, loss_weight_capture
        )
        return loss, energy

    u, dm, loss_hist, energy_hist = _spgd_loop(
        turb,
        inf_flat_t,
        delta_used,
        dm_used,
        alpha,
        max_iter,
        objective,
        snapshot_indices,
        snapshot_store,
        metric=metric,
        metric_names=("loss", "energy"),
    )

    I_final = far_field_intensity_metric(turb + dm)
    I_np = to_numpy(I_final / (I_final.sum() + 1e-12))
    return u, dm, loss_hist, energy_hist, I_np
