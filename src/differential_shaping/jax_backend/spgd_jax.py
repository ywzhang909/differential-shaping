# -*- coding: utf-8 -*-
"""
JAX / chromatix version of ``optimization.spgd`` — Stochastic Parallel Gradient
Descent with two-sided perturbation (point + shaping).

The two-sided perturbation estimator is identical to the torch SPGD:
``scalar = -alpha * (J(u+d) - J(u-d)) / (2*delta^2)``.  For point correction the
objective is the centroid-radius ``J`` (from ``compute_metrics``); for shaping
the objective is the target-shaping loss (from ``shaping_jax.shaping_metric_jax``).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from loguru import logger

from differential_shaping.params import (
    alpha_spgd,
    delta_amp,
    n_act,
    seed_spgd,
    max_iter as _default_max_iter,
)
from differential_shaping.params import N

from .optics_jax import compute_metrics, dm_surface, to_jax, to_numpy

__all__ = ["spgd_optimization", "spgd_shaping_optimization"]


def _spgd_loop(
    turb,
    inf_flat,
    delta_patterns: jax.Array,
    dm_deltas: jax.Array,
    alpha: float,
    max_iter: int,
    objective,
    snapshot_indices=None,
    snapshot_store=None,
    metric=None,
    metric_names=("J", "SR"),
) -> tuple:
    """Shared inner loop for point + shaping SPGD.

    ``metric`` is a callable ``phase -> (a, b)`` whose two scalars are recorded
    per-iteration and returned as ``(hist_a, hist_b)``.  Point correction passes
    a J/SR metric; shaping passes a loss/energy metric so the returned contract
    matches torch ``(u, dm, loss_hist, energy_hist)``.
    """
    # ``inf_flat`` is flat (n_act, N*N); the transposed basis is (N*N, n_act) so
    # that ``dm_surface = inf_flat_T @ u`` matches the torch convention
    # (``dm = inf_flat @ u``).
    inf_flat_T = inf_flat.T  # (N*N, n_act)
    u = jnp.zeros(n_act, dtype=jnp.float32)
    dm_u = jnp.zeros((N, N), dtype=jnp.float32)

    if metric is None:

        def metric(phase):
            J, SR, _ = compute_metrics(phase)
            return J, SR

    a_name, b_name = metric_names
    hist_a = np.zeros(max_iter)
    hist_b = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, np.zeros(n_act, dtype=np.float64)))

    for it in range(max_iter):
        delta = jnp.max(jnp.abs(delta_patterns[it]))
        J_p = objective(turb + dm_u + dm_deltas[it])
        J_m = objective(turb + dm_u - dm_deltas[it])
        scalar = -alpha * (J_p - J_m) / (2 * delta**2 + 1e-30)
        u = u + scalar * delta_patterns[it]
        dm_u = dm_u + scalar * dm_deltas[it]

        a, b = metric(turb + dm_u)
        hist_a[it] = float(a)
        hist_b[it] = float(b)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            from .gs_jax import recover_u_jax

            snapshot_store.append(
                (it + 1, to_numpy(recover_u_jax(inf_flat_T, dm_u)).copy())
            )

        if it % 500 == 0 or it == max_iter - 1:
            logger.info(
                f"SPGD[jax] {it:4d}: {a_name}={hist_a[it]:.6f}, {b_name}={hist_b[it]:.6f}"
            )

    from .gs_jax import recover_u_jax

    u_np = to_numpy(recover_u_jax(inf_flat_T, dm_u))
    dm_np = to_numpy(dm_u)
    return u_np, dm_np, hist_a, hist_b


def spgd_optimization(
    turb_phase,
    inf_flat,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
    max_iter: int = _default_max_iter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SPGD point correction (J = centroid radius)."""
    turb = to_jax(turb_phase)
    inf_flat_t = to_jax(inf_flat)

    rng = np.random.RandomState(seed_spgd)
    patterns = 2 * rng.randint(0, 2, (max_iter, n_act)).astype(np.float32) - 1
    delta_patterns = jnp.asarray(patterns) * delta_amp
    dm_deltas = (delta_patterns @ inf_flat_t).reshape(max_iter, N, N)

    def objective(phase):
        J, _, _ = compute_metrics(phase)
        return J

    return _spgd_loop(
        turb,
        inf_flat_t,
        delta_patterns,
        dm_deltas,
        alpha_spgd,
        max_iter,
        objective,
        snapshot_indices,
        snapshot_store,
    )


def spgd_shaping_optimization(
    turb_phase,
    inf_flat,
    target,
    max_iter: int = _default_max_iter,
    alpha: float = alpha_spgd,
    seed: int = seed_spgd,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
    label: str | None = None,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SPGD shaping (target-loss objective)."""
    from .shaping_jax import shaping_metric_jax

    turb = to_jax(turb_phase)
    inf_flat_t = to_jax(inf_flat)
    target_t = jnp.clip(to_jax(target), min=0.0)
    target_n = target_t / (target_t.sum() + 1e-12)

    rng = np.random.RandomState(seed_spgd)
    patterns = 2 * rng.randint(0, 2, (max_iter, n_act)).astype(np.float32) - 1
    delta_patterns = jnp.asarray(patterns) * delta_amp
    dm_deltas = (delta_patterns @ inf_flat_t).reshape(max_iter, N, N)

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
        delta_patterns,
        dm_deltas,
        alpha,
        max_iter,
        objective,
        snapshot_indices,
        snapshot_store,
        metric=metric,
        metric_names=("loss", "energy"),
    )

    from .optics_jax import far_field_intensity_metric

    I_final = far_field_intensity_metric(turb + dm)
    I_np = to_numpy(I_final / (I_final.sum() + 1e-12))
    return u, dm, loss_hist, energy_hist, I_np
