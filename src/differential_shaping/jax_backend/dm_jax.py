# -*- coding: utf-8 -*-
"""
JAX version of ``simulation.dm``.

Deformable-mirror actuator grid and Gaussian influence functions on JAX arrays.

Mirrors the torch ``dm.py`` exactly: 16x16 = 256 actuators, Gaussian influence
functions (``sigma_inf = act_spacing * 0.65``), each piston-centred inside the
pupil and normalised to unit max-abs.  Returns ``inf_flat`` of shape
``(n_act, N, N)`` (same layout as the torch module) so the optimisers can reuse
the identical DM-surface / least-squares-projection logic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from differential_shaping.params import (
    D,
    N,
    act_spacing,
    n_act,
    n_act_x,
    n_act_y,
    sigma_inf,
)

from .pupil_jax import XX_jax, YY_jax, pupil_float_jax, pupil_mask_jax

__all__ = ["generate_actuator_grid", "generate_influence_functions", "inf_flat_jax"]


def generate_actuator_grid() -> tuple[jax.Array, jax.Array]:
    """Return flattened actuator (X, Y) centres in metres."""
    margin = act_spacing / 2
    xs = jnp.linspace(-D / 2 + margin, D / 2 - margin, n_act_x)
    ys = jnp.linspace(-D / 2 + margin, D / 2 - margin, n_act_y)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    return X.ravel(), Y.ravel()


def generate_influence_functions() -> jax.Array:
    """Build the (n_act, N, N) Gaussian influence-function matrix.

    For each actuator (x0, y0):
        z = exp(-(r^2)/(2 sigma_inf^2)) * pupil
        z -= mean(z inside pupil)
        z outside pupil = 0
        z /= max|z|
    """
    act_x, act_y = generate_actuator_grid()
    n_pup = jnp.sum(pupil_mask_jax) + 1e-30

    def _one(i):
        x0 = act_x[i]
        y0 = act_y[i]
        r2 = (XX_jax - x0) ** 2 + (YY_jax - y0) ** 2
        z = jnp.exp(-r2 / (2 * sigma_inf**2)) * pupil_float_jax
        z = z - jnp.sum(z) / n_pup
        z = jnp.where(pupil_mask_jax, z, 0.0)
        z = z / (jnp.max(jnp.abs(z)) + 1e-15)
        return z

    inf = jax.vmap(_one)(jnp.arange(n_act))
    return inf


# Precomputed influence-function matrix (module-level, matches the torch
# module which also computes these once at import).
inf_flat_jax = generate_influence_functions()
