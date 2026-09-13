# -*- coding: utf-8 -*-
"""
Far-field target-shaping helpers (JAX version of ``optimization.shaping``).

Defines the focal-plane intensity *targets* (square / triangle), the
energy-in-target metric, and the shared shaping loss used by every shaping
optimizer (backprop and finite-difference alike) so they all optimise an
identical objective.  The JAX ``optics`` forward model is reused.

Public API
----------
    make_square_target(half_width)       -> (N, N) float32 binary mask
    make_triangle_target(size, apex)     -> (N, N) float32 binary mask
    energy_in_target_jax(I, target)      -> scalar
    shaping_metric_jax(phase, ...)       -> (loss, energy, I_n)
"""

from __future__ import annotations

import jax.numpy as jnp

from differential_shaping.params import N

from .optics_jax import far_field_intensity_metric, to_jax

__all__ = [
    "make_square_target",
    "make_triangle_target",
    "energy_in_target_jax",
    "shaping_metric_jax",
]


def make_square_target(half_width: int = 6) -> jax.Array:
    """Binary square target centred on the focal plane (side 2*half_width+1)."""
    c = jnp.arange(N, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(c, c, indexing="ij")
    mask = (jnp.abs(xx - N // 2) <= half_width) & (jnp.abs(yy - N // 2) <= half_width)
    return mask.astype(jnp.float32)


def make_triangle_target(size: int = 11, apex: str = "up") -> jax.Array:
    """Binary isosceles-triangle target centred on the focal plane."""
    c = jnp.arange(N, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(c, c, indexing="ij")
    x0, y0 = N // 2, N // 2
    h = jnp.float32(size)
    b2 = jnp.float32(size) / 2.0
    top = jnp.float32(y0 - size / 2.0)
    bot = jnp.float32(y0 + size / 2.0)

    if apex == "up":
        margin = b2 * (yy - top) / h
    elif apex == "down":
        margin = b2 * (bot - yy) / h
    elif apex == "right":
        margin = b2 * (xx - jnp.float32(x0 - size / 2.0)) / h
    elif apex == "left":
        margin = b2 * (jnp.float32(x0 + size / 2.0) - xx) / h
    else:
        raise ValueError(f"Unknown apex: {apex!r}")

    if apex in ("up", "down"):
        inside = (yy >= top) & (yy <= bot) & (jnp.abs(xx - x0) <= margin)
    else:
        inside = (
            (xx >= jnp.float32(x0 - size / 2.0))
            & (xx <= jnp.float32(x0 + size / 2.0))
            & (jnp.abs(yy - y0) <= margin)
        )
    return inside.astype(jnp.float32)


def energy_in_target_jax(intensity: jax.Array, target: jax.Array) -> jax.Array:
    """Fraction of conserved far-field energy captured inside the target."""
    return (intensity * target).sum() / (intensity.sum() + 1e-12)


def shaping_metric_jax(
    phase: jax.Array,
    target_t: jax.Array,
    target_n: jax.Array,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Conservation-respecting shaping loss + energy-in-target + normalised I."""
    intensity = far_field_intensity_metric(phase)
    I_n = intensity / (intensity.sum() + 1e-12)
    loss_match = jnp.mean((I_n - target_n) ** 2)
    energy = (I_n * target_t).sum() / (target_t.sum() + 1e-12)
    loss = loss_weight_match * loss_match - loss_weight_capture * energy
    return loss, energy, I_n
