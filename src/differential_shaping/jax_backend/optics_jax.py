# -*- coding: utf-8 -*-
"""
JAX / chromatix version of ``simulation.optics`` — the differentiable
simulation core.

The single most important operation in this codebase is the coherent
Fraunhofer far-field transform:

    E = pupil * exp(i * piston_rm(phase))     # pupil-plane complex field
    I = |fftshift(fft2(E))|^2                 # focal-plane intensity

which maps onto ``chromatix.functional.fft`` (``fx.fft(E, shift=True)``)
verified numerically identical to ``torch.fft.fftshift(fft2(E))`` to float32
precision.  All the public functions here mirror the torch ``optics`` module
so the optimisers / visualisation can be swapped backends with no other
change.

Public API
----------
    far_field_intensity_metric(phase)   -> (N, N) focal intensity
    far_field_field(phase)              -> (N, N) complex focal field g
    compute_metrics(phase)              -> (J, SR, intensity)
    dm_surface(u, inf_flat_T)           -> (N, N) DM surface
    to_numpy / to_jax                   -> conversion helpers
"""

from __future__ import annotations

import jax.numpy as jnp
import chromatix.functional as fx

from differential_shaping.params import N

from .pupil_jax import I0_peak_jax, pupil_float_jax, xx_metric_jax, yy_metric_jax
from .turbulence_jax import remove_piston

__all__ = [
    "far_field_intensity_metric",
    "far_field_field",
    "compute_metrics",
    "dm_surface",
    "gaussian_window_jax",
    "to_numpy",
    "to_jax",
]

# Gaussian steering window for the Torch-GD on-axis-energy loss (sigma = 3 px).
_SIGMA_G = 3.0
_coord_g = jnp.arange(N, dtype=jnp.float32)
_YY_g, _XX_g = jnp.meshgrid(_coord_g, _coord_g, indexing="ij")
gaussian_window_jax = jnp.exp(
    -((_XX_g - N // 2) ** 2 + (_YY_g - N // 2) ** 2) / (2 * _SIGMA_G**2)
)


def far_field_field(phase: jax.Array) -> jax.Array:
    """Complex focal-plane field ``g = fftshift(fft2(pupil * exp(i*phi)))``.

    Uses ``jnp.fft.fftshift(jnp.fft.fft2(E))`` directly so the result is
    exactly the torch ``optics`` reference (``gs._gs_refine_phase`` needs this
    complex field to extract the focal phase).  ``chromatix.functional.fft``
    with ``shift=True`` would be ``fftshift(fft2(ifftshift(E)))`` — the
    chromatix "centred-input" convention — which differs from the plain
    ``fftshift(fft2(E))`` used by the torch reference and the index-based
    ``(0,0)`` pupil grid by a global phase of π, breaking the GS amplitude
    projection.
    """
    E = pupil_float_jax * jnp.exp(1j * remove_piston(phase))
    return jnp.fft.fftshift(jnp.fft.fft2(E, axes=(0, 1)), axes=(0, 1))


def far_field_intensity_metric(phase: jax.Array) -> jax.Array:
    """``|fftshift(fft2(pupil * exp(i piston_rm(phase))))|^2`` (JAX/chromatix)."""
    g = far_field_field(phase)
    return jnp.abs(g) ** 2


def compute_metrics(
    phase: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (J = centroid mean radius in pixels, SR, intensity) as JAX arrays."""
    intensity = far_field_intensity_metric(phase)
    total = jnp.sum(intensity) + 1e-30
    cx = jnp.sum(xx_metric_jax * intensity) / total
    cy = jnp.sum(yy_metric_jax * intensity) / total
    r_pix = jnp.sqrt((xx_metric_jax - cx) ** 2 + (yy_metric_jax - cy) ** 2)
    J_mr = jnp.sum(r_pix * intensity) / total
    SR = jnp.max(intensity) / (I0_peak_jax + 1e-30)
    return J_mr, SR, intensity


def dm_surface(u: jax.Array, inf_flat_T: jax.Array) -> jax.Array:
    """DM surface = inf_flat^T @ u, reshaped to (N, N)."""
    return (inf_flat_T @ u).reshape(N, N)


def to_numpy(x) -> "np.ndarray":  # type: ignore[name-defined]
    """Convert a JAX array to a float64 numpy array (for visualisation / saving)."""
    import numpy as np

    a = jnp.asarray(x)
    if jnp.iscomplexobj(a):
        a = jnp.abs(a)
    return np.asarray(a.astype("float64"))


def to_jax(a) -> jax.Array:
    """Convert a numpy array to a float32 JAX array."""
    import numpy as np

    return jnp.asarray(np.asarray(a, dtype="float64"), dtype=jnp.float32)
