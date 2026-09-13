# -*- coding: utf-8 -*-
"""
JAX version of ``simulation.pupil``.

Pupil grid, circular mask, and the diffraction-limited reference field
``I0`` — all as JAX arrays (float32 / complex64) so they compose with the
chromatix functional layer and ``jax.grad``.

Mirrors the torch module's constants one-to-one:
    ``pupil_mask_jax``   : (N, N) bool circular aperture
    ``pupil_float_jax``  : (N, N) float32 aperture (0/1)
    ``I0_jax``           : (N, N) complex64  fftshift(fft2(pupil))
    ``I0_int_jax``       : (N, N) float32    |fftshift(fft2(pupil))|^2
    ``I0_peak_jax``      : scalar float32    I0_int.max()
    ``xx_metric_jax`` / ``yy_metric_jax`` : (N, N) integer pixel coords
    ``XX_jax`` / ``YY_jax`` : (N, N) float32 pupil-plane coords (m)
"""

from __future__ import annotations

import jax.numpy as jnp
import chromatix.functional as fx

from differential_shaping.params import D, N, pixel_size

__all__ = [
    "XX_jax",
    "YY_jax",
    "grid_1d_jax",
    "pupil_mask_jax",
    "pupil_float_jax",
    "I0_jax",
    "I0_int_jax",
    "I0_peak_jax",
    "xx_metric_jax",
    "yy_metric_jax",
]

# Pupil-plane coordinate grid in metres (centred on 0), matching
# ``pupil.grid_1d_t = (arange(N) - N/2) * pixel_size``.
grid_1d_jax = (jnp.arange(N) - jnp.float32(N / 2)) * jnp.float32(pixel_size)
XX_jax, YY_jax = jnp.meshgrid(grid_1d_jax, grid_1d_jax, indexing="xy")

# Circular aperture: r <= D/2.  Mirrors ``pupil.pupil_mask``.
pupil_mask_jax = (XX_jax**2 + YY_jax**2) <= jnp.float32((D / 2) ** 2)
pupil_float_jax = pupil_mask_jax.astype(jnp.float32)

# Integer pixel-index grid (ij indexing), used by the centroid metric.
yy_metric_jax, xx_metric_jax = jnp.meshgrid(
    jnp.arange(N), jnp.arange(N), indexing="ij"
)
xx_metric_jax = xx_metric_jax.astype(jnp.float32)
yy_metric_jax = yy_metric_jax.astype(jnp.float32)

# Diffraction-limited reference field: fftshift(fft2(pupil)).  The complex
# amplitude (used by GS as the ideal Airy amplitude) and its intensity.
# ``fx.fft(x, shift=True)`` == ``fftshift(fft2(ifftshift(x)))``; for a real
# non-centred input (pupil) this equals ``fftshift(fft2(x))`` to float32.
I0_jax = fx.fft(pupil_float_jax, axes=(0, 1), shift=True)
I0_int_jax = jnp.abs(I0_jax) ** 2
I0_peak_jax = jnp.max(I0_int_jax)
