# -*- coding: utf-8 -*-
"""
JAX version of ``simulation.slm``.

Pixelated phase SLM simulation with fill-factor dead-zone modelling, on JAX
arrays.  Mirrors the torch module's algorithm one-to-one: a fine (4x sub-pixel)
grid is used to resolve the periodic pixel dead-zones so that 60% / 80% / 100%
fill factors are smoothly distinguishable in the far field.

The only change from the torch version is the backend (JAX + ``fx.fft``); the
physics, the ``1/_SLM_UP**4`` intensity scaling (to align with the 128-grid
ideal ``I0_peak``), and the 8-bit phase quantization are identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chromatix.functional as fx

from differential_shaping.params import (
    N,
    slm_pixel_phase_steps,
    slm_pixel_pitch_px,
)

from .turbulence_jax import remove_piston

__all__ = [
    "quantize_slm_phase",
    "slm_fill_mask",
    "slm_field",
    "slm_far_field_intensity",
    "slm_far_field_intensity_metric",
]

_SLAM_UP = 4


def _fine_grid(M: int) -> tuple[jax.Array, jax.Array]:
    xq = (jnp.arange(M, dtype=jnp.float32) - M / 2.0) / _SLAM_UP
    XX, YY = jnp.meshgrid(xq, xq, indexing="xy")
    return XX, YY


def _fine_slm_mask(
    fill_factor: float, pixel_pitch_px: float, M: int
) -> jax.Array:
    half_pitch = pixel_pitch_px / 2.0
    a_half = half_pitch * float(fill_factor) ** 0.5

    XX, YY = _fine_grid(M)
    px_center = jnp.floor(XX / pixel_pitch_px + 0.5) * pixel_pitch_px
    py_center = jnp.floor(YY / pixel_pitch_px + 0.5) * pixel_pitch_px

    active = (
        (jnp.abs(XX - px_center) <= a_half)
        & (jnp.abs(YY - py_center) <= a_half)
    ).astype(jnp.float32)
    return active


def _fine_pupil(M: int) -> jax.Array:
    XX, YY = _fine_grid(M)
    rr = jnp.sqrt(XX**2 + YY**2)
    return (rr <= N / 2.0).astype(jnp.float32)


def _upsample_phase(phase: jax.Array, M: int) -> jax.Array:
    """Nearest-neighbour upsample (N, N) phase to (M, M) fine grid."""
    K = _SLAM_UP
    # jnp.repeat repeats each element K times along a given axis.
    return jnp.repeat(jnp.repeat(phase, K, axis=0), K, axis=1)


def quantize_slm_phase(
    phase: jax.Array, levels: int = slm_pixel_phase_steps
) -> jax.Array:
    """Quantize continuous phase to the SLM's discrete grey steps."""
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    phase_wrapped = jnp.remainder(phase, 2 * jnp.pi)
    idx = jnp.round(phase_wrapped / (2 * jnp.pi) * (levels - 1))
    idx = jnp.clip(idx, 0, levels - 1)
    return idx / (levels - 1) * (2 * jnp.pi)


def slm_field(
    phase: jax.Array,
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    quantize: bool = True,
) -> jax.Array:
    """Build the exit-pupil complex field after a pixelated SLM (fine grid)."""
    if not 0.0 < fill_factor <= 1.0:
        raise ValueError(f"fill_factor must be in (0, 1], got {fill_factor}")
    if pixel_pitch_px < 1.0:
        raise ValueError(f"pixel_pitch_px must be >= 1, got {pixel_pitch_px}")

    M = N * _SLAM_UP
    phase_clean = remove_piston(phase)
    if quantize:
        phase_clean = quantize_slm_phase(phase_clean)

    mask_fine = _fine_slm_mask(fill_factor, pixel_pitch_px, M)
    pupil_fine = _fine_pupil(M)
    phase_up = _upsample_phase(phase_clean, M)

    return mask_fine * pupil_fine * jnp.exp(1j * phase_up)


def slm_far_field_intensity(
    phase: jax.Array,
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    quantize: bool = True,
) -> jax.Array:
    """Far-field intensity after a pixelated SLM (with fill factor), fine grid."""
    E = slm_field(phase, fill_factor, pixel_pitch_px=pixel_pitch_px, quantize=quantize)
    return jnp.abs(fx.fft(E, axes=(0, 1), shift=True)) ** 2 / (_SLAM_UP**4)


def slm_far_field_intensity_metric(
    phase: jax.Array,
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    quantize: bool = True,
) -> jax.Array:
    """Alias of :func:`slm_far_field_intensity`."""
    return slm_far_field_intensity(
        phase, fill_factor, pixel_pitch_px=pixel_pitch_px, quantize=quantize
    )


def slm_fill_mask(
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    grid_N: int | None = None,
) -> jax.Array:
    """Pixelated SLM fill-factor amplitude mask (for visualization)."""
    if not 0.0 < fill_factor <= 1.0:
        raise ValueError(f"fill_factor must be in (0, 1], got {fill_factor}")
    if pixel_pitch_px < 1.0:
        raise ValueError(f"pixel_pitch_px must be >= 1, got {pixel_pitch_px}")

    M = N * _SLAM_UP
    pupil_fine = _fine_pupil(M)
    mask_fine = _fine_slm_mask(fill_factor, pixel_pitch_px, M) * pupil_fine
    if grid_N is None:
        return mask_fine
    step = M // grid_N
    mask = mask_fine[::step, ::step]
    if mask.shape[0] != grid_N:
        mask = mask[:grid_N, :grid_N]
    return mask
