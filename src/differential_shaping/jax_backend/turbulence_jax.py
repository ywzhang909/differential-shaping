# -*- coding: utf-8 -*-
"""
JAX version of ``simulation.turbulence``.

Kolmogorov-like phase-screen synthesis on JAX arrays.  The PSD form
(``0.023 * r0^(-5/3) * fr^(-11/3)``), the ``* (N**2)`` ifft scaling, and the
algorithm are identical to the torch version; only the backend changes.

Determinism note: ``jax.random`` uses a threefry PRNG, so a given seed yields
a *different* realisation than the torch ``Generator`` stream, but it is fully
deterministic (bitwise-identical across runs for the same seed) — the same
caveat the torch module documents.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from differential_shaping.params import N, pixel_size

from .pupil_jax import pupil_mask_jax

__all__ = ["generate_turbulence_phase", "generate_phase_screen", "remove_piston"]


def remove_piston(phase: jax.Array) -> jax.Array:
    """Remove piston inside the pupil and set the outside-pupil area to 0."""
    out = jnp.where(pupil_mask_jax, phase, jnp.array(0.0))
    piston = jnp.sum(out) / (jnp.sum(pupil_mask_jax) + 1e-30)
    return jnp.where(pupil_mask_jax, out - piston, jnp.array(0.0))


def generate_phase_screen(
    N: int,
    pixel_size: float,
    r0: float,
    seed: int | None = None,
) -> jax.Array:
    """Synthesise a raw Kolmogorov-like phase screen at arbitrary size ``(N, N)``.

    JAX port of the torch ``generate_phase_screen`` — same PSD form and the
    ``* (N**2)`` inverse-FFT scaling.  Piston is removed over the whole screen.

    Parameters
    ----------
    N : int
        Grid size (pixels per side).
    pixel_size : float
        Pupil-plane sampling pitch (m).
    r0 : float
        Fried parameter (m).
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    jax.Array
        Float32 array of shape ``(N, N)``.
    """
    if seed is None:
        seed = 0
    key = random.PRNGKey(seed)

    D_local = N * pixel_size
    df = 1.0 / D_local

    # Frequency grid (matches torch.fft.fftfreq ordering).
    fx_ = jnp.fft.fftfreq(N, d=pixel_size)
    FX, FY = jnp.meshgrid(fx_, fx_, indexing="ij")
    fr = jnp.sqrt(FX**2 + FY**2)

    mask = fr > 0
    PSD = jnp.where(
        mask,
        0.023 * r0 ** (-5.0 / 3.0) * jnp.where(mask, fr, 1.0) ** (-11.0 / 3.0),
        0.0,
    )

    key, k1, k2 = random.split(key, 3)
    noise = random.normal(k1, (N, N), dtype=jnp.float32) + 1j * random.normal(
        k2, (N, N), dtype=jnp.float32
    )

    phase_fft = jnp.sqrt(PSD) * noise * df
    phase = jnp.real(jnp.fft.ifft2(phase_fft)) * (N**2)

    return phase - jnp.mean(phase)


def generate_turbulence_phase(
    N: int,
    pixel_size: float,
    r0: float,
    target_rms: float = 1.0,
    seed: int | None = None,
) -> jax.Array:
    """Generate a Kolmogorov-like phase screen scaled to ``target_rms`` inside the pupil.

    JAX port of the torch ``generate_turbulence_phase``.

    Parameters
    ----------
    N : int
        Grid size (pixels per side).
    pixel_size : float
        Pupil-plane sampling pitch (m).
    r0 : float
        Fried parameter (m).
    target_rms : float
        Desired RMS of the phase inside the pupil (rad).
    seed : int | None
        RNG seed.

    Returns
    -------
    jax.Array
        Float32 array of shape ``(N, N)``.
    """
    phase = generate_phase_screen(N, pixel_size, r0, seed)
    phase = remove_piston(phase)

    # RMS over the pupil region.
    vals = jnp.where(pupil_mask_jax, phase, jnp.array(0.0))
    mean = jnp.sum(vals) / (jnp.sum(pupil_mask_jax) + 1e-30)
    var = jnp.sum((vals - mean) ** 2) / (jnp.sum(pupil_mask_jax) + 1e-30)
    raw_rms = jnp.sqrt(var)

    if raw_rms <= 1e-30:
        raise RuntimeError(
            "Generated phase screen has near-zero RMS; check PSD/sampling parameters."
        )
    return phase * (target_rms / raw_rms)
