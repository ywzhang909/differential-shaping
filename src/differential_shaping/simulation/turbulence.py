# -*- coding: utf-8 -*-
"""
Turbulence phase screen generation (Kolmogorov-like spectral synthesis).

This module has been rewritten to operate entirely in canonical PyTorch
float32 on CPU — no NumPy is used in the physics.  The Kolmogorov PSD
form, the ``* (N**2)`` ifft scaling, and the overall algorithm are
unchanged; only the backend changed from NumPy to PyTorch.

**Determinism note**: the sampled noise realization produced by this
torch implementation differs numerically from the old NumPy version even
for the same seed.  This is expected — the two RNG streams are
independent.  Within this module the output is fully deterministic given
the same seed (bitwise-identical tensors).
"""

import torch
import torch.fft

from ..params import N, pixel_size, r0, target_phase_rms
from .pupil import pupil_mask

__all__ = ["remove_piston", "generate_turbulence_phase"]

_DEVICE = torch.device("cpu")
_DTYPE = torch.float32


def remove_piston(phase: torch.Tensor) -> torch.Tensor:
    """Remove piston inside the pupil and set the outside-pupil area to 0."""
    out = phase.clone()
    piston = out[pupil_mask].mean()
    out = out - piston
    out[~pupil_mask] = 0.0
    return out


def generate_turbulence_phase(
    N: int,
    pixel_size: float,
    r0: float,
    target_rms: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a Kolmogorov-like phase screen and scale pupil RMS to target_rms.

    Discrete spectral synthesis note:
    - frequency spacing is df = 1 / (N * pixel_size) = 1 / D;
    - torch.fft.ifft2 contains 1 / N^2, so multiply the inverse
      transform by N^2.

    Parameters
    ----------
    N : int
        Grid size (pixels per side).
    pixel_size : float
        Pupil-plane sampling pitch (m).
    r0 : float
        Fried parameter (m).
    target_rms : float, optional
        Desired RMS of the phase inside the pupil (rad). Default 1.0.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(N, N)`` on CPU.
    """
    gen = torch.Generator(device=_DEVICE)
    if seed is not None:
        gen.manual_seed(seed)

    D_local = N * pixel_size
    df = 1.0 / D_local

    fx = torch.fft.fftfreq(N, d=pixel_size, device=_DEVICE, dtype=_DTYPE)
    fy = torch.fft.fftfreq(N, d=pixel_size, device=_DEVICE, dtype=_DTYPE)
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    fr = torch.sqrt(FX**2 + FY**2)

    PSD = torch.zeros_like(fr)
    mask = fr > 0
    PSD[mask] = 0.023 * r0 ** (-5.0 / 3.0) * fr[mask] ** (-11.0 / 3.0)

    # Complex noise via seeded generator (two real draws combined).
    noise = (
        torch.randn((N, N), generator=gen, device=_DEVICE, dtype=_DTYPE)
        + 1j
        * torch.randn((N, N), generator=gen, device=_DEVICE, dtype=_DTYPE)
    ).to(torch.complex64)

    phase_fft = torch.sqrt(PSD) * noise * df
    phase = torch.real(torch.fft.ifft2(phase_fft)) * (N**2)

    phase = remove_piston(phase)

    raw_rms = phase[pupil_mask].std()
    if raw_rms <= 1e-30:
        raise RuntimeError(
            "Generated phase screen has near-zero RMS; "
            "check PSD/sampling parameters."
        )
    phase = phase * (torch.tensor(target_rms, dtype=_DTYPE, device=_DEVICE) / raw_rms)
    return phase
