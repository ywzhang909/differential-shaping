# -*- coding: utf-8 -*-
"""
Focal-plane intensity, performance metrics, DM surface, and display utilities.

This is the single canonical optics module for the simulation physics layer,
implemented fully in PyTorch (float32 CPU).  It absorbs the previous
``torch_optics.py`` counterparts so there is no numpy/torch duplication:

  - ``far_field_intensity_metric`` / ``compute_metrics`` / ``far_field_intensity_padded``
  - ``remove_piston`` (re-export convenience; canonical impl in ``turbulence.py``)
  - ``dm_surface`` (DM surface = inf_flat^T @ u)
  - ``gaussian_window`` (loss steering window for Torch-GD)
  - ``to_numpy`` / ``to_torch`` conversion helpers

All public functions accept and return torch tensors.  Visualisation modules
use the ``to_numpy`` helper at the numpy/torch boundary (rendering is
post-processing, not simulation physics).
"""

import numpy as np
import torch
import torch.fft

from ..params import N, pad_factor_show
from .pupil import pupil_float_t, I0_peak_t, xx_metric_t, yy_metric_t
from .turbulence import remove_piston

_DEVICE = torch.device("cpu")
_DTYPE = torch.float32

# Gaussian weight window used as the "maximize on-axis energy" steering loss.
# sigma ~ 3 pixels (approx. Airy-core radius at this sampling) -> dense smooth
# gradients across the central spot region.
_SIGMA_G = 3.0
_coord = torch.arange(N, dtype=_DTYPE, device=_DEVICE)
_YY_g, _XX_g = torch.meshgrid(_coord, _coord, indexing="ij")
gaussian_window = torch.exp(
    -((_XX_g - N // 2) ** 2 + (_YY_g - N // 2) ** 2) / (2 * _SIGMA_G**2)
)


def far_field_intensity_metric(phase: torch.Tensor) -> torch.Tensor:
    """``|fftshift(fft2(pupil * exp(i piston_rm(phase))))|^2`` (torch)."""
    E = pupil_float_t * torch.exp(1j * remove_piston(phase))
    return torch.abs(torch.fft.fftshift(torch.fft.fft2(E))) ** 2


def compute_metrics(
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (J = centroid mean radius in pixels, SR, intensity) as torch."""
    I = far_field_intensity_metric(phase)
    total = I.sum() + 1e-30
    cx = (xx_metric_t * I).sum() / total
    cy = (yy_metric_t * I).sum() / total
    r_pix = torch.sqrt((xx_metric_t - cx) ** 2 + (yy_metric_t - cy) ** 2)
    J_mr = (r_pix * I).sum() / total
    SR = I.max() / (I0_peak_t + 1e-30)
    return J_mr, SR, I


def far_field_intensity_padded(
    phase: torch.Tensor, pad_factor: int = pad_factor_show
) -> torch.Tensor:
    """Zero-padded Fraunhofer intensity for visualization only (torch)."""
    M = N * pad_factor
    s = (M - N) // 2
    E = pupil_float_t * torch.exp(1j * remove_piston(phase))
    Epad = torch.zeros((M, M), dtype=E.dtype, device=_DEVICE)
    Epad[s : s + N, s : s + N] = E
    return torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(Epad)))) ** 2


def crop_center(
    img: torch.Tensor, half_width_pix: int
) -> tuple[torch.Tensor, list[float]]:
    M = img.shape[0]
    c = M // 2
    a = max(0, c - half_width_pix)
    b = min(M, c + half_width_pix + 1)
    crop = img[a:b, a:b]
    # x/y in units of lambda*f/D.  One unpadded FFT pixel corresponds to lambda*f/D.
    extent = [
        (a - c) / pad_factor_show,
        (b - 1 - c) / pad_factor_show,
        (a - c) / pad_factor_show,
        (b - 1 - c) / pad_factor_show,
    ]
    return crop, extent


def dm_surface(u: torch.Tensor, inf_flat_T: torch.Tensor) -> torch.Tensor:
    """DM surface = inf_flat^T @ u, reshaped to (N, N)."""
    return (inf_flat_T @ u).reshape(N, N)


# ------------------------- conversion helpers ------------------------------
def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach a float/complex torch tensor and return a float64 numpy array."""
    return t.detach().cpu().numpy().astype(np.float64)


def to_torch(a: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    """Convert a numpy array to an autograd-tracked float32 CPU tensor."""
    t = torch.from_numpy(np.asarray(a, dtype=np.float64)).to(
        dtype=_DTYPE, device=_DEVICE
    )
    if requires_grad:
        t.requires_grad_(True)
    return t


def torch_far_field_intensity(phase: torch.Tensor) -> torch.Tensor:
    """Back-compat alias: same as :func:`far_field_intensity_metric`."""
    return far_field_intensity_metric(phase)


def torch_compute_metrics(
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Back-compat alias: same as :func:`compute_metrics`."""
    return compute_metrics(phase)


def torch_dm_surface(u: torch.Tensor, inf_flat_T: torch.Tensor) -> torch.Tensor:
    """Back-compat alias: same as :func:`dm_surface`."""
    return dm_surface(u, inf_flat_T)


def torch_far_field_intensity_padded(
    phase: torch.Tensor, pad_factor: int = pad_factor_show
) -> torch.Tensor:
    """Back-compat alias: same as :func:`far_field_intensity_padded`."""
    return far_field_intensity_padded(phase, pad_factor)