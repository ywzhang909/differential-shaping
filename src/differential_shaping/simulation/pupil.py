# -*- coding: utf-8 -*-
"""
Pupil grid, mask, and focal-plane metric arrays.

This module is imported by turbulence, dm, optics, and visualization modules
to access the shared computational grid and pupil geometry.

**Primary API** – canonical PyTorch float32 tensors on CPU:
    ``grid_1d_t``, ``XX_t``, ``YY_t``, ``pupil_mask``, ``pupil_float_t``,
    ``I0_t``, ``I0_peak_t``, ``xx_metric_t``, ``yy_metric_t``

**Backward-compat shims** – plain numpy arrays kept for the matplotlib / PIL
visualization layer only (not used in simulation physics):
    ``grid_1d``, ``XX``, ``YY``, ``pupil``, ``pupil_float``, ``I0``,
    ``I0_peak``, ``xx_metric``, ``yy_metric``, ``extent_pupil_mm``
"""

import numpy as np
import torch
import torch.fft

from ..params import N, pixel_size, D

__all__ = [
    # torch primary API
    "grid_1d_t",
    "XX_t",
    "YY_t",
    "pupil_mask",
    "pupil_float_t",
    "I0_t",
    "I0_peak_t",
    "xx_metric_t",
    "yy_metric_t",
    # numpy backward-compat shims (visualization layer)
    "grid_1d",
    "XX",
    "YY",
    "pupil",
    "pupil_float",
    "I0",
    "I0_peak",
    "xx_metric",
    "yy_metric",
    "extent_pupil_mm",
]

# ------------------------- device / dtype defaults -------------------------
_DEVICE = torch.device("cpu")
_DTYPE = torch.float32

# ------------------------- torch grids and pupil -------------------------
grid_1d_t = (torch.arange(N, dtype=_DTYPE, device=_DEVICE) - N / 2) * pixel_size
XX_t, YY_t = torch.meshgrid(grid_1d_t, grid_1d_t, indexing="xy")
pupil_mask = (XX_t**2 + YY_t**2) <= (D / 2) ** 2        # bool tensor
pupil_float_t = pupil_mask.to(dtype=_DTYPE)

# ------------------------- torch focal-plane metric arrays -------------------------
I0_t = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil_float_t))) ** 2
I0_peak_t = I0_t.max()

# np.indices((N,N)) uses ij-indexing: first dim = row (y), second = col (x)
yy_metric_t, xx_metric_t = torch.meshgrid(
    torch.arange(N, dtype=_DTYPE, device=_DEVICE),
    torch.arange(N, dtype=_DTYPE, device=_DEVICE),
    indexing="ij",
)

# ------------------------- numpy backward-compat shims -------------------------
# Plain ndarray attributes derived from the torch tensors above.  These exist
# only so that the matplotlib / PIL visualization layer keeps working without
# modification.
grid_1d = grid_1d_t.numpy()
XX = XX_t.numpy()
YY = YY_t.numpy()
pupil = pupil_mask.numpy()          # bool ndarray
pupil_float = pupil_float_t.numpy()

extent_pupil_mm = [-D / 2 * 1e3, D / 2 * 1e3, -D / 2 * 1e3, D / 2 * 1e3]

I0 = I0_t.numpy()
I0_peak = float(I0_peak_t.item())

xx_metric = xx_metric_t.numpy()
yy_metric = yy_metric_t.numpy()
