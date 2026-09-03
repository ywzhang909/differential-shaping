# -*- coding: utf-8 -*-
"""
Simulation package: pupil, turbulence, DM, optics (all canonical PyTorch).

The simulation physics layer is implemented in PyTorch (float32 CPU).  Numpy
appears only as backward-compatibility shims for the visualisation layer and
in the ``to_numpy`` / ``to_torch`` conversion helpers.
"""

from .pupil import (
    grid_1d,
    grid_1d_t,
    XX,
    XX_t,
    YY,
    YY_t,
    pupil,
    pupil_mask,
    pupil_float,
    pupil_float_t,
    extent_pupil_mm,
    I0,
    I0_t,
    I0_peak,
    I0_peak_t,
    xx_metric,
    xx_metric_t,
    yy_metric,
    yy_metric_t,
)
from .turbulence import remove_piston, generate_turbulence_phase
from .dm import generate_actuator_grid, generate_influence_functions
from .optics import (
    far_field_intensity_metric,
    compute_metrics,
    far_field_intensity_padded,
    crop_center,
    dm_surface,
    gaussian_window,
    to_numpy,
    to_torch,
)

__all__ = [
    # pupil (torch canonical)
    "grid_1d_t", "XX_t", "YY_t", "pupil_mask", "pupil_float_t",
    "I0_t", "I0_peak_t", "xx_metric_t", "yy_metric_t",
    # pupil (numpy viz shims)
    "grid_1d", "XX", "YY", "pupil", "pupil_float",
    "extent_pupil_mm", "I0", "I0_peak", "xx_metric", "yy_metric",
    # turbulence / dm / optics
    "remove_piston", "generate_turbulence_phase",
    "generate_actuator_grid", "generate_influence_functions",
    "far_field_intensity_metric", "compute_metrics",
    "far_field_intensity_padded", "crop_center",
    "dm_surface", "gaussian_window", "to_numpy", "to_torch",
]