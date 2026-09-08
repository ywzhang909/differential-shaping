"""
Simulation package: pupil, turbulence, DM, optics (all canonical PyTorch).

The simulation physics layer is implemented in PyTorch (float32 CPU).  Numpy
appears only as backward-compatibility shims for the visualisation layer and
in the ``to_numpy`` / ``to_torch`` conversion helpers.
"""

from .dm import generate_actuator_grid, generate_influence_functions
from .optics import (
    compute_metrics,
    crop_center,
    dm_surface,
    far_field_intensity_metric,
    far_field_intensity_padded,
    gaussian_window,
    to_numpy,
    to_torch,
)
from .pupil import (
    I0,
    XX,
    YY,
    I0_peak,
    I0_peak_t,
    I0_t,
    XX_t,
    YY_t,
    extent_pupil_mm,
    grid_1d,
    grid_1d_t,
    pupil,
    pupil_float,
    pupil_float_t,
    pupil_mask,
    xx_metric,
    xx_metric_t,
    yy_metric,
    yy_metric_t,
)
from .slm import (
    quantize_slm_phase,
    slm_far_field_intensity,
    slm_far_field_intensity_many,
    slm_field,
    slm_fill_mask,
)
from .turbulence import generate_phase_screen, generate_turbulence_phase, remove_piston

__all__ = [
    "I0",
    "XX",
    "YY",
    "I0_peak",
    "I0_peak_t",
    "I0_t",
    "XX_t",
    "YY_t",
    "compute_metrics",
    "crop_center",
    "dm_surface",
    "extent_pupil_mm",
    "far_field_intensity_metric",
    "far_field_intensity_padded",
    "gaussian_window",
    "generate_actuator_grid",
    "generate_influence_functions",
    "generate_phase_screen",
    "generate_turbulence_phase",
    # pupil (numpy viz shims)
    "grid_1d",
    # pupil (torch canonical)
    "grid_1d_t",
    "pupil",
    "pupil_float",
    "pupil_float_t",
    "pupil_mask",
    # slm
    "quantize_slm_phase",
    "slm_far_field_intensity",
    "slm_far_field_intensity_many",
    "slm_field",
    "slm_fill_mask",
    # turbulence / dm / optics
    "remove_piston",
    "to_numpy",
    "to_torch",
    "xx_metric",
    "xx_metric_t",
    "yy_metric",
    "yy_metric_t",
]
