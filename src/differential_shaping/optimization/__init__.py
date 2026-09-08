# -*- coding: utf-8 -*-
"""Optimization package: SPGD, H-GD, Torch-GD, and GS algorithms."""

from .spgd import update_by_two_sided_perturbation, spgd_optimization
from .hgd import hgd_optimization
from .torch_gd import torch_gd_optimization
from .gs import gs_optimization, gs_shaping_optimization
from .shaping import (
    make_square_target,
    make_triangle_target,
    target_shaping_optimization,
    spgd_shaping_optimization,
    hgd_shaping_optimization,
    energy_in_target,
)
from .slm_shaping import slm_forward_cropped, slm_shaping_forward
from .devices import (
    DEVICE_CHOICES,
    device_compute_metrics,
    device_forward,
    device_scope,
)

__all__ = [
    "update_by_two_sided_perturbation",
    "spgd_optimization",
    "hgd_optimization",
    "torch_gd_optimization",
    "gs_optimization",
    "gs_shaping_optimization",
    "make_square_target",
    "make_triangle_target",
    "target_shaping_optimization",
    "spgd_shaping_optimization",
    "hgd_shaping_optimization",
    "energy_in_target",
    "slm_forward_cropped",
    "slm_shaping_forward",
    "DEVICE_CHOICES",
    "device_compute_metrics",
    "device_forward",
    "device_scope",
]
