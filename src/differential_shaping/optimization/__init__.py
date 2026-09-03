# -*- coding: utf-8 -*-
"""Optimization package: SPGD, H-GD, and PyTorch-backprop (Torch-GD) algorithms."""

from .spgd import update_by_two_sided_perturbation, spgd_optimization
from .hgd import hgd_optimization
from .torch_gd import torch_gd_optimization
from .shaping import (
    make_square_target,
    make_triangle_target,
    target_shaping_optimization,
    energy_in_target,
)

__all__ = [
    "update_by_two_sided_perturbation",
    "spgd_optimization",
    "hgd_optimization",
    "torch_gd_optimization",
    "make_square_target",
    "make_triangle_target",
    "target_shaping_optimization",
    "energy_in_target",
]