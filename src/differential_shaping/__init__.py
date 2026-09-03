# -*- coding: utf-8 -*-
"""Differential shaping - H-GD / SPGD wavefront-sensorless AO simulation.

Packages:
    - simulation: pupil, turbulence, deformable mirror, far-field optics
    - optimization: SPGD and H-GD wavefront-sensorless optimizers
    - visualization: multi-panel comparison figures
"""

from . import simulation, optimization, visualization

__all__ = ["simulation", "optimization", "visualization"]