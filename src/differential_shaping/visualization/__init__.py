# -*- coding: utf-8 -*-
"""Visualization package: comparison plots, convergence curves, and step GIFs."""

from .plots import plot_results, plot_convergence_all, ALGO_COLORS
from .plots import plot_beam_shape, plot_shaping_convergence
from .animations import (
    get_frame_indices,
    collect_frame_data,
    collect_frame_data_from_snapshots,
    render_combined_frame,
    write_step_gif,
    build_shaping_frames,
    write_shaping_gif,
)

__all__ = [
    "plot_results",
    "plot_convergence_all",
    "ALGO_COLORS",
    "plot_beam_shape",
    "plot_shaping_convergence",
    "get_frame_indices",
    "collect_frame_data",
    "collect_frame_data_from_snapshots",
    "render_combined_frame",
    "write_step_gif",
    "build_shaping_frames",
    "write_shaping_gif",
]