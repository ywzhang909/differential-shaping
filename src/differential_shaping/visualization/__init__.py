# -*- coding: utf-8 -*-
"""Visualization package: comparison plots, convergence curves, and step GIFs."""

from .plots import plot_results, plot_convergence_all, ALGO_COLORS
from .plots import (
    plot_beam_shape,
    plot_shaping_convergence,
    plot_shaping_convergence_comparison,
)
from .slm_plots import (
    DEFAULT_FILL_FACTORS,
    plot_slm_fill_factor_comparison,
    plot_slm_fill_mask_comparison,
    plot_slm_full_comparison,
)
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
    "plot_shaping_convergence_comparison",
    "DEFAULT_FILL_FACTORS",
    "plot_slm_fill_factor_comparison",
    "plot_slm_fill_mask_comparison",
    "plot_slm_full_comparison",
    "get_frame_indices",
    "collect_frame_data",
    "collect_frame_data_from_snapshots",
    "render_combined_frame",
    "write_step_gif",
    "build_shaping_frames",
    "write_shaping_gif",
]
