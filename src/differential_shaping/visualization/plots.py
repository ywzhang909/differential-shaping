# -*- coding: utf-8 -*-
"""
Visualization of SPGD vs H-GD optimisation results.

Generates a multi-panel figure comparing initial, SPGD-corrected,
and H-GD-corrected focal-plane spots plus convergence histories
and residual phase maps.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..params import pad_factor_show, spot_half_width_lamD
from ..simulation.pupil import pupil, extent_pupil_mm
from ..simulation.optics import far_field_intensity_padded, compute_metrics, crop_center, to_numpy as _to_np, to_torch as _to_t
from ..simulation.turbulence import remove_piston


# Colour map used for consistent per-algorithm styling across plots / animations.
ALGO_COLORS = {
    "SPGD": "tab:blue",
    "H-GD": "tab:orange",
    "Torch-GD": "tab:red",
}


def plot_convergence_all(
    J_init: float,
    SR_init: float,
    algo_results: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    """Plot J and SR convergence curves for all optimisers together.

    Args:
        J_init / SR_init: metric values of the uncorrected (turbulence-only) field.
        algo_results: dict mapping algorithm name -> (J_hist, SR_hist), e.g.
                      {"SPGD": (J_spgd, SR_spgd), "H-GD": ..., "Torch-GD": ...}.
        out_path: destination PNG path.
    """
    if not algo_results:
        raise ValueError("algo_results must contain at least one algorithm")

    fig, (ax_j, ax_sr) = plt.subplots(1, 2, figsize=(13, 5))

    for name, (J_hist, SR_hist) in algo_results.items():
        color = ALGO_COLORS.get(name, "tab:gray")
        it_j = np.arange(J_hist.size) if J_hist.ndim == 1 else J_hist.shape[1]
        ax_j.plot(np.arange(J_hist.size), J_hist, color=color, label=name)
        ax_sr.plot(np.arange(SR_hist.size), SR_hist, color=color, label=name)

    ax_j.axhline(J_init, color="k", linestyle="--", alpha=0.7, label="Initial")
    ax_sr.axhline(SR_init, color="k", linestyle="--", alpha=0.7, label="Initial")

    ax_j.set_xlabel("Iteration")
    ax_j.set_ylabel("J = mean radius (pixels)")
    ax_j.set_title("Centroid radius convergence")
    ax_j.grid(True, alpha=0.4)
    ax_j.legend(loc="best")

    ax_sr.set_xlabel("Iteration")
    ax_sr.set_ylabel("Strehl ratio")
    ax_sr.set_title("Strehl ratio convergence")
    ax_sr.grid(True, alpha=0.4)
    ax_sr.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results(
    turb_phase: np.ndarray,
    phase_spgd: np.ndarray,
    phase_hgd: np.ndarray,
    J_init: float,
    SR_init: float,
    J_spgd: np.ndarray,
    SR_spgd: np.ndarray,
    J_hgd: np.ndarray,
    SR_hgd: np.ndarray,
    out_path: Path,
) -> None:
    J_spgd_end, SR_spgd_end, _ = compute_metrics(_to_t(phase_spgd))
    J_spgd_end, SR_spgd_end = float(J_spgd_end), float(SR_spgd_end)
    J_hgd_end, SR_hgd_end, _ = compute_metrics(_to_t(phase_hgd))
    J_hgd_end, SR_hgd_end = float(J_hgd_end), float(SR_hgd_end)

    I0_pad = _to_np(far_field_intensity_padded(_to_t(np.zeros_like(turb_phase))))
    I0_pad_peak = np.max(I0_pad)
    spots = [
        ("Initial", turb_phase, J_init, SR_init),
        ("SPGD", phase_spgd, J_spgd_end, SR_spgd_end),
        ("H-GD", phase_hgd, J_hgd_end, SR_hgd_end),
    ]
    half_pix = int(round(spot_half_width_lamD * pad_factor_show))

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.05], height_ratios=[1, 1])

    last_im = None
    for col, (name, phase, Jv, SRv) in enumerate(spots):
        ax = fig.add_subplot(gs[0, col])
        I_rel = _to_np(far_field_intensity_padded(_to_t(phase))) / (I0_pad_peak + 1e-30)
        I_db_raw, extent = crop_center(_to_t(10 * np.log10(np.maximum(I_rel, 1e-8))), half_pix)
        I_db = _to_np(I_db_raw)
        last_im = ax.imshow(I_db, extent=extent, origin="lower", cmap="jet", vmin=-40, vmax=0)
        ax.set_title(f"{name}\nJ={Jv:.2f} pix, SR={SRv:.3f}")
        ax.set_xlabel(r"$x/(\lambda f/D)$")
        ax.set_ylabel(r"$y/(\lambda f/D)$")
        ax.set_aspect("equal")
    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(last_im, cax=cax, label="Intensity / ideal peak (dB)")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(J_spgd, label="SPGD")
    ax.plot(J_hgd, label="H-GD")
    ax.axhline(J_init, color="k", linestyle="--", label="Initial")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("J = mean radius (pixels)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(SR_spgd, label="SPGD")
    ax.plot(SR_hgd, label="H-GD")
    ax.axhline(SR_init, color="k", linestyle="--", label="Initial")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SR")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # Initial phase screen display
    ax = fig.add_subplot(gs[1, 2])
    phase_show = np.ma.array(_to_np(remove_piston(_to_t(turb_phase))), mask=~pupil)
    lim = np.max(np.abs(phase_show))
    im_phase = ax.imshow(
        phase_show,
        extent=extent_pupil_mm,
        origin="lower",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )
    ax.set_title(f"Initial phase screen\nRMS={np.std(_to_np(turb_phase)[pupil]):.3f} rad")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(im_phase, ax=ax, fraction=0.046, pad=0.04, label="rad")

    # SPGD residual phase display
    ax = fig.add_subplot(gs[1, 3])
    residual_spgd = _to_np(remove_piston(_to_t(phase_spgd)))
    residual_spgd_show = np.ma.array(residual_spgd, mask=~pupil)
    lim_spgd = max(np.max(np.abs(residual_spgd_show)), 1e-6)
    im_res_spgd = ax.imshow(
        residual_spgd_show,
        extent=extent_pupil_mm,
        origin="lower",
        cmap="RdBu_r",
        vmin=-lim_spgd,
        vmax=lim_spgd,
        interpolation="nearest",
    )
    ax.set_title(f"SPGD residual phase\nRMS={np.std(residual_spgd[pupil]):.3f} rad")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(im_res_spgd, ax=ax, fraction=0.046, pad=0.04, label="rad")

    # H-GD residual phase display
    ax = fig.add_subplot(gs[1, 4])
    residual = _to_np(remove_piston(_to_t(phase_hgd)))
    residual_show = np.ma.array(residual, mask=~pupil)
    lim = max(np.max(np.abs(residual_show)), 1e-6)
    im_res = ax.imshow(
        residual_show,
        extent=extent_pupil_mm,
        origin="lower",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )
    ax.set_title(f"H-GD residual phase\nRMS={np.std(residual[pupil]):.3f} rad")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(im_res, ax=ax, fraction=0.046, pad=0.04, label="rad")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Target-shaping specific plots
# ---------------------------------------------------------------------------
def plot_beam_shape(
    I_np: np.ndarray,
    target: np.ndarray,
    shape_label: str,
    out_path: Path,
    iteration: int | None = None,
    contour_level: float = 0.35,
) -> None:
    """Render the shaped far-field intensity with the target contour overlaid.

    Uses a linear, max-normalised color map so the *shape* of the bright region
    (square / triangle) is clearly visible, and draws the binary target as a
    white contour.  ``I_np`` is the max-normalised far-field intensity to plot.
    """
    I_n = I_np / (I_np.max() + 1e-12)
    target_b = np.asarray(target).astype(bool)

    fig, ax = plt.subplots(figsize=(6.4, 6))
    im = ax.imshow(
        I_n,
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    # target contour (only the boundary, from the binary mask)
    if target_b.any():
        ax.contour(target_b.astype(float), levels=[0.5], colors=["w"], linewidths=1.6)
    title = f"Far-field ⇒ {shape_label}"
    if iteration is not None:
        title += f" (iter {iteration})"
    ax.set_title(title)
    ax.set_xlabel(r"$x/(\lambda f/D)$")
    ax.set_ylabel(r"$y/(\lambda f/D)$")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised intensity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_shaping_convergence(
    energy_hist: np.ndarray,
    shape_label: str,
    out_path: Path,
) -> None:
    """Plot the conservation-respecting energy-in-target convergence curve."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(energy_hist.size), energy_hist, color="tab:red", lw=1.8,
            label=f"energy in {shape_label} target")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of conserved energy in target")
    ax.set_title(f"Beam shaping convergence — {shape_label}")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
