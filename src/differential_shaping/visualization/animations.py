# -*- coding: utf-8 -*-
"""
Step-by-step evolution visualisation : far-field spot (LEFT) + DM surface (RIGHT).

For each optimizer the pipeline renders one combined two-panel figure per
(subsampled) iteration and assembles the frames into a looped GIF.  The spot
panel reuses the zero-padded, dB-scaled display from ``plots.py``; the DM panel
shows the mirror surface (signed, symmetric RdBu colormap) on the pupil grid.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ..params import pad_factor_show, spot_half_width_lamD, N
from ..simulation.optics import (
    far_field_intensity_padded,
    far_field_intensity_metric,
    crop_center,
    compute_metrics,
    to_numpy as _to_np,
    to_torch as _to_t,
)
from ..simulation.pupil import pupil, extent_pupil_mm


# ---------------------------------------------------------------------------
# Frame-index subsampling : zoom into early fast dynamics, coarse later.
# ---------------------------------------------------------------------------
def get_frame_indices(max_iter: int) -> list[int]:
    """Return ascending iteration indices to snapshot for a stepping GIF.

    Includes iteration 0, fine spacing over the first ~100 iterations, medium
    spacing through mid-convergence, and coarse spacing for the tail.  Always
    capped at ``max_iter - 1`` (the last recorded state) so the indices remain
    valid as indexes into ``J_history`` / ``SR_history`` (length ``max_iter``).
    """
    last = max(max_iter - 1, 0)
    indices: list[int] = [0]
    # fine: every 10 for iters 10..100
    indices.extend(range(10, min(101, last + 1), 10))
    # medium: every 20 for iters 200..300
    indices.extend(range(200, min(301, last + 1), 20))
    # coarse: every 50 for iters 400..last
    indices.extend(range(400, last + 1, 50))
    return sorted({int(i) for i in indices if i <= last})


# ---------------------------------------------------------------------------
# Reconstruct (spot phase, DM surface, metric) at arbitrary iteration index.
# ---------------------------------------------------------------------------
def _dm_at(u_it: np.ndarray, inf_flat: np.ndarray) -> np.ndarray:
    """DM surface (N, N) from actuator commands u_it (n_act,) and inf_flat."""
    return (inf_flat.T @ u_it).reshape(N, N)


def collect_frame_data(
    turb_phase: np.ndarray,
    inf_flat: np.ndarray,
    u_history: np.ndarray,       # (n_frames, n_act) sampled actuator commands
    J_history: np.ndarray,       # array indexed by iteration for metric lookup
    SR_history: np.ndarray,      # array indexed by iteration
    frame_idx: list[int],
) -> list[dict]:
    """Build per-frame render data for the numpy optimisers (SPGD / H-GD).

    Args:
        turb_phase: (N, N) turbulence phase.
        inf_flat:   (n_act, N*N) flattened influence functions.
        u_history:  (n_frames, n_act) actuator commands sampled at ``frame_idx``.
        J_history / SR_history: full-length metric histories (indexed by iter).
        frame_idx:  the iteration indices returned by ``get_frame_indices``.

    Returns:
        list of dicts, one per frame: {step, dm_u, phase, J, SR}.
    """
    if hasattr(turb_phase, 'detach'):
        turb_phase = _to_np(turb_phase)
    if hasattr(inf_flat, 'detach'):
        inf_flat = _to_np(inf_flat)
    data: list[dict] = []
    for k, it in enumerate(frame_idx):
        dm_k = _dm_at(u_history[k], inf_flat)
        phase_k = turb_phase + dm_k
        J_k = float(J_history[it])
        SR_k = float(SR_history[it])
        data.append({"step": int(it), "dm_u": dm_k, "phase": phase_k, "J": J_k, "SR": SR_k})
    return data


def collect_frame_data_from_snapshots(
    turb_phase: np.ndarray,
    inf_flat: np.ndarray,
    snapshots: list[tuple[int, np.ndarray]],
    J_history: np.ndarray,
    SR_history: np.ndarray,
    frame_idx: list[int],
) -> list[dict]:
    """Build per-frame render data from a ``(iter, u)`` snapshot list.

    Used uniformly by all three optimisers (SPGD / H-GD / Torch-GD).  For each
    requested frame index, use the closest recorded snapshot ``u`` to compute
    the DM surface and metrics.  Mirrors the numpy forward model (not torch) so
    every animation panel uses identical rendering math.
    """
    snap = sorted(snapshots, key=lambda x: x[0])

    # Map every frame index to the nearest *preceding-or-equal* snapshot state.
    def u_at(it: int) -> np.ndarray:
        best = snap[0][1]
        for s_it, s_u in snap:
            if s_it <= it:
                best = s_u
            else:
                break
        return best

    if hasattr(turb_phase, 'detach'):
        turb_phase = _to_np(turb_phase)
    if hasattr(inf_flat, 'detach'):
        inf_flat = _to_np(inf_flat)
    data: list[dict] = []
    for it in frame_idx:
        u_it = u_at(it)
        dm_it = _dm_at(u_it, inf_flat)
        phase_it = turb_phase + dm_it
        J_it = float(J_history[it]) if it < len(J_history) else float(np.nan)
        SR_it = float(SR_history[it]) if it < len(SR_history) else float(np.nan)
        data.append({"step": int(it), "dm_u": dm_it, "phase": phase_it, "J": J_it, "SR": SR_it})
    return data


# ---------------------------------------------------------------------------
# Combined two-panel frame : SPOT (left) | DM SURFACE (right)
# ---------------------------------------------------------------------------
def render_combined_frame(
    frame: dict,
    optimizer_name: str,
    out_png: Path | None = None,
) -> Image.Image:
    """Render one combined frame (spot left / DM right) and return a PIL image.

    Spot panel: zero-padded far-field intensity, dB referenced to the ideal
    (pupil-only) peak, cropped to +/- spot_half_width_lamD lambda*f/D units.
    DM panel : the mirror surface mask-masked to the pupil, symmetric RdBu_r.

    If ``out_png`` is given the frame is also written as an individual PNG.
    """
    half_pix = int(round(spot_half_width_lamD * pad_factor_show))

    # --- spot panel (reuse the exact display math from plots.py) -----------
    I0_pad = _to_np(far_field_intensity_padded(_to_t(np.zeros_like(frame["phase"]))))
    I0_pad_peak = float(np.max(I0_pad))
    I_rel = _to_np(far_field_intensity_padded(_to_t(frame["phase"]))) / (I0_pad_peak + 1e-30)
    I_db_raw, extent = crop_center(_to_t(10 * np.log10(np.maximum(I_rel, 1e-8))), half_pix)
    I_db = _to_np(I_db_raw)

    dm_show = np.ma.array(frame["dm_u"], mask=~pupil)
    dm_lim = max(float(np.max(np.abs(dm_show[~dm_show.mask]))), 1e-6) if dm_show.count() else 1.0

    fig, (ax_spot, ax_dm) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]}
    )

    im_spot = ax_spot.imshow(
        I_db, extent=extent, origin="lower", cmap="jet", vmin=-40, vmax=0
    )
    ax_spot.set_title(f"{optimizer_name} iter {frame['step']}\nJ={frame['J']:.2f} pix, SR={frame['SR']:.3f}")
    ax_spot.set_xlabel(r"$x/(\lambda f/D)$")
    ax_spot.set_ylabel(r"$y/(\lambda f/D)$")
    ax_spot.set_aspect("equal")
    plt.colorbar(im_spot, ax=ax_spot, fraction=0.046, pad=0.04, label="Intensity / ideal peak (dB)")

    im_dm = ax_dm.imshow(
        dm_show,
        extent=extent_pupil_mm,
        origin="lower",
        cmap="RdBu_r",
        vmin=-dm_lim,
        vmax=dm_lim,
        interpolation="nearest",
    )
    ax_dm.set_title(f"DM surface (RMS={float(np.std(frame['dm_u'][pupil])):.3f} rad)")
    ax_dm.set_xlabel("x (mm)")
    ax_dm.set_ylabel("y (mm)")
    ax_dm.set_aspect("equal")
    plt.colorbar(im_dm, ax=ax_dm, fraction=0.046, pad=0.04, label="rad")

    fig.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        pil = Image.open(out_png).copy()
        plt.close(fig)
        return pil

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# GIF assembly
# ---------------------------------------------------------------------------
def write_step_gif(
    optimizer_name: str,
    frames: list[dict],
    out_path: Path,
    duration_ms: int = 200,
    frame_dir: Path | None = None,
) -> Path:
    """Render combined frames and assemble a looped GIF.

    Args:
        optimizer_name: label used in the frame titles (e.g. "SPGD").
        frames:         per-frame data dicts (see ``collect_frame_data*``).
        out_path:       destination GIF path.
        duration_ms:    per-frame duration (ms).
        frame_dir:      optional dir; if given, each PNG frame is also saved.

    Returns:
        the written ``out_path``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    pil_frames: list[Image.Image] = []
    for i, fr in enumerate(frames):
        png_path = frame_dir / f"{optimizer_name.lower().replace('-', '')}_frame_{fr['step']:05d}.png" if frame_dir else None
        pil = render_combined_frame(fr, optimizer_name, out_png=png_path)
        pil_frames.append(pil.convert("P", palette=Image.ADAPTIVE, colors=256))

    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


# ---------------------------------------------------------------------------
# Beam-shaping evolution: far-field intensity (with target contour) only
# ---------------------------------------------------------------------------
def build_shaping_frames(
    phase_series: list[np.ndarray],
    target: np.ndarray,
    shape_label: str,
    iters: list[int],
    energy_series: np.ndarray,
) -> list[Image.Image]:
    """Render a stepped sequence of the far field reshaping into ``target``.

    Each frame shows the max-normalised (linear) far-field intensity with the
    binary target contour overlaid in white, plus an energy-in-target legend.

    Args:
        phase_series: list of (N, N) phase screens (already includes DM) at
            each requested iteration.
        target:       (N, N) binary target mask.
        shape_label:  e.g. "square" / "triangle".
        iters:        iteration index for each frame (len == len(phase_series)).
        energy_series: full-length energy-in-target history for metric lookup.

    Returns:
        list of PIL frames ready for ``write_shaping_gif``.
    """
    pil_frames: list[Image.Image] = []
    target_b = np.asarray(target).astype(bool)
    for k, phase in enumerate(phase_series):
        # Max-normalised linear far-field intensity.
        I = _to_np(far_field_intensity_metric(_to_t(phase)))
        I_n = I / (I.max() + 1e-12)
        it = iters[k]
        energy = float(energy_series[it]) if it < len(energy_series) else float(np.nan)

        fig, ax = plt.subplots(figsize=(5.6, 5))
        im = ax.imshow(I_n, origin="lower", cmap="hot", vmin=0, vmax=1,
                       interpolation="nearest")
        if target_b.any():
            ax.contour(target_b.astype(float), levels=[0.5], colors=["w"], linewidths=1.4)
        ax.set_title(f"{shape_label} shaping — iter {it}\nenergy in target = {energy:.3f}")
        ax.set_xlabel(r"$x/(\lambda f/D)$")
        ax.set_ylabel(r"$y/(\lambda f/D)$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised intensity")
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=256))
    return pil_frames


def write_shaping_gif(
    frames: list[Image.Image],
    out_path: Path,
    duration_ms: int = 200,
) -> Path:
    """Assemble pre-rendered shaping frames into a looped GIF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no shaping frames to assemble")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path