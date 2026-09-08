# -*- coding: utf-8 -*-
"""
Click-based entry point for the H-GD / SPGD / Torch-GD adaptive-optics
simulation and differential far-field beam shaping.

Compared to the original fixed SPGD-vs-H-GD spot-correction script, this
version lets the user choose, at the command line:

    * the **target shape**  -- ``point`` (Strehl spot correction),
      ``square`` or ``triangle`` (conservation-respecting far-field shaping);
    * the **algorithm**     -- ``spgd``, ``hgd`` or ``torch-gd``;
    * the **phase device**  -- ``dm`` (deformable mirror, ideal continuous phase),
      ``slm`` (pixelated SLM with adjustable fill factor) or ``ideal`` (perfect
      per-pixel phase; analytic point correction / direct-phase shaping);
    * whether the turbulence is **dynamic** (a time-evolving set of correlated
      phase screens, corrected frame by frame) or **static** (a single screen,
      optimised once).

Pipeline (per turbulence screen / frame):
    1. Generate a Kolmogorov-like turbulence phase screen.
    2. Build the deformable-mirror influence functions.
    3. Run the selected optimizer to either correct the wavefront to a point
       or reshape the far-field spot into the requested target shape.
    4. Plot and save the resulting figure.

Run ``python run.py --help`` for the full option list.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

from differential_shaping import params
from differential_shaping.optimization import (
    DEVICE_CHOICES,
    device_compute_metrics,
    device_forward,
    device_scope,
    gs_optimization,
    gs_shaping_optimization,
    hgd_optimization,
    hgd_shaping_optimization,
    make_square_target,
    make_triangle_target,
    spgd_optimization,
    spgd_shaping_optimization,
    target_shaping_optimization,
    torch_gd_optimization,
)
from differential_shaping.simulation import (
    generate_influence_functions,
    generate_phase_screen,
    generate_turbulence_phase,
    pupil_mask,
)
from differential_shaping.visualization import (
    build_shaping_frames,
    collect_frame_data_from_snapshots,
    get_frame_indices,
    plot_beam_shape,
    plot_convergence_all,
    plot_shaping_convergence,
    write_shaping_gif,
    write_step_gif,
)

# Algorithm choices exposed on the CLI (lower-case values).
ALGORITHMS = ("spgd", "hgd", "torch-gd", "gs")
SHAPES = ("point", "square", "triangle")

TORCH_ITERS = 1000  # Torch-GD converges in far fewer steps than the numeric methods.
TORCH_LR = 0.01
GS_ITERS = 600  # GS is a projection method; a few hundred steps suffice.
SHAPING_ITERS = 600  # differential shaping converges in a few hundred steps.


# ---------------------------------------------------------------------------
# Turbulence handling (static vs dynamic)
# ---------------------------------------------------------------------------
def _generate_single_screen() -> torch.Tensor:
    """One static Kolmogorov turbulence phase screen as a torch tensor."""
    return generate_turbulence_phase(
        params.N,
        params.pixel_size,
        params.r0,
        params.target_phase_rms,
        params.seed_phase,
    )


def _generate_dynamic_frames(n_frames: int, shift: int = 2) -> list[torch.Tensor]:
    """Generate a dynamic turbulence sequence via the Taylor frozen-flow model.

    One *large* Kolmogorov phase screen is synthesised (large enough that every
    sampled window is fully inside it), and each subsequent frame is an ``N x N``
    window moved diagonally by ``shift`` pixels from the previous one — i.e. the
    atmosphere is frozen and simply advects across the pupil, exactly as if a
    constant wind were blowing.  Each pupil-sized window is then pupil-masked and
    re-normalised to the target phase RMS.

    Args:
        n_frames: number of turbulence frames to crop out of the big screen.
        shift: pixels the viewing window moves per frame (the "wind speed").
            Must be >= 0; ``shift == 0`` gives n_frames identical (frozen) frames.

    Returns:
        List of ``(N, N)`` torch float32 phase screens (rad).
    """
    if shift < 0:
        raise ValueError(f"shift must be >= 0, got {shift}")

    # Big screen must fit the final window: last offset + N <= big size.
    big_N = params.N + max(n_frames - 1, 0) * shift
    big = generate_phase_screen(
        big_N,
        params.pixel_size,
        params.r0,
        params.seed_phase,
    )

    frames: list[torch.Tensor] = []
    for k in range(n_frames):
        off = k * shift
        win = big[off : off + params.N, off : off + params.N].clone()
        # Mask to the pupil aperture and re-scale its RMS to the target.
        win = win * pupil_mask
        rms = float(win[pupil_mask].std())
        if rms > 1e-30:
            win = win * (params.target_phase_rms / rms)
        frames.append(win)
    return frames


# ---------------------------------------------------------------------------
# Optimizer dispatch
# ---------------------------------------------------------------------------
def _run_point_correction(
    turb_phase: np.ndarray | torch.Tensor,
    inf_flat: np.ndarray,
    algorithm: str,
    max_iter: int,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correct the wavefront to a point (maximise on-axis / Strehl energy)."""
    if algorithm == "spgd":
        return spgd_optimization(
            turb_phase,
            inf_flat,
            max_iter=max_iter,
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
        )
    if algorithm == "hgd":
        return hgd_optimization(
            turb_phase,
            inf_flat,
            max_iter=max_iter,
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
        )
    if algorithm == "torch-gd":
        return torch_gd_optimization(
            turb_phase,
            inf_flat,
            max_iter=min(max_iter, TORCH_ITERS),
            lr=TORCH_LR,
            seed=params.seed_spgd,
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
        )
    if algorithm == "gs":
        return gs_optimization(
            turb_phase,
            inf_flat,
            max_iter=min(max_iter, GS_ITERS),
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
        )
    raise click.BadParameter(f"unknown algorithm: {algorithm}")


def _run_shaping(
    turb_phase: np.ndarray | torch.Tensor,
    inf_flat: np.ndarray,
    algorithm: str,
    shape: str,
    target: np.ndarray,
    max_iter: int,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
    direct_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the far-field spot to ``target`` (square / triangle)."""
    iters = min(max_iter, SHAPING_ITERS)
    if algorithm == "spgd":
        return spgd_shaping_optimization(
            turb_phase,
            inf_flat,
            target,
            max_iter=iters,
            label=shape,
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
        )
    if algorithm == "hgd":
        return hgd_shaping_optimization(
            turb_phase,
            inf_flat,
            target,
            max_iter=iters,
            label=shape,
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
        )
    if algorithm == "torch-gd":
        return target_shaping_optimization(
            turb_phase,
            inf_flat,
            target,
            max_iter=iters,
            lr=TORCH_LR,
            seed=params.seed_spgd,
            label=shape,
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
            direct_phase=direct_phase,
        )
    if algorithm == "gs":
        return gs_shaping_optimization(
            turb_phase,
            inf_flat,
            target,
            max_iter=iters,
            label=shape,
            snapshot_indices=snapshot_indices,
            snapshot_store=snapshot_store,
        )
    raise click.BadParameter(f"unknown algorithm: {algorithm}")


def _make_target(shape: str) -> np.ndarray | None:
    """Return the far-field target pattern for a shaping shape (None for point)."""
    if shape == "square":
        return make_square_target(half_width=6).numpy()
    if shape == "triangle":
        return make_triangle_target(size=11, apex="up").numpy()
    return None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _save_point_figure(
    turb_phase: torch.Tensor,
    phase_corr: torch.Tensor,
    J_init: float,
    SR_init: float,
    J_end: float,
    SR_end: float,
    out_path: Path,
    forward=None,
) -> None:
    """Simple initial-vs-corrected spot figure for the point-correction case.

    ``forward`` optionally replaces the padded ideal propagator with a device
    forward (e.g. the SLM cropped model) so the displayed spots match the
    device that actually produced the metrics.
    """
    import matplotlib.pyplot as plt

    from differential_shaping.simulation.optics import far_field_intensity_padded

    if forward is not None:
        I0_pad = np.max(np.asarray(forward(torch.zeros_like(turb_phase))))
    else:
        I0_pad = np.max(
            np.asarray(far_field_intensity_padded(torch.zeros_like(turb_phase)))
        )
    spots = [
        ("Initial", turb_phase, J_init, SR_init),
        ("Corrected", phase_corr, J_end, SR_end),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, phase, Jv, SRv) in zip(axes, spots):
        if forward is not None:
            I_rel = np.asarray(forward(phase)) / (I0_pad + 1e-30)
        else:
            I_rel = np.asarray(far_field_intensity_padded(phase)) / (I0_pad + 1e-30)
        I_db = np.log10(np.maximum(I_rel, 1e-8))
        ax.imshow(I_db, origin="lower", cmap="jet", vmin=-8, vmax=0)
        ax.set_title(f"{name}\nJ={Jv:.3f} pix, SR={SRv:.3f}")
        ax.set_xlabel(r"$x/(\lambda f/D)$")
        ax.set_ylabel(r"$y/(\lambda f/D)$")
    fig.suptitle("Point correction (Strehl / on-axis energy)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation (line charts + evolution GIFs -> report-XXXX.md)
# ---------------------------------------------------------------------------
_ALGO_LABEL = {"spgd": "SPGD", "hgd": "H-GD", "torch-gd": "Torch-GD", "gs": "GS"}


def _to_numpy_phase(phase):
    """Convert a torch tensor or numpy array to a plain numpy ndarray."""
    detach = getattr(phase, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(phase)


def _build_point_frame_series(
    turb_phase: np.ndarray | torch.Tensor,
    inf_flat: np.ndarray,
    snapshots: list[tuple[int, np.ndarray]],
    J_hist: np.ndarray,
    SR_hist: np.ndarray,
    iters: list[int],
) -> list[dict]:
    """Per-step render data (spot | DM) for the point-correction GIF."""
    return collect_frame_data_from_snapshots(
        turb_phase, inf_flat, snapshots, J_hist, SR_hist, iters
    )


def _build_shaping_frame_series(
    turb_phase: np.ndarray | torch.Tensor,
    inf_flat: np.ndarray,
    snapshots: list[tuple[int, np.ndarray]],
    target: np.ndarray,
    shape: str,
    iters: list[int],
    eng_hist: np.ndarray,
    direct_phase: bool = False,
) -> list:
    """Per-step render data for the beam-shaping GIF."""
    turb_phase_np = _to_numpy_phase(turb_phase)
    inf_flat_np = _to_numpy_phase(inf_flat)
    snap = sorted(snapshots, key=lambda x: x[0])
    phase_series: list[np.ndarray] = []
    steps: list[int] = []
    for it in iters:
        best_u = snap[0][1] if snap else np.zeros(inf_flat_np.shape[0])
        for s_it, s_u in snap:
            if s_it <= it:
                best_u = s_u
            else:
                break
        if direct_phase:
            # u is the flattened per-pixel phase (N*N,) -- no DM basis.
            phase_series.append(turb_phase_np + best_u.reshape(params.N, params.N))
        else:
            phase_series.append(
                turb_phase_np + (inf_flat_np.T @ best_u).reshape(params.N, params.N)
            )
        steps.append(it)
    return build_shaping_frames(phase_series, target, shape, steps, eng_hist)


def _write_report(
    base_dir: Path,
    shape: str,
    algorithm: str,
    dynamic: int,
    n_frames: int,
    max_iter: int,
    inf_flat: np.ndarray,
    screens: list,
    results: list[dict],
    report_iters: list[int],
    device: str = "dm",
) -> Path:
    """Assemble a report-XXXX.md with line charts and evolution GIFs.

    Writes everything under ``base_dir / report-{tag}`` and returns the
    markdown path.  ``tag`` embeds the run parameters plus a timestamp so the
    report filename is unique per run.
    """
    algo_label = _ALGO_LABEL[algorithm]
    tag = (
        f"{shape}_{algorithm}_{'dyn' if dynamic > 0 else 'static'}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    rdir = base_dir / f"report-{tag}"
    rdir.mkdir(parents=True, exist_ok=True)

    md_lines: list[str] = []
    md_lines.append(f"# Report — {algo_label} · {shape}")
    md_lines.append("")
    md_lines.append(f"- 目标形状 (shape): `{shape}`")
    md_lines.append(f"- 算法 (algorithm): `{algo_label}`")
    if dynamic > 0:
        md_lines.append(
            f"- 湍流 (turbulence): 动态 frozen-flow (dynamic, {n_frames} 帧, "
            f"每帧平移 {dynamic} px)"
        )
    else:
        md_lines.append("- 湍流 (turbulence): 静态 (static)")
    md_lines.append(f"- 迭代数 (max_iter): {max_iter}")
    md_lines.append(f"- 器件 (device): `{device}`")
    md_lines.append(f"- 帧数 (frames optimized): {len(results)}")
    md_lines.append("")

    ideal_point = device == "ideal" and shape == "point"

    for res in results:
        fi = res["frame"]
        md_lines.append(f"## Frame {fi}")
        md_lines.append("")
        md_lines.append("### 收敛曲线")
        md_lines.append("")

        if shape == "point":
            conv_path = rdir / f"conv_frame{fi}.png"
            plot_convergence_all(
                res["J_init"],
                res["SR_init"],
                {algo_label: (res["J_hist"], res["SR_hist"])},
                conv_path,
            )
            md_lines.append(f"![convergence]({conv_path.name})")
            md_lines.append("")
            md_lines.append(f"- J: {res['J_init']:.3f} → {res['J_end']:.3f} pix")
            md_lines.append(f"- SR: {res['SR_init']:.4f} → {res['SR_end']:.4f}")
            md_lines.append("")
            if ideal_point:
                md_lines.append(
                    "*(理想相位器件：校正为解析解（湍流共轭 → 剩余相位为零），"
                    "无演化动画)*"
                )
                md_lines.append("")
            else:
                md_lines.append("### 演化动画 (spot 左 / DM 面型 右)")
                md_lines.append("")
                frames = _build_point_frame_series(
                    screens[fi],
                    inf_flat,
                    res["snapshots"],
                    res["J_hist"],
                    res["SR_hist"],
                    report_iters,
                )
                gif_path = rdir / f"steps_frame{fi}.gif"
                write_step_gif(algo_label, frames, gif_path, duration_ms=200)
                md_lines.append(f"![steps]({gif_path.name})")
                md_lines.append("")
        else:
            conv_path = rdir / f"conv_frame{fi}.png"
            plot_shaping_convergence(res["eng_hist"], shape, conv_path)
            md_lines.append(f"![convergence]({conv_path.name})")
            md_lines.append("")
            md_lines.append(
                f"- energy-in-target: {res['eng_hist'][0]:.3f} → {res['energy']:.3f}"
            )
            md_lines.append("")
            md_lines.append("### 演化动画 (整形成目标形状)")
            md_lines.append("")
            target_np = _make_target(shape)
            assert target_np is not None, "shaping requires a square/triangle target"
            frames = _build_shaping_frame_series(
                screens[fi],
                inf_flat,
                res["snapshots"],
                target_np,
                shape,
                report_iters,
                res["eng_hist"],
                direct_phase=(device == "ideal"),
            )
            gif_path = rdir / f"shaping_frame{fi}.gif"
            write_shaping_gif(frames, gif_path, duration_ms=250)
            md_lines.append(f"![shaping]({gif_path.name})")
            md_lines.append("")

    markdown = "\n".join(md_lines) + "\n"
    md_path = rdir / f"report-{tag}.md"
    md_path.write_text(markdown, encoding="utf-8")
    logger.info(f"Wrote report: {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--shape",
    "-s",
    type=click.Choice(SHAPES),
    default="point",
    show_default=True,
    help="Target: 'point'=Strehl spot correction; 'square'/'triangle'=far-field beam shaping.",
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(ALGORITHMS),
    default="spgd",
    show_default=True,
    help="Optimizer to run.",
)
@click.option(
    "--device",
    "-D",
    type=click.Choice(DEVICE_CHOICES),
    default="dm",
    show_default=True,
    help="Phase-control device: 'dm'=deformable mirror (ideal continuous phase); "
    "'slm'=pixelated SLM (coarse pixel pitch + fill factor, optional 8-bit quantization); "
    "'ideal'=perfect per-pixel phase (analytic point correction; direct per-pixel "
    "torch-gd shaping).",
)
@click.option(
    "--fill-factor",
    type=click.FloatRange(0.05, 1.0),
    default=None,
    help="SLM pixel fill factor in (0, 1] (device=slm only; default: params.slm_fill_factor).",
)
@click.option(
    "--quantize/--no-quantize",
    "quantize",
    default=False,
    show_default=True,
    help="Quantize the SLM phase to 8-bit levels (device=slm only). Off by default "
    "so Torch-GD autograd stays differentiable (round breaks the gradient).",
)
@click.option(
    "--dynamic",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Dynamic turbulence wind speed in pixels/frame (Taylor frozen-flow: one large screen, "
    "crop N x N window shifted this many px per frame). 0 = static (single screen, one run).",
)
@click.option(
    "--n-frames",
    default=20,
    show_default=True,
    help="Number of turbulence frames when --dynamic > 0.",
)
@click.option(
    "--max-iter",
    type=click.IntRange(min=1),
    default=None,
    help="Override max iterations (algorithm-specific caps still apply).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output figure path (default: built from shape/algorithm).",
)
@click.option(
    "--report/--no-report",
    "report",
    default=False,
    show_default=True,
    help="Generate a report-XXXX.md with line charts and evolution GIFs.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for the generated report (default: script directory).",
)
def main(
    shape: str,
    algorithm: str,
    device: str,
    fill_factor: float | None,
    quantize: bool,
    dynamic: int,
    n_frames: int,
    max_iter: int | None,
    out: Path | None,
    report: bool,
    out_dir: Path | None,
) -> None:
    """Run the H-GD / SPGD / Torch-GD AO simulation and/or beam shaping."""
    max_iter = max_iter or params.max_iter
    logger.info(
        f"=== AO pipeline | shape={shape} algorithm={algorithm} device={device} "
        f"turbulence={'dynamic' if dynamic > 0 else 'static'} ==="
    )

    # Device validation: an ideal per-pixel phase device only makes sense with
    # the autograd shaper (16k pixel-phase actuators would be hopeless for the
    # finite-difference SPGD/H-GD/GS steppers).
    if device not in DEVICE_CHOICES:
        raise click.BadParameter(f"unknown device: {device!r}")
    if device == "ideal" and shape != "point" and algorithm != "torch-gd":
        raise click.BadParameter(
            "ideal-device shaping optimises the per-pixel phase directly and is "
            "only available with --algorithm torch-gd."
        )
    if quantize and device != "slm":
        raise click.BadParameter("--quantize only applies to --device slm.")
    ff = (
        fill_factor
        if (device == "slm" and fill_factor is not None)
        else params.slm_fill_factor
    )
    if device == "slm":
        logger.info(
            f"SLM device: fill_factor={ff:.3f} quantize={quantize} "
            f"pixel_pitch={params.slm_pixel_pitch_px:.0f} px"
        )
    use_quantize = quantize and device == "slm"
    if use_quantize and algorithm == "torch-gd":
        logger.warning(
            "SLM phase quantization with Torch-GD zeros the phase gradient "
            "(round is non-differentiable); the run will not converge."
        )
    forward = device_forward(device, ff if device == "slm" else None, use_quantize)
    cm_device = device_compute_metrics(forward)

    # 0. Shared DM influence functions.
    inf_funcs = generate_influence_functions()
    inf_flat = inf_funcs.reshape(params.n_act, -1).numpy()
    logger.info(
        f"DM: {params.n_act} actuators; spacing={params.act_spacing * 1e3:.3f} mm; "
        f"influence width={params.sigma_inf * 1e3:.3f} mm"
    )

    # 1. Build turbulence: a single static screen or a Taylor frozen-flow sequence.
    if dynamic > 0:
        screens = _generate_dynamic_frames(n_frames, shift=dynamic)
        logger.info(
            f"Dynamic turbulence (frozen-flow): {n_frames} frames, "
            f"window shifted {dynamic} px/frame"
        )
    else:
        screens = [_generate_single_screen()]

    # 2. Target pattern (None for point correction).
    target = _make_target(shape)

    # 3. Run the optimizer on each screen.
    results: list[dict] = []
    report_iters = None
    for i, turb_phase in enumerate(screens):
        # Frame indices for the evolution GIF (only needed when reporting).
        snapshots: list[tuple[int, np.ndarray]] = []
        if report:
            run_iters = (
                min(max_iter, params.max_iter)
                if shape == "point"
                else min(max_iter, SHAPING_ITERS)
            )
            report_iters = get_frame_indices(run_iters)
            snapshots = []

        if shape == "point" and device == "ideal":
            # Ideal device, point focus: the correction is analytic -- the
            # device imprints the exact turbulence conjugate, leaving a flat
            # (zero) residual phase.  No optimizer, no evolution.
            with torch.no_grad():
                J_init, SR_init, _ = cm_device(turb_phase)
                J_end, SR_end, _ = cm_device(torch.zeros_like(turb_phase))
            results.append(
                {
                    "frame": i,
                    "J_init": float(J_init),
                    "SR_init": float(SR_init),
                    "J_end": float(J_end),
                    "SR_end": float(SR_end),
                    "J_hist": np.array([float(J_init), float(J_end)]),
                    "SR_hist": np.array([float(SR_init), float(SR_end)]),
                    "dm_u": -turb_phase,
                    "snapshots": [],
                }
            )
            logger.info(
                f"Ideal-device correction[frame {i}]: SR {float(SR_init):.4f} -> "
                f"{float(SR_end):.4f}"
            )
        else:
            with device_scope(device, ff if device == "slm" else None, use_quantize):
                if shape == "point":
                    u, dm_u, J_hist, SR_hist = _run_point_correction(
                        turb_phase,
                        inf_flat,
                        algorithm,
                        max_iter,
                        snapshot_indices=report_iters,
                        snapshot_store=snapshots,
                    )
                    J_end, SR_end, _ = cm_device(turb_phase + torch.as_tensor(dm_u))
                    results.append(
                        {
                            "frame": i,
                            "J_init": float(J_hist[0]),
                            "SR_init": float(SR_hist[0]),
                            "J_end": float(J_end),
                            "SR_end": float(SR_end),
                            "J_hist": J_hist,
                            "SR_hist": SR_hist,
                            "dm_u": dm_u,
                            "snapshots": snapshots,
                        }
                    )
                    logger.info(
                        f"Correction[frame {i}]: SR {float(SR_hist[0]):.4f} -> "
                        f"{float(SR_hist[-1]):.4f}"
                    )
                else:
                    assert target is not None, "shaping requires a square/triangle target"
                    u, dm_u, loss_hist, eng_hist, I_np = _run_shaping(
                        turb_phase,
                        inf_flat,
                        algorithm,
                        shape,
                        target,
                        max_iter,
                        snapshot_indices=report_iters,
                        snapshot_store=snapshots,
                        direct_phase=(device == "ideal"),
                    )
                    results.append(
                        {
                            "frame": i,
                            "energy": float(eng_hist[-1]),
                            "eng_hist": eng_hist,
                            "I_np": I_np.copy(),
                            "snapshots": snapshots,
                        }
                    )
                    logger.info(
                        f"Shaping[{shape}][frame {i}]: energy-in-target "
                        f"{eng_hist[0]:.3f} -> {eng_hist[-1]:.3f}"
                    )

    # 4. Report.
    for r in results:
        if shape == "point":
            logger.info(f"  frame {r['frame']}: SR={r['SR_end']:.4f}")
        else:
            logger.info(f"  frame {r['frame']}: energy-in-target={r['energy']:.3f}")

    # 5. Save a figure (from the last screen).
    out_path = out or (
        Path(__file__).parent
        / f"ao_result_{shape}_{algorithm}_{device}_{'dynamic' if dynamic else 'static'}.png"
    )
    last = results[-1]
    if shape == "point":
        _save_point_figure(
            screens[-1],
            screens[-1] + torch.as_tensor(last["dm_u"]),
            last["J_init"],
            last["SR_init"],
            last["J_end"],
            last["SR_end"],
            out_path,
            forward=forward if device == "slm" else None,
        )
    else:
        assert target is not None, "shaping requires a square/triangle target"
        plot_beam_shape(
            last["I_np"],
            target,
            f"{shape} ({algorithm}, {device})",
            out_path,
        )
    logger.info(f"Saved figure: {out_path}")

    # 6. Generate the report with line charts + evolution GIFs (optional).
    if report:
        _base = out_dir or Path(__file__).parent
        _write_report(
            base_dir=_base,
            shape=shape,
            algorithm=algorithm,
            dynamic=dynamic,
            n_frames=n_frames,
            max_iter=max_iter,
            inf_flat=inf_flat,
            screens=screens,
            results=results,
            report_iters=report_iters or [0],
            device=device,
        )


if __name__ == "__main__":
    main()
