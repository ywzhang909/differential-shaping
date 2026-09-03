# -*- coding: utf-8 -*-
"""
Full-pipeline entry point : run SPGD / H-GD / Torch-GD, then produce the
step-evolution GIFs, the convergence-curve figure, and a Markdown report.

All generated artifacts (GIFs, figures, report) plus a copy of the original
single-file script are written under ``docs/``.

Layout produced:
    docs/
        H_GD_phase_spot_fixedv2.py          # copy of the original script
        report/
            REPORT.md
            figures/convergence_all.png
            gifs/steps_spgd.gif
            gifs/steps_hgd.gif
            gifs/steps_torch_gd.gif
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from loguru import logger

from differential_shaping import params
from differential_shaping.simulation import (
    generate_turbulence_phase,
    generate_influence_functions,
    compute_metrics,
    pupil_mask,
)
from differential_shaping.optimization import (
    spgd_optimization,
    hgd_optimization,
    torch_gd_optimization,
)
from differential_shaping.visualization import (
    get_frame_indices,
    collect_frame_data_from_snapshots,
    write_step_gif,
    plot_convergence_all,
)

PROJECT_ROOT = Path(__file__).parent
ORIGINAL_SCRIPT = PROJECT_ROOT / "H_GD_phase_spot_fixedv2.py"
DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_DIR = DOCS_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"
GIFS_DIR = REPORT_DIR / "gifs"

# Per-algorithm run settings (Torch-GD runs fewer iterations because each step
# is a full forward+backward autograd pass and converges appreciably faster).
TORCH_ITERS = 1000
TORCH_LR = 0.01


def _run_optimizer(
    name: str,
    turb_phase: np.ndarray,
    inf_flat: np.ndarray,
    frames_idx: list[int],
) -> dict:
    """Run one optimizer and return results + ``(iter, u)`` snapshots."""
    logger.info(f"Running {name} ...")
    snapshots: list[tuple[int, np.ndarray]] = []
    if name == "SPGD":
        u, dm_u, J_hist, SR_hist = spgd_optimization(
            turb_phase, inf_flat, snapshot_indices=frames_idx, snapshot_store=snapshots
        )
    elif name == "H-GD":
        u, dm_u, J_hist, SR_hist = hgd_optimization(
            turb_phase, inf_flat, snapshot_indices=frames_idx, snapshot_store=snapshots
        )
    elif name == "Torch-GD":
        u, dm_u, J_hist, SR_hist = torch_gd_optimization(
            turb_phase,
            inf_flat,
            max_iter=TORCH_ITERS,
            lr=TORCH_LR,
            seed=params.seed_spgd,
            snapshot_indices=frames_idx,
            snapshot_store=snapshots,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    return {"u": u, "dm_u": dm_u, "J_hist": J_hist, "SR_hist": SR_hist, "snapshots": snapshots}


def _write_report(markdown: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    logger.info(f"Wrote report: {path}")


def main() -> None:
    logger.info("=== H-GD / SPGD / Torch-GD wavefront-sensorless AO pipeline ===")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GIFS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Shared turbulence field + DM.
    turb_phase = generate_turbulence_phase(
        params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
    )
    logger.info(f"Turbulence pupil RMS = {float(turb_phase[pupil_mask].std()):.6f} rad")

    J_init, SR_init, _ = compute_metrics(turb_phase)
    J_init = float(J_init)
    SR_init = float(SR_init)
    logger.info(f"Initial (no correction): J={J_init:.3f} pix, SR={SR_init:.4f}")

    inf_funcs = generate_influence_functions()
    inf_flat = inf_funcs.reshape(params.n_act, -1)

    # 2. Run all three optimizers.
    spgd = _run_optimizer("SPGD", turb_phase, inf_flat, get_frame_indices(params.max_iter))
    hgd = _run_optimizer("H-GD", turb_phase, inf_flat, get_frame_indices(params.max_iter))
    torchg = _run_optimizer("Torch-GD", turb_phase, inf_flat, get_frame_indices(TORCH_ITERS))

    results = {
        "SPGD": (spgd["J_hist"], spgd["SR_hist"]),
        "H-GD": (hgd["J_hist"], hgd["SR_hist"]),
        "Torch-GD": (torchg["J_hist"], torchg["SR_hist"]),
    }
    final_metrics = {
        name: (float(hist[0][-1]), float(hist[1][-1])) for name, hist in results.items()
    }

    # 3. Per-algorithm step-evolution GIFs (spot LEFT / DM RIGHT per frame).
    gif_paths: dict[str, Path] = {}
    for name, res in [("SPGD", spgd), ("H-GD", hgd), ("Torch-GD", torchg)]:
        frames_idx = get_frame_indices(
            params.max_iter if name != "Torch-GD" else TORCH_ITERS
        )
        frames = collect_frame_data_from_snapshots(
            turb_phase, inf_flat, res["snapshots"], res["J_hist"], res["SR_hist"], frames_idx
        )
        gif_path = GIFS_DIR / f"steps_{name.lower().replace('-', '')}.gif"
        write_step_gif(name, frames, gif_path, duration_ms=200, frame_dir=GIFS_DIR / "frames")
        gif_paths[name] = gif_path
        logger.info(f"GIF saved: {gif_path} ({len(frames)} frames)")

    # 4. Combined convergence-curve figure.
    conv_path = FIGURES_DIR / "convergence_all.png"
    plot_convergence_all(J_init, SR_init, results, conv_path)
    logger.info(f"Convergence figure saved: {conv_path}")

    # 5. Copy the original single-file script into docs/.
    if ORIGINAL_SCRIPT.exists():
        shutil.copy2(ORIGINAL_SCRIPT, DOCS_DIR / ORIGINAL_SCRIPT.name)
        logger.info(f"Copied original script -> {DOCS_DIR / ORIGINAL_SCRIPT.name}")

    # 6. Generate the Markdown report.
    rows = "\n".join(
        f"| {name} | {len(hist[0])} | {hist[0][-1]:.3f} | {hist[1][-1]:.4f} |"
        f" {J_init - hist[0][-1]:.3f} | {hist[1][-1] - SR_init:.4f} |"
        for name, hist in results.items()
    )
    markdown = f"""# H-GD / SPGD / Torch-GD 无波前传感自适应光学仿真报告

本报告对比三种无波前传感优化算法在湍流相位校正下的性能:
随机并行梯度下降(SPGD)、Hadamard 梯度下降(H-GD)、以及基于 **PyTorch 反向传播**的梯度下降(Torch-GD)。

## 一、系统与仿真参数

| 参数 | 数值 | 说明 |
|---|---|---|
| 波长 $\\lambda$ | 532 nm | |
| 采样间距 | 10 μm | 出瞳平面 |
| 光栅 $N \\times N$ | 128 × 128 | |
| 口径 | 1.28 mm | $D = N \\cdot\\Delta$ |
| 焦距 $f$ | 0.1 m | |
| Fried 参数 $r_0$ | 60 μm | |
| 目标相位 RMS | 1.0 rad | |
| DM 促动器 | 16 × 16 (256) | 高斯影响函数 |

## 二、三种优化算法

| 算法 | 方法 | 每次迭代代价 |
|---|---|---|
| SPGD | 双边随机扰动梯度估计 | 2 × 前向传播 |
| H-GD | Hadamard 扰动, 遍历模式空间 | 2 × 前向传播 |
| **Torch-GD** | **PyTorch 自动微分, 反向传播梯度 + Adam** | **1 前向 + 1 反向** |

Torch-GD 将可微分远场传播模型(`simulation.optics`)作为前向,
对 DM 促动器命令 `u` 做自动微分, 最小化高斯加权轴上能量负值(等效于最大化 Strehl 比), 用 Adam 更新。

## 三、收敛性能对比

| 算法 | 迭代数 | 最终 J (pix) | 最终 SR | ΔJ | ΔSR |
|---|---|---|---|---|---|
{rows}

## 四、逐步演化动画 (Spot 左 / DM 面型 右)

每组为单张帧图: 左侧为远期光斑(dB 色标), 右侧为 DM 面型(对称 RdBu)。

- [SPGD 逐步演化](./gifs/steps_spgd.gif)
- [H-GD 逐步演化](./gifs/steps_hgd.gif)
- [Torch-GD 逐步演化](./gifs/steps_torch_gd.gif)

![SPGD](./gifs/steps_spgd.gif)
![H-GD](./gifs/steps_hgd.gif)
![Torch-GD](./gifs/steps_torch_gd.gif)

## 五、收敛曲线

![Convergence curves](./figures/convergence_all.png)

## 六、原始脚本

原始单文件实现见 [`H_GD_phase_spot_fixedv2.py`](../H_GD_phase_spot_fixedv2.py)。
"""
    _write_report(markdown, REPORT_DIR / "REPORT.md")

    # 7. Summary log.
    logger.info("\n===== Final performance comparison =====")
    for name, hist in results.items():
        logger.info(
            f"{name:9s} final: J={hist[0][-1]:.3f} pix, SR={hist[1][-1]:.4f}"
            f"  (ΔJ={J_init - hist[0][-1]:.3f}, ΔSR={hist[1][-1] - SR_init:.4f})"
        )


if __name__ == "__main__":
    main()