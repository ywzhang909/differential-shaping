# -*- coding: utf-8 -*-
"""
SLM 像素效应下的光束整形实验入口 (填充因子 × 算法对比)。

在既有 "三种算法 (SPGD / H-GD / Torch-GD) + 目标损失远场整形" 的实验框架上,
把前向模型替换为像素化 SLM 模型 (:func:`slm_far_field_intensity`), 进而对比:

  - 不同 SLM 填充因子 (默认 60% / 80% / 100%) 下, 同样 600 次迭代的整形效果;
  - 三种算法 (数值梯度 SPGD / H-GD 与反向传播 Torch-GD) 各自的表现。

物理要点
--------
像素化 SLM 的填充因子 < 1 时:

  1. 死区遮挡使总透过能量近似正比于 fill_factor;
  2. 像素周期在远场产生离散衍射级 (±16·n λf/D);
  3. 有限有效区带来 sinc 包络。

因此可用于整形 (目标区能量捕获) 的守恒能量随填充因子下降而减少, 且中央级
强度形状被 sinc 包络调制 —— 预期整形成本随填充因子减小而劣化, 且只有
Torch-GD (反向传播, 精确梯度) 能真正整形, SPGD / H-GD 仍停留在初始值附近
(与无像素化时的结论一致)。

实现方式
--------
所有三个整形优化器内部都通过 ``shaping.shaping_metric -> shaping.far_field_intensity_metric``
(模块级名字) 调用前向模型。实验脚本用一个上下文管理器在运行时把该模块级名字
替换为 SLM 像素化前向 (细网格 FFT -> 裁剪到中央 (N, N) 窗口, 即与理想前向
相同的 ±64 lambda*f/D 视场), 退出时恢复原前向 —— 三种算法因此优化的是
*完全相同* 的目标损失, 对比公平。

用法::

    # 默认 60% / 80% / 100% x [square, triangle] x [SPGD, H-GD, Torch-GD]
    python run_slm_shaping.py compare

    # 自定义填充因子与迭代数
    python run_slm_shaping.py compare --fill-factors 0.5,0.75,1.0 --iterations 400

生成 (默认输出目录 ``slm_shaping_results/``):
    - ``shaping_<shape>_ff<ff>_conv.png``  每个 (目标, 填充因子) 的三方法收敛对比
    - ``shaping_<shape>_final_grid.png``   每个目标的 [填充因子 x 方法] 最终光斑网格
    - ``slm_shaping_<shape>_energy_bar.png`` 每个目标的最终能量入目标条形图
    - ``shaping_summary.csv``              全部运行结果汇总表
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.image import AxesImage

from differential_shaping import params
from differential_shaping.optimization import (
    hgd_shaping_optimization,
    make_square_target,
    make_triangle_target,
    slm_shaping_forward,
    spgd_shaping_optimization,
    target_shaping_optimization,
)
from differential_shaping.simulation import (
    generate_influence_functions,
    generate_turbulence_phase,
)
from differential_shaping.visualization import plot_shaping_convergence_comparison

OUT_DIR = Path(__file__).parent / "slm_shaping_results"

DEFAULT_FILL_FACTORS = (0.60, 0.80, 1.00)
DEFAULT_SHAPES = ("square", "triangle")
SHAPING_METHODS = ("SPGD", "H-GD", "Torch-GD")
SHAPING_FILE_TAG = {"SPGD": "spgd", "H-GD": "hgd", "Torch-GD": "torch_gd"}
METHOD_COLORS = {"SPGD": "tab:blue", "H-GD": "tab:green", "Torch-GD": "tab:red"}

SHAPER_TARGETS = {
    "square": make_square_target(half_width=6),  # 13 x 13 px
    "triangle": make_triangle_target(size=11, apex="up"),  # 11 px 三角形
}


def _run_one_shaper(
    method: str,
    shape: str,
    target_np: np.ndarray,
    turb_phase: np.ndarray,
    inf_flat: np.ndarray,
    max_iter: int,
) -> dict:
    """以 SLM 像素化前向运行一次整形 (与 run_full._run_one_shaper 同构)。"""
    if method == "SPGD":
        u, dm_u, loss_hist, eng_hist, I_np = spgd_shaping_optimization(
            turb_phase, inf_flat, target_np, max_iter=max_iter, label=shape
        )
    elif method == "H-GD":
        u, dm_u, loss_hist, eng_hist, I_np = hgd_shaping_optimization(
            turb_phase, inf_flat, target_np, max_iter=max_iter, label=shape
        )
    else:  # Torch-GD (反向传播)
        u, dm_u, loss_hist, eng_hist, I_np = target_shaping_optimization(
            turb_phase,
            inf_flat,
            target_np,
            max_iter=max_iter,
            lr=0.02,  # SHAPING_LR, 同 run_full.py
            seed=params.seed_spgd,
            label=shape,
        )
    return {
        "method": method,
        "shape": shape,
        "u": u,
        "dm_u": dm_u,
        "loss_hist": loss_hist,
        "energy_hist": eng_hist,
        "I_np": I_np,
    }


def _plot_final_grid(
    results: dict[tuple, dict],
    shape: str,
    fill_factors: tuple[float, ...],
    out_path: Path,
    iterations: int = 600,
) -> None:
    """[填充因子 x 方法] 最终光斑网格图 (每一格叠加目标轮廓)。"""
    n_ff, n_m = len(fill_factors), len(SHAPING_METHODS)
    fig, axes = plt.subplots(n_ff, n_m, figsize=(4.2 * n_m, 4.0 * n_ff))
    if n_ff == 1:
        axes = axes[None, :]
    if n_m == 1:
        axes = axes[:, None]
    im: AxesImage | None = None
    target_np = SHAPER_TARGETS[shape].numpy()
    for i, ff in enumerate(fill_factors):
        for j, method in enumerate(SHAPING_METHODS):
            ax = axes[i, j]
            res = results[(shape, ff, method)]
            I_n = res["I_np"] / (res["I_np"].max() + 1e-12)
            im = ax.imshow(
                I_n,
                origin="lower",
                cmap="hot",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            if target_np.any():
                ax.contour(
                    target_np.astype(float),
                    levels=[0.5],
                    colors=["w"],
                    linewidths=1.0,
                )
            title = f"{method} · FF={ff:.2f}" if i == 0 else f"FF={ff:.2f}"
            ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"FF={ff:.0%}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.text(
                0.02,
                0.98,
                f"E={res['energy_hist'][-1]:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="lime",
                bbox=dict(boxstyle="round", fc="black", alpha=0.55, pad=0.15),
            )
    fig.suptitle(
        f"SLM pixelated beam shaping → {shape} ({iterations} iters)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    cbar_ax = fig.add_axes((0.955, 0.12, 0.02, 0.72))
    assert im is not None
    fig.colorbar(im, cax=cbar_ax, label="Normalised intensity")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved final grid: {out_path}")


def _plot_energy_bar(
    results: dict[tuple, dict],
    shape: str,
    fill_factors: tuple[float, ...],
    out_path: Path,
) -> None:
    """最终能量入目标 (按填充因子分组的算法条形图)。"""
    x = np.arange(len(fill_factors))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for j, method in enumerate(SHAPING_METHODS):
        vals = [
            float(results[(shape, ff, method)]["energy_hist"][-1])
            for ff in fill_factors
        ]
        ax.bar(
            x + (j - 1) * width,
            vals,
            width,
            label=method,
            color=METHOD_COLORS[method],
        )
        for xi, v in zip(x + (j - 1) * width, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"FF={ff:.0%}" for ff in fill_factors])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Final fraction of conserved energy in target")
    ax.set_title(f"SLM pixelated shaping — final energy in {shape} target by method")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved energy bar: {out_path}")


@click.group()
def cli() -> None:
    """SLM 像素效应下的光束整形实验 (填充因子 x 算法)。"""


@cli.command()
@click.option(
    "--fill-factors",
    type=str,
    default="0.60,0.80,1.00",
    show_default=True,
    help="逗号分隔的填充因子列表 (0~1)。",
)
@click.option(
    "--shapes",
    type=str,
    default="square,triangle",
    show_default=True,
    help="逗号分隔的整形目标形状 (square / triangle)。",
)
@click.option(
    "--iterations",
    type=click.IntRange(10, 5000),
    default=600,
    show_default=True,
    help="每个 (目标, 填充因子, 算法) 组合的优化迭代数。",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=OUT_DIR,
    show_default=True,
    help="输出目录。",
)
def compare(fill_factors: str, shapes: str, iterations: int, out_dir: Path) -> None:
    """对比不同填充因子下三种算法 (SPGD / H-GD / Torch-GD) 的整形效果。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff_list = tuple(float(v) for v in fill_factors.split(","))
    shape_list = tuple(s.strip() for s in shapes.split(","))
    for ff in ff_list:
        if not 0.0 < ff <= 1.0:
            raise click.BadParameter("fill factors must be in (0, 1]")
    for s in shape_list:
        if s not in SHAPER_TARGETS:
            raise click.BadParameter(f"unknown shape {s!r}, use square|triangle")

    logger.info(
        f"=== SLM pixelated beam shaping | fill factors={ff_list} | "
        f"shapes={shape_list} | {iterations} iters ==="
    )

    # 共享湍流场 + DM (与 run_full.py 一致)。
    turb_phase = (
        generate_turbulence_phase(
            params.N,
            params.pixel_size,
            params.r0,
            params.target_phase_rms,
            params.seed_phase,
        )
        .numpy()
    )
    inf_funcs = generate_influence_functions()
    # 整形优化器接受 numpy/torch 皆可 (内部 to_torch), 统一转 numpy 保持契约。
    inf_flat = inf_funcs.reshape(params.n_act, -1).numpy()

    results: dict[tuple, dict] = {}
    summary_rows: list[dict] = []

    for shape in shape_list:
        target_np = SHAPER_TARGETS[shape].numpy()
        for ff in ff_list:
            with slm_shaping_forward(fill_factor=ff):
                for method in SHAPING_METHODS:
                    logger.info(f"Shaping → {shape} | FF={ff:.2f} | {method} ...")
                    res = _run_one_shaper(
                        method, shape, target_np, turb_phase, inf_flat, iterations
                    )
                    results[(shape, ff, method)] = res
                    summary_rows.append(
                        {
                            "shape": shape,
                            "fill_factor": f"{ff:.2f}",
                            "method": method,
                            "final_energy": f"{res['energy_hist'][-1]:.4f}",
                            "init_energy": f"{res['energy_hist'][0]:.4f}",
                            "delta_energy": f"{res['energy_hist'][-1] - res['energy_hist'][0]:.4f}",
                            "final_peak": f"{float(res['I_np'].max()):.4f}",
                            "final_loss": f"{res['loss_hist'][-1]:.4e}",
                        }
                    )
                    logger.info(
                        f"  energy-in-target: {res['energy_hist'][0]:.4f} → "
                        f"{res['energy_hist'][-1]:.4f}"
                    )

            # 每个 (目标, 填充因子) 的三方法收敛对比图。
            conv_path = out_dir / f"shaping_{shape}_ff{ff:.2f}_conv.png"
            plot_shaping_convergence_comparison(
                {
                    method: results[(shape, ff, method)]["energy_hist"]
                    for method in SHAPING_METHODS
                },
                f"{shape} · SLM FF={ff:.0%}",
                conv_path,
            )

        # 每个目标的最终光斑网格图 + 能量条形图。
        _plot_final_grid(
            results, shape, ff_list, out_dir / f"shaping_{shape}_final_grid.png",
            iterations,
        )
        _plot_energy_bar(results, shape, ff_list, out_dir / f"slm_shaping_{shape}_energy_bar.png")

    # 汇总表 (CSV + 终端打印)。
    csv_path = out_dir / "shaping_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Summary CSV saved: {csv_path}")

    print("\n===== SLM pixelated shaping summary (final energy-in-target) =====")
    print(f"{'shape':10s} {'FF':>5s} {'SPGD':>8s} {'H-GD':>8s} {'Torch-GD':>9s}")
    for shape in shape_list:
        for ff in ff_list:
            row = " ".join(
                f"{float(results[(shape, ff, m)]['energy_hist'][-1]):8.4f}"
                for m in SHAPING_METHODS
            )
            print(f"{shape:10s} {ff:5.2f} {row}")

    # 把图也复制到 docs/report/figures/ 供报告引用 (避免重复前缀)。
    report_fig_dir = Path(__file__).parent / "docs" / "report" / "figures"
    if report_fig_dir.is_dir():
        for png in out_dir.glob("*.png"):
            target = report_fig_dir / (
                png.name if png.name.startswith("slm_") else f"slm_{png.name}"
            )
            import shutil

            shutil.copy2(png, target)
            logger.info(f"Copied figure -> {target}")


if __name__ == "__main__":
    cli()