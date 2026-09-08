# -*- coding: utf-8 -*-
"""
SLM (空间光调制器) 像素效应仿真入口。

在既有自适应光学 / 光束整形平台基础上, 增加对像素化相位型 SLM 的建模,
重点仿真 **填充因子 (fill factor)** 对远场衍射光斑的影响。

用法::

    # 对比默认的 60% / 80% / 100% 三种填充因子, 并分别作图
    python run_slm.py compare

    # 仅对单个可调填充因子 (0~1) 仿真并保存单张远场图
    python run_slm.py single --fill-factor 0.75 --out out_slm_075.png

生成:
    - ``slm_fill_factor_compare.png``     三种填充因子的远场 dB 对比图
    - ``slm_fill_mask_compare.png``       三种填充因子的像素化掩模对比
    - ``slm_fill_factor_full.png``        掩模(上)+远场(下) 综合对比
    - (single 模式) 单张远场图
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

from differential_shaping import params
from differential_shaping.simulation import (
    generate_turbulence_phase,
    slm_far_field_intensity,
    slm_far_field_intensity_many,
)
from differential_shaping.simulation.optics import to_numpy as _to_np
from differential_shaping.visualization.slm_plots import (
    DEFAULT_FILL_FACTORS,
    plot_slm_fill_factor_comparison,
    plot_slm_fill_mask_comparison,
    plot_slm_full_comparison,
)

OUT_DIR = Path(__file__).parent / "slm_results"


def _default_phase() -> torch.Tensor:
    """生成一束用于 SLM 演示的相位 (可选: 高斯聚焦相位 或 湍流相位)。

    这里用一束理想化平面波 + 轻微像差相位来突出 SLM 像素效应本身, 而不是
    湍流校正场景: 便于观察不同填充因子造成的 sinc 包络差异。
    """
    # 用平面波 (零相位) 即可清晰展示像素化 sinc 包络; 也可叠加轻微湍流
    # 让光斑更有实感, 但会掩盖填充因子效应。 此处取零相位平面波。
    N = params.N
    return torch.zeros((N, N), dtype=torch.float32)


def _defocused_phase() -> torch.Tensor:
    """带离焦相位 (defocus) 的 SLM 相位, 展示非平面波下的填充因子效应。"""
    N = params.N
    x = torch.arange(N, dtype=torch.float32) - N / 2
    XX, YY = torch.meshgrid(x, x, indexing="xy")
    r2 = XX**2 + YY**2
    # 归一化离焦: 边缘相位约 ~2*pi。
    defocus = torch.pi / 2.0 * (2 * r2 / (N / 2) ** 2 - 1)
    return defocus


@click.group()
def cli() -> None:
    """SLM 像素效应 (填充因子) 仿真。"""


@cli.command()
@click.option(
    "--fill-factor",
    "-f",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    show_default=True,
    help="要仿真的单个填充因子 (0~1)。 留空则用默认 60%/80%/100% 三档。",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="输出图路径 (默认 slm_results/)。",
)
@click.option(
    "--defocus",
    is_flag=True,
    default=False,
    show_default=True,
    help="改用带离焦相位的 SLM 相位 (否则为平面波零相位)。",
)
def single(
    fill_factor: float | None,
    out: Path | None,
    defocus: bool,
) -> None:
    """对单个可调填充因子 (0~1) 仿真并保存远场图。

    输出默认写入 ``slm_results/slm_single_ff{ff:.0%}.png``。
    """
    ff = fill_factor if fill_factor is not None else params.slm_fill_factor
    if not 0.0 < ff <= 1.0:
        raise click.BadParameter(f"--fill-factor 需在 (0, 1] 内, 收到 {ff}")

    phase = _defocused_phase() if defocus else _default_phase()
    I = _to_np(slm_far_field_intensity(phase, fill_factor=ff))
    logger.info(f"single SLM simulation: fill_factor={ff:.0%}")

    out_path = out or (OUT_DIR / f"slm_single_ff{ff:.0%}.png")
    plot_slm_fill_factor_comparison({ff: I}, out_path, fill_factors=(ff,))
    logger.info(f"Saved: {out_path}")


@cli.command()
@click.option(
    "--fill-factors",
    "-f",
    type=str,
    default="0.60,0.80,1.00",
    show_default=True,
    help="逗号分隔的填充因子列表 (0~1), 例如 '0.60,0.80,1.00'。",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="输出目录 (默认 slm_results/)。",
)
@click.option(
    "--defocus",
    is_flag=True,
    default=False,
    show_default=True,
    help="改用带离焦相位的 SLM 相位 (否则为平面波零相位)。",
)
def compare(
    fill_factors: str,
    out_dir: Path | None,
    defocus: bool,
) -> None:
    """对比多个填充因子 (0~1) 下的远场衍射光斑。

    依次计算每个填充因子的远场强度, 并生成三张图:
      - ``slm_fill_factor_compare.png``  远场 dB 逐行对比
      - ``slm_fill_mask_compare.png``    像素化掩模逐列对比
      - ``slm_fill_factor_full.png``     掩模 + 远场 综合对比
    """
    ffs = tuple(float(x.strip()) for x in fill_factors.split(",") if x.strip())
    if not ffs or any(not 0.0 < ff <= 1.0 for ff in ffs):
        raise click.BadParameter(
            "--fill-factors 需为 (0,1] 内的逗号分隔数字, 例如 '0.60,0.80,1.00'"
        )

    phase = _defocused_phase() if defocus else _default_phase()
    intens = slm_far_field_intensity_many(phase, list(ffs))
    intens_np = {ff: _to_np(v) for ff, v in intens.items()}
    logger.info(f"compare SLM simulation: fill_factors={ffs}")

    odir = out_dir or OUT_DIR
    odir.mkdir(parents=True, exist_ok=True)

    # 1) 远场 dB 对比
    plot_slm_fill_factor_comparison(intens_np, odir / "slm_fill_factor_compare.png", ffs)
    # 2) 掩模对比
    plot_slm_fill_mask_comparison(ffs, odir / "slm_fill_mask_compare.png")
    # 3) 综合 (掩模上排 + 远场下排)
    plot_slm_full_comparison(intens_np, ffs, odir / "slm_fill_factor_full.png")

    for ff in ffs:
        logger.info(f"  FF={ff:.0%}: peak={float(intens_np[ff].max()):.3e} (arb. units)")
    logger.info(f"Saved comparison figures to: {odir}")


if __name__ == "__main__":
    cli()
