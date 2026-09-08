# -*- coding: utf-8 -*-
"""
SLM 像素效应对比可视化。 生成填充因子 (fill factor) 对比图, 展示 60% / 80% /
100% 填充因子下远场衍射光斑的差异, 以及对应的像素化振幅掩模。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

from ..params import pad_factor_show, spot_half_width_lamD
from ..simulation.optics import to_numpy as _to_np
from ..simulation.slm import slm_fill_mask

# 中文字体 (Windows 常见 CJK 字体), 避免 matplotlib 缺字警告; 找不到则退回默认。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 填充因子对比使用的默认值 (60%, 80%, 100%)。
DEFAULT_FILL_FACTORS = (0.60, 0.80, 1.00)


def _crop_center_db(img: np.ndarray, half_width_pix: int) -> np.ndarray:
    """以 dB 显示裁剪中心区域 (numpy 实现, 复用 focal 的裁剪语义)。

    与 ``optics.crop_center`` 一致: 显示窗口为 ``+/- half_width_lamD * pad_factor``
    个 lambda*f/D 单元。
    """
    M = img.shape[0]
    c = M // 2
    a = max(0, c - half_width_pix)
    b = min(M, c + half_width_pix + 1)
    return img[a:b, a:b]


def plot_slm_fill_factor_comparison(
    intensities: dict[float, np.ndarray],
    out_path: Path,
    fill_factors: tuple[float, ...] | None = None,
) -> None:
    """绘制不同填充因子下的远场强度对比图。

    Parameters
    ----------
    intensities : dict[float, np.ndarray]
         ``{fill_factor: far_field_intensity_ndarray}``, 各为 ``(N, N)``。
    out_path : Path
        输出 PNG 路径。
    fill_factors : tuple[float, ...] | None, optional
        实际使用的填充因子列表 (用于图题)。 默认取 ``intensities`` 的键。
    """
    ffs = list(fill_factors or intensities.keys())
    if not ffs:
        raise ValueError("intensities must contain at least one fill factor")
    n = len(ffs)

    # 统一归一化到整个对比组的全局峰值, 以便公平比较不同填充因子的亮度。
    global_peak = max(float(np.max(v)) for v in intensities.values())

    half_pix = round(spot_half_width_lamD * pad_factor_show)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6.4, 3.4 * n),
        squeeze=False,
    )

    im: AxesImage | None = None
    for row, ff in enumerate(ffs):
        ax = axes[row, 0]
        I = intensities[ff]
        I_rel = I / (global_peak + 1e-30)
        I_db_full = 10 * np.log10(np.maximum(I_rel, 1e-8))
        I_db = _crop_center_db(I_db_full, half_pix)
        im = ax.imshow(
            I_db,
            origin="lower",
            cmap="jet",
            vmin=-40,
            vmax=0,
            interpolation="nearest",
        )
        ax.set_title(f"FF = {ff:.0%} — 远场强度 (dB)")
        ax.set_xlabel(r"$x/(\lambda f/D)$")
        ax.set_ylabel(r"$y/(\lambda f/D)$")
        ax.set_aspect("equal")

    assert im is not None
    fig.colorbar(
        im, ax=axes[:, 0], fraction=0.046, pad=0.04, label="Intensity / global peak (dB)"
    )
    fig.suptitle(
        "SLM 填充因子对远场衍射的影响 (60% / 80% / 100%)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_slm_fill_mask_comparison(
    fill_factors: tuple[float, ...] = DEFAULT_FILL_FACTORS,
    out_path: Path | None = None,
) -> dict[float, np.ndarray]:
    """绘制并返回多个填充因子对应的像素化掩模。

    Parameters
    ----------
    fill_factors : tuple[float, ...]
        要显示的填充因子 (默认 60% / 80% / 100%)。
    out_path : Path | None, optional
        若给定, 将掩模对比图保存到该路径。

    Returns
    -------
    dict[float, np.ndarray]
        ``{ff: mask_ndarray}`` (numpy 数组)。
    """
    masks = {ff: _to_np(slm_fill_mask(ff)) for ff in fill_factors}

    if out_path is not None:
        fig, axes = plt.subplots(
            1, len(fill_factors), figsize=(4.2 * len(fill_factors), 4), squeeze=False
        )
        for ax, (ff, mask) in zip(axes[0], masks.items()):
            ax.imshow(mask, origin="lower", cmap="gray", interpolation="nearest")
            ax.set_title(f"FF = {ff:.0%}")
            ax.set_xlabel("pixel")
            ax.set_ylabel("pixel")
            ax.set_aspect("equal")
        fig.suptitle("SLM 像素化填充因子掩模", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return masks


# 便捷组合: 掩模 + 远场一张图展示。
def plot_slm_full_comparison(
    intensities: dict[float, np.ndarray],
    fill_factors: tuple[float, ...],
    out_path: Path,
) -> None:
    """上排: 各填充因子像素掩模; 下排: 对应远场 dB 光斑。

    Parameters
    ----------
    intensities : dict[float, np.ndarray]
        ``{ff: I_far}``。
    fill_factors : tuple[float, ...]
        要展示的填充因子 (应与 ``intensities`` 的键一致)。
    out_path : Path
        输出 PNG 路径。
    """
    ffs = list(fill_factors)
    if not ffs:
        raise ValueError("fill_factors must contain at least one fill factor")
    n = len(ffs)
    global_peak = max(float(np.max(v)) for v in intensities.values())
    half_pix = round(spot_half_width_lamD * pad_factor_show)

    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), squeeze=False)

    im: AxesImage | None = None
    for col, ff in enumerate(ffs):
        # 上排: 掩模
        mask = _to_np(slm_fill_mask(ff))
        axm = axes[0, col]
        axm.imshow(mask, origin="lower", cmap="gray", interpolation="nearest")
        axm.set_title(f"FF = {ff:.0%} 像素掩模")
        axm.set_xlabel("pixel")
        axm.set_ylabel("pixel")
        axm.set_aspect("equal")

        # 下排: 远场 dB
        I = intensities[ff]
        I_rel = I / (global_peak + 1e-30)
        I_db = _crop_center_db(10 * np.log10(np.maximum(I_rel, 1e-8)), half_pix)
        ax = axes[1, col]
        im = ax.imshow(
            I_db, origin="lower", cmap="jet", vmin=-40, vmax=0, interpolation="nearest"
        )
        ax.set_title(f"FF = {ff:.0%} 远场 (dB)")
        ax.set_xlabel(r"$x/(\lambda f/D)$")
        ax.set_ylabel(r"$y/(\lambda f/D)$")
        ax.set_aspect("equal")

    assert im is not None
    fig.colorbar(
        im,
        ax=axes[1, :],
        fraction=0.046,
        pad=0.04,
        label="Intensity / global peak (dB)",
    )
    fig.suptitle("SLM 填充因子对远场衍射的影响", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
