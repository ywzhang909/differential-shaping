# -*- coding: utf-8 -*-
"""
SLM (空间光调制器) 像素效应仿真模块。

本模块在既有 PyTorch 前向衍射模型基础上, 增加像素化相位型空间光调制器
(pixelated SLM) 的建模, 重点刻画 **填充因子 (fill factor)** 对远场衍射
结果的影响:

  - ``quantize_slm_phase``     : 相位量化 (如 8-bit / 256 级)。
  - ``slm_far_field_intensity``: 计入填充因子像素效应后的远场强度。
  - ``slm_fill_mask``          : 生成像素化的振幅掩模 (可视化)。
  - ``slm_field``              : 构造显式的像素化出瞳复振幅。

物理模型
--------
一个真实的像素化相位型 SLM 由周期排列的方形像素组成, 每个像素中心距为
``pitch``, 其中仅有边长为 ``a = pitch * sqrt(fill_factor)`` 的中心有效区对
入射光起作用, 其余为近似 0 透射的 "死区" (电极间隙 / 未调制区)。

为了**准确**刻画这一周期性结构对远场的影响 —— 包括:

  - 由死区遮挡导致的整体能量下降 (总透过约正比于 fill_factor);
  - 由像素周期导致的离散衍射级 (comb 结构);
  - 由有限有效区边长导致的 sinc 包络;

本模块在 **细网格 (亚像素) 分辨率** 上显式构造像素化振幅掩模, 再经
FFT 衍射到远场。 这样不同填充因子 (60% / 80% / 100% 乃至任意 0~1) 的
结构差异都能被平滑分辨, 而不受粗网格 (pitch=8 时一个像素仅 8 个粗网格点)
的量化误差影响。

实现约定与项目其余物理层一致: 全部为 PyTorch float32 (CPU)。
"""

from __future__ import annotations

import torch

from differential_shaping.params import (
    N,
    slm_fill_factor,
    slm_pixel_phase_steps,
    slm_pixel_pitch_px,
)

from .optics import remove_piston

__all__ = [
    "quantize_slm_phase",
    "slm_fill_mask",
    "slm_field",
    "slm_far_field_intensity",
    "slm_fill_factor",
]

_DEVICE = torch.device("cpu")
_DTYPE = torch.float32

# 每个仿真像素 (10 um) 的亚像素采样倍数。 用于分辨出 SLM 像素内的填充
# 因子死区结构 (pitch=8 sim-px -> 每个 SLM 像素 8*4=32 个细网格点, 足以分辨
# 60%/80%/100% 的填充因子差异)。
_SLM_UP = 4


def _fine_grid(M: int) -> tuple[torch.Tensor, torch.Tensor]:
    """返回细网格坐标 ``(M, M)`` 的 XX, YY (以仿真像素为单位)。"""
    xq = torch.arange(M, dtype=_DTYPE, device=_DEVICE) - M / 2.0
    xq = xq / _SLM_UP  # 转换回"仿真像素"单位 (每 _SLM_UP 个细网格点 = 1 仿真像素)
    XX, YY = torch.meshgrid(xq, xq, indexing="xy")
    return XX, YY


def _fine_slm_mask(
    fill_factor: float, pixel_pitch_px: float, M: int
) -> torch.Tensor:
    """在细网格 ``(M, M)`` 上构造像素化振幅掩模 (含死区)。

    每个 SLM 像素中心距为 ``pixel_pitch_px`` (仿真像素单位), 有效区半边长
    ``a_half = (pixel_pitch_px / 2) * sqrt(fill_factor)`` (仿真像素单位)。
    细网格坐标用 ``_SLM_UP`` 倍过采样, 因此填充因子的变化能被平滑分辨。
    """
    half_pitch = pixel_pitch_px / 2.0
    a_half = half_pitch * float(fill_factor) ** 0.5

    XX, YY = _fine_grid(M)

    px_center = torch.floor(XX / pixel_pitch_px + 0.5) * pixel_pitch_px
    py_center = torch.floor(YY / pixel_pitch_px + 0.5) * pixel_pitch_px

    active = (
        (torch.abs(XX - px_center) <= a_half)
        & (torch.abs(YY - py_center) <= a_half)
    ).to(dtype=_DTYPE)
    return active


def _fine_pupil(M: int) -> torch.Tensor:
    """细网格上的圆形孔径掩模 (半径 = N/2 仿真像素)。"""
    XX, YY = _fine_grid(M)
    rr = torch.sqrt(XX**2 + YY**2)
    return (rr <= N / 2.0).to(dtype=_DTYPE)


def _upsample_phase(phase: torch.Tensor, M: int) -> torch.Tensor:
    """把 ``(N, N)`` 连续相位最近邻升采样到 ``(M, M)`` 细网格。

    SLM 相位在 SLM 像素 / 仿真像素内近似恒定, 用最近邻按 ``_SLM_UP`` 倍
    重复即可 (保持"每个像素一个相位值"的离散特性, 不引入像素内平滑)。
    """
    K = _SLM_UP
    return phase.repeat_interleave(K, dim=0).repeat_interleave(K, dim=1)


# ---------------------------------------------------------------------------
# 相位量化
# ---------------------------------------------------------------------------
def quantize_slm_phase(
    phase: torch.Tensor, levels: int = slm_pixel_phase_steps
) -> torch.Tensor:
    """把连续相位量化到 SLM 可用的离散灰度台阶。

    真实 SLM 只能给出离散的相位台阶 (例如 8-bit 的 256 级)。 这里对
    ``[0, 2*pi)`` 区间的相位均匀量化。

    Parameters
    ----------
    phase : torch.Tensor
        连续相位 (rad), 任意形状。
    levels : int, optional
        量化级数 (默认 256)。

    Returns
    -------
    torch.Tensor
        量化后的相位 (rad, 位于 ``[0, 2*pi)`` 内)。
    """
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    phase_wrapped = torch.remainder(phase, 2 * torch.pi)
    idx = torch.round(phase_wrapped / (2 * torch.pi) * (levels - 1))
    idx = torch.clamp(idx, 0, levels - 1)
    return idx / (levels - 1) * (2 * torch.pi)


# ---------------------------------------------------------------------------
# 出瞳复振幅 与 远场强度 (细网格显式像素化)
# ---------------------------------------------------------------------------
def slm_field(
    phase: torch.Tensor,
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    quantize: bool = True,
) -> torch.Tensor:
    """构造经像素化 SLM 调制后的出瞳复振幅 (细网格 ``(N*_SLM_UP, N*_SLM_UP)``)。

    复振幅 = pupil_fine * slm_fill_mask_fine * exp(i * SLM_phase_fine)。
    细网格尺寸为 ``(N*_SLM_UP, N*_SLM_UP)``, 用于精确分辨填充因子死区。

    Parameters
    ----------
    phase : torch.Tensor
        SLM 加载的连续相位 ``(N, N)`` (rad)。
    fill_factor : float
        填充因子 (0, 1]。
    pixel_pitch_px : float, optional
        SLM 像素中心距 (仿真网格像素单位)。
    quantize : bool, optional
        是否先对相位做 SLM 量化 (默认 True)。

    Returns
    -------
    torch.Tensor
        复数出瞳场, 尺寸 ``(M, M)``, ``M = N * _SLM_UP``。
    """
    if not 0.0 < fill_factor <= 1.0:
        raise ValueError(f"fill_factor must be in (0, 1], got {fill_factor}")
    if pixel_pitch_px < 1.0:
        raise ValueError(f"pixel_pitch_px must be >= 1, got {pixel_pitch_px}")

    M = N * _SLM_UP
    phase_clean = remove_piston(phase)
    if quantize:
        phase_clean = quantize_slm_phase(phase_clean)

    mask_fine = _fine_slm_mask(fill_factor, pixel_pitch_px, M)
    pupil_fine = _fine_pupil(M)
    phase_up = _upsample_phase(phase_clean, M)

    return mask_fine * pupil_fine * torch.exp(1j * phase_up)


def slm_far_field_intensity(
    phase: torch.Tensor,
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    quantize: bool = True,
) -> torch.Tensor:
    """经像素化 SLM (含填充因子) 调制后的远场强度 ``|fftshift(fft2(E))|^2``。

    在细网格上显式构造像素化出瞳场后做 FFT 衍射, 同时包含以下效应:
      - 死区遮挡导致的整体能量下降 (正比于 fill_factor);
      - 像素周期引起的离散衍射级;
      - 有限有效区边长引起的 sinc 包络。

    数值约定: 细网格 FFT 幅度约正比于 ``1/Δ_fine²``, 比理想 128 网格前向
    (``1/Δ²``) 大 ``_SLM_UP²`` 倍, 故强度 (幅度平方) 需除以 ``_SLM_UP**4``
    (=256) 才能与 ``optics.far_field_intensity_metric`` (128 网格, 同一孔径)
    对齐 —— 零相位 / FF=1.0 时峰值即 (近似) 理想衍射极限峰值 ``I0_peak_t``,
    因此 Strehl 比/绝对强度与理想器件口径一致可比。

    Parameters
    ----------
    phase : torch.Tensor
        SLM 相位 ``(N, N)`` (rad)。
    fill_factor : float
        填充因子 (0, 1]。
    pixel_pitch_px : float, optional
        SLM 像素中心距 (仿真网格像素单位)。
    quantize : bool, optional
        是否先量化相位。

    Returns
    -------
    torch.Tensor
        远场强度, 尺寸 ``(M, M)``, ``M = N * _SLM_UP``。
    """
    E = slm_field(phase, fill_factor, pixel_pitch_px=pixel_pitch_px, quantize=quantize)
    return torch.abs(torch.fft.fftshift(torch.fft.fft2(E))) ** 2 / (_SLM_UP**4)


# ---------------------------------------------------------------------------
# 填充因子掩模 (可视化, 降采样到 (N, N))
# ---------------------------------------------------------------------------
def slm_fill_mask(
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    grid_N: int | None = None,
) -> torch.Tensor:
    """生成像素化 SLM 的填充因子振幅掩模 (可视化)。

    在细网格 (``(N*_SLM_UP, N*_SLM_UP)``) 上构造能分辨填充因子死区的像素
    掩模 (默认) —— 每个 SLM 像素显示为 ``pixel_pitch_px*_SLM_UP`` 见方的
    方块, 其中仅 ``sqrt(fill_factor)`` 比例的中心有效区为 1, 死区为 0,
    因此 60% / 80% / 100% 的差异清晰可见。 若给定 ``grid_N``, 则降采样到
    ``(grid_N, grid_N)`` (会丢失亚像素死区细节, 不推荐用于对比图)。

    返回值不参与远场计算 (远场直接用细网格 :func:`slm_far_field_intensity`),
    仅用于绘制像素结构示意图。

    Parameters
    ----------
    fill_factor : float
        填充因子, 0 < fill_factor <= 1。
    pixel_pitch_px : float, optional
        SLM 像素中心距 (仿真网格像素单位)。
    grid_N : int | None, optional
        若给定, 将掩模降采样到 ``(grid_N, grid_N)``; 默认 None 返回
        细网格 ``(N*_SLM_UP, N*_SLM_UP)``。

    Returns
    -------
    torch.Tensor
        float32 掩模, 有效像素内为 1, 间隙/死区为 0。
    """
    if not 0.0 < fill_factor <= 1.0:
        raise ValueError(f"fill_factor must be in (0, 1], got {fill_factor}")
    if pixel_pitch_px < 1.0:
        raise ValueError(f"pixel_pitch_px must be >= 1, got {pixel_pitch_px}")

    M = N * _SLM_UP
    pupil_fine = _fine_pupil(M)
    mask_fine = _fine_slm_mask(fill_factor, pixel_pitch_px, M) * pupil_fine
    if grid_N is None:
        return mask_fine
    # 降采样到 (grid_N, grid_N)。
    step = M // grid_N
    mask = mask_fine[::step, ::step]
    if mask.shape[0] != grid_N:
        mask = mask[:grid_N, :grid_N]
    return mask


# ---------------------------------------------------------------------------
# 便捷: 在给定相位上求多个填充因子对应的远场强度 (供可视化对比使用)。
# ---------------------------------------------------------------------------
def slm_far_field_intensity_many(
    phase: torch.Tensor,
    fill_factors: list[float],
    pixel_pitch_px: float = slm_pixel_pitch_px,
    quantize: bool = True,
) -> dict[float, torch.Tensor]:
    """对多个填充因子分别计算远场强度, 返回 {fill_factor: intensity} 字典。

    Parameters
    ----------
    phase : torch.Tensor
        SLM 相位 ``(N, N)`` (rad)。
    fill_factors : list[float]
        要仿真的填充因子列表。
    pixel_pitch_px : float, optional
        SLM 像素中心距 (仿真网格单位)。
    quantize : bool, optional
        是否先量化相位。

    Returns
    -------
    dict[float, torch.Tensor]
        ``{fill_factor: I_far}``。
    """
    return {
        ff: slm_far_field_intensity(
            phase, ff, pixel_pitch_px=pixel_pitch_px, quantize=quantize
        )
        for ff in fill_factors
    }


# 便捷别名 (与 optics.far_field_intensity_metric 对应)
def slm_far_field_intensity_metric(
    phase: torch.Tensor,
    fill_factor: float,
    pixel_pitch_px: float = slm_pixel_pitch_px,
    quantize: bool = True,
) -> torch.Tensor:
    """别名: 与 :func:`slm_far_field_intensity` 相同。"""
    return slm_far_field_intensity(
        phase, fill_factor, pixel_pitch_px=pixel_pitch_px, quantize=quantize
    )
