# -*- coding: utf-8 -*-
"""
SLM 像素效应前向下的光束整形粘合层。

三个整形优化器 (:mod:`differential_shaping.optimization.shaping`) 内部通过模块级名字
``shaping.far_field_intensity_metric`` 调用前向模型。本模块提供把该模块级名字临时
替换为**像素化 SLM 前向** 的上下文管理器, 复用同一套目标损失与优化器做
"不同填充因子下的整形" 实验:

  - ``slm_shaping_forward(fill_factor)``   上下文: 期间所有整形调用走 SLM 前向;
  - ``slm_forward_cropped(fill_factor)``   返回 (N, N) 裁剪后的 SLM 前向可调用对象。

SLM 前向 (:func:`slm_far_field_intensity`) 返回细网格 ``(N*_SLM_UP, N*_SLM_UP)``
强度; 这里裁剪到中央 ``(N, N)`` 窗口 —— 与理想前向完全相同的 ±64 λf/D 视场 ——
使既有损失接口 (目标掩模为 (N, N)) 无需改动。

相位量化 (``quantize=False`` 默认关闭): 量化用到不可微的 ``round``, 反向传播
(Torch-GD) 需要关闭量化以保持 autograd 图连通; 且理想前向本身也不做量化,
关闭量化保证三种算法对比公平。
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from differential_shaping import params
from differential_shaping.optimization import shaping as shaping_mod
from differential_shaping.simulation.slm import slm_far_field_intensity

__all__ = ["slm_forward_cropped", "slm_shaping_forward"]


def slm_forward_cropped(
    fill_factor: float, quantize: bool = False
):
    """构造裁剪到中央 (N, N) 窗口的 SLM 像素化前向。

    Parameters
    ----------
    fill_factor : float
        SLM 填充因子 (0, 1]。
    quantize : bool, optional
        是否先做 SLM 相位量化 (默认 False, 保持可微与公平对比)。

    Returns
    -------
    callable
        ``forward(phase: (N, N) torch.Tensor) -> (N, N) torch.Tensor`` 强度。
    """
    n = params.N

    def forward(phase: torch.Tensor) -> torch.Tensor:
        intensity = slm_far_field_intensity(
            phase, fill_factor=fill_factor, quantize=quantize
        )
        off = (intensity.shape[-1] - n) // 2
        return intensity[..., off : off + n, off : off + n]

    return forward


@contextmanager
def slm_shaping_forward(fill_factor: float, quantize: bool = False):
    """临时把 ``shaping.far_field_intensity_metric`` 换成 SLM 像素化前向。

    ``shaping_metric`` 以及三个整形优化器在调用时按模块级名字查找前向, 因此
    在 ``with`` 块内的所有整形优化 (SPGD / H-GD / Torch-GD) 自动切换到同一
    SLM 前向; 退出时恢复原始理想前向。
    """
    original = shaping_mod.far_field_intensity_metric
    shaping_mod.far_field_intensity_metric = slm_forward_cropped(
        fill_factor, quantize=quantize
    )
    try:
        yield
    finally:
        shaping_mod.far_field_intensity_metric = original