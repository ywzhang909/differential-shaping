# -*- coding: utf-8 -*-
"""
Phase-control device abstraction for the AO pipeline (run.py ``--device``).

The original pipeline assumes a **deformable mirror** (DM) in front of an
*ideal* forward model: the DM imprints a continuous phase ``inf_flat^T @ u``,
and the focal-plane intensity is computed with the ideal Fraunhofer propagator
``optics.far_field_intensity_metric``.

This module generalises that single device into selectable device models so the
same optimisation pipeline (SPGD / H-GD / Torch-GD / GS, point correction and
beam shaping alike) can be run against:

  ``dm``
      Deformable mirror + ideal continuous phase (the classic model).
  ``slm``
      Pixelated SLM: the phase is sampled on a coarse ``slm_pixel_pitch_px``
      grid (zero-order-hold up-sampled) and attenuated by the pixel fill factor
      (see ``simulation.slm``).  Optionally 8-bit phase quantization.
  ``ideal``
      A perfect per-pixel phase device: any ``(N, N)`` phase can be imprinted
      exactly.  This is the theoretical upper bound -- point correction becomes
      analytic (conjugate the turbulence -> flat phase), and beam shaping
      optimises the per-pixel phase directly (``target_shaping_optimization``
      with ``direct_phase=True``, Torch-GD only).

Mechanism
---------
The optimisers import ``compute_metrics`` and ``far_field_intensity_metric``
*by name* from ``simulation.optics`` at module load time and resolve them at
call time inside their hot loops.  :func:`device_scope` therefore swaps those
names in each optimizer module's namespace for the duration of the context, so
every objective evaluation -- perturbative (SPGD / H-GD), projection-logging
(GS) and autograd (Torch-GD) -- flows *through* the selected device forward.
For ``dm`` / ``ideal`` the forwards are already ideal, so the scope is a no-op.

The forward models are differentiable (the SLM one with ``quantize=False``, the
default), so Torch-GD backpropagates through the pixelation automatically.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch

from differential_shaping import params
from differential_shaping.simulation.optics import (
    compute_metrics as _ideal_compute_metrics,
    far_field_intensity_metric as _ideal_forward,
)
from differential_shaping.simulation.pupil import I0_peak_t, xx_metric_t, yy_metric_t

__all__ = [
    "DEVICE_CHOICES",
    "device_forward",
    "device_compute_metrics",
    "device_scope",
]

DEVICE_CHOICES = ("dm", "slm", "ideal")

# A device forward: (N, N) total phase -> (N, N) focal-plane intensity (torch).
Forward = Callable[[torch.Tensor], torch.Tensor]


def _slm_forward(fill_factor: float | None, quantize: bool) -> Forward:
    """Build the cropped SLM forward, importing lazily to keep layering clean.

    ``simulation`` must never import ``optimization`` at module load; the SLM
    *physics* lives in ``simulation.slm``, and ``slm_forward_cropped`` (the
    optimizer-friendly ``(N, N) -> (N, N)`` wrapper) lives in this package.
    """
    from differential_shaping.optimization.slm_shaping import slm_forward_cropped

    ff = params.slm_fill_factor if fill_factor is None else float(fill_factor)
    return slm_forward_cropped(fill_factor=ff, quantize=quantize)


def device_forward(
    device: str = "dm",
    fill_factor: float | None = None,
    quantize: bool = False,
) -> Forward:
    """Return the differentiable far-field forward for the requested device.

    Args:
        device: one of ``DEVICE_CHOICES``.
        fill_factor: SLM pixel fill factor in ``(0, 1]`` (only used for ``slm``;
            ``None`` -> ``params.slm_fill_factor``).
        quantize: quantize the SLM phase to 8-bit levels (``slm`` only).  Off by
            default so the forward stays differentiable for Torch-GD.

    Returns:
        ``phase (N, N) -> intensity (N, N)`` torch tensor forward model.
    """
    if device == "slm":
        return _slm_forward(fill_factor, quantize)
    if device in ("dm", "ideal"):
        return _ideal_forward
    raise ValueError(f"unknown device: {device!r} (expected one of {DEVICE_CHOICES})")


def device_compute_metrics(
    forward: Forward,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Wrap a device forward into a ``compute_metrics``-compatible evaluator.

    ``(phase) -> (J, SR, intensity)`` with the exact ``optics.compute_metrics``
    definitions -- J = centroid mean radius (pixels), SR = peak intensity /
    diffraction-limited peak ``I0_peak_t`` -- but evaluated *through* ``forward``,
    so SPGD / H-GD / GS / Torch-GD metrics and run.py reporting stay
    device-consistent.
    """

    def _cm(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intensity = forward(phase)
        total = intensity.sum() + 1e-30
        cx = (xx_metric_t * intensity).sum() / total
        cy = (yy_metric_t * intensity).sum() / total
        r_pix = torch.sqrt((xx_metric_t - cx) ** 2 + (yy_metric_t - cy) ** 2)
        J_mr = (r_pix * intensity).sum() / total
        SR = intensity.max() / (I0_peak_t + 1e-30)
        return J_mr, SR, intensity

    return _cm


@contextmanager
def device_scope(
    device: str = "dm",
    fill_factor: float | None = None,
    quantize: bool = False,
):
    """Evaluate optimizer objectives *through* the selected device forward.

    Patches the module-level metric names the optimizers resolve at call time
    (``spgd.compute_metrics``, ``hgd.compute_metrics``,
    ``torch_gd.compute_metrics`` / ``torch_gd.far_field_intensity_metric``,
    ``gs.compute_metrics`` / ``gs.far_field_intensity_metric`` and
    ``shaping.far_field_intensity_metric``) for the duration of the context and
    restores them on exit.  For ``dm`` / ``ideal`` it is a no-op (their forwards
    are already the ideal one).

    Example::

        with device_scope("slm", fill_factor=0.8):
            u, dm_u, J_hist, SR_hist = torch_gd_optimization(turb, inf_flat, ...)
    """
    if device not in DEVICE_CHOICES:
        raise ValueError(
            f"unknown device: {device!r} (expected one of {DEVICE_CHOICES})"
        )
    if device != "slm":
        # dm / ideal: optimizers already use the ideal forward model.
        yield
        return

    forward = device_forward(device, fill_factor, quantize)
    cm = device_compute_metrics(forward)

    import differential_shaping.optimization.gs as _gs
    import differential_shaping.optimization.hgd as _hgd
    import differential_shaping.optimization.shaping as _shaping
    import differential_shaping.optimization.spgd as _spgd
    import differential_shaping.optimization.torch_gd as _torch_gd

    swaps: list[tuple[object, str, object]] = [
        (_spgd, "compute_metrics", cm),
        (_hgd, "compute_metrics", cm),
        (_torch_gd, "compute_metrics", cm),
        (_torch_gd, "far_field_intensity_metric", forward),
        (_gs, "compute_metrics", cm),
        (_gs, "far_field_intensity_metric", forward),
        (_shaping, "far_field_intensity_metric", forward),
    ]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in swaps]
    try:
        for mod, name, value in swaps:
            setattr(mod, name, value)
        yield
    finally:
        for mod, name, old in saved:
            setattr(mod, name, old)