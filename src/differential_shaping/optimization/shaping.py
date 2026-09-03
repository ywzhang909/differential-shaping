# -*- coding: utf-8 -*-
"""
Differential (backpropagation-driven) far-field beam shaping.

While ``torch_gd.py`` maximises on-axis energy (the Strehl objective), this
module turns the *target loss itself* into the differentiator: a user-supplied
far-field target intensity pattern (e.g. a square or a triangle) is compared to
the simulated focal-plane intensity, and the PyTorch autograd graph backpropagates
the resulting loss directly to the DM actuator commands ``u``.

Physically this relies on the **conservation of total intensity**: the far-field
intensity ``|FFT(pupil * exp(i*theta))|^2`` has a fixed total energy independent
of the DM phase (Parseval / the pupil magnitude ``|exp(i*theta)| = 1``).  Energy
can therefore only be *redistributed* into the target region, not created or
destroyed — which is exactly what the loss below drives.

Public API
----------
    make_square_target(half_width)        -> (N, N) torch float32 binary mask
    make_triangle_target(size)            -> (N, N) torch float32 binary mask
    target_shaping_optimization(...)      -> (u, dm_u, loss_hist, energy_hist, I_np)
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from ..params import N, n_act, max_iter as _default_max_iter
from ..simulation.optics import (
    to_torch,
    to_numpy,
    dm_surface,
    far_field_intensity_metric,
)
from ..simulation.pupil import pupil_float_t

__all__ = [
    "make_square_target",
    "make_triangle_target",
    "target_shaping_optimization",
    "energy_in_target",
]

_DEVICE = torch.device("cpu")
_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Target pattern generators (focal-plane intensity targets)
# ---------------------------------------------------------------------------
def make_square_target(half_width: int = 6) -> torch.Tensor:
    """Binary square target centred on the focal plane.

    A filled square of (2*half_width+1) x (2*half_width+1) pixels.  On the
    unpadded FFT grid one pixel is one ``lambda*f/D`` unit, so ``half_width``
    is in Airy-core scale units (the Airy core radius is ~2.44 pixels).

    Returns:
        (N, N) float32 tensor, 1 inside the square and 0 outside.
    """
    c = torch.arange(N, dtype=_DTYPE, device=_DEVICE)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    mask = (torch.abs(xx - N // 2) <= half_width) & (
        torch.abs(yy - N // 2) <= half_width
    )
    return mask.to(dtype=_DTYPE)


def make_triangle_target(
    size: int = 11, apex: str = "up"
) -> torch.Tensor:
    """Binary isosceles triangle target centred on the focal plane.

    ``size`` is the triangle height and base length (base half-width is
    ``size/2``).  Each pixel is selected by a linear half-width profile that
    goes to zero at the apex and reaches ``size/2`` at the base, so the result
    is an exact filled triangle (not a rectangle).

    Returns:
        (N, N) float32 tensor, 1 inside the triangle and 0 outside.
    """
    c = torch.arange(N, dtype=_DTYPE, device=_DEVICE)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    x0, y0 = N // 2, N // 2
    h = size  # triangle height
    b2 = size / 2.0  # base half-width
    top = y0 - h / 2.0
    bot = y0 + h / 2.0

    if apex == "up":
        # narrow at the top (apex), wide at the bottom (base)
        margin = b2 * (yy - top) / h
    elif apex == "down":
        # wide at the top (base), narrow at the bottom (apex)
        margin = b2 * (bot - yy) / h
    elif apex == "right":
        margin = b2 * (xx - (x0 - h / 2.0)) / h
    elif apex == "left":
        margin = b2 * ((x0 + h / 2.0) - xx) / h
    else:
        raise ValueError(f"Unknown apex: {apex!r}")

    if apex in ("up", "down"):
        inside = (yy >= top) & (yy <= bot) & (torch.abs(xx - x0) <= margin)
    else:
        inside = (xx >= x0 - h / 2.0) & (xx <= x0 + h / 2.0) & (
            torch.abs(yy - y0) <= margin
        )
    return inside.to(dtype=_DTYPE)


# ---------------------------------------------------------------------------
# Shape-matching metric (conservation-respecting)
# ---------------------------------------------------------------------------
def energy_in_target(I: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of conserved far-field energy captured inside the target.

    ``(I * target).sum() / I.sum()`` — since ``I.sum()`` is invariant under the
    DM phase this directly reports how much of the *fixed* energy has been
    pushed into the desired shape (1.0 = perfect, all energy in target).
    """
    return (I * target).sum() / (I.sum() + 1e-12)


# ---------------------------------------------------------------------------
# Differential target-loss DM shaping optimisation
# ---------------------------------------------------------------------------
def target_shaping_optimization(
    turb_phase: np.ndarray,
    inf_flat: np.ndarray,
    target: np.ndarray | torch.Tensor,
    max_iter: int = _default_max_iter,
    lr: float = 0.02,
    seed: int = 42,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
    label: str | None = None,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the far-field spot into ``target`` via backprop on the target loss.

    Minimises a conservation-respecting loss against the target intensity:

        I_n   = I  / sum(I)              # normalised (respects conserved total)
        T_n   = T  / sum(T)
        loss  = W_m * mean((I_n - T_n)^2)  +  W_c * (-energy_in_target)

    The MSE term drives the *shape* of the distribution toward the target; the
    capture term biases energy into the compact target region.  Both gradients
    are computed by autograd through the differentiable FFT forward model and
    applied with Adam to all ``n_act`` DM commands.

    Args:
        turb_phase: (N, N) float64 turbulence phase (rad).
        inf_flat:   (n_act, N*N) flattened DM influence functions.
        target:     (N, N) non-negative target intensity pattern (numpy or torch).
        max_iter:   number of Adam steps.
        lr:         Adam learning rate.
        seed:       RNG seed for reproducibility.
        loss_weight_match / loss_weight_capture: blend of the two loss terms.
        label: optional human-readable name for logging (e.g. "square"/"triangle").
        snapshot_indices: iteration indices at which to record ``(iter, u)``.
        snapshot_store:   mutable list appended with ``(iter, u)`` snapshots.

    Returns:
        (u, dm_u, loss_hist, energy_hist, I_np):
            u          (n_act,)  float64 final DM commands.
            dm_u       (N, N)    float64 final DM surface.
            loss_hist  (max_iter,) float64 backprop training loss history.
            energy_hist (max_iter,) float64 energy-in-target fraction history.
            I_np       (N, N)    float64 final normalised far-field intensity.
    """
    torch.manual_seed(seed)

    turb_t = to_torch(turb_phase)
    inf_flat_t = to_torch(inf_flat)
    inf_flat_T = inf_flat_t.t().contiguous()  # (N*N, n_act)

    target_t = (
        to_torch(target)
        if isinstance(target, np.ndarray)
        else target.to(dtype=_DTYPE, device=_DEVICE)
    )
    target_t = torch.clamp(target_t, min=0.0)
    target_n = target_t / (target_t.sum() + 1e-12)
    log_name = label or _shape_name(target_t)

    u = torch.zeros(n_act, requires_grad=True, device=pupil_float_t.device)
    optimizer = torch.optim.Adam([u], lr=lr)

    loss_hist = np.zeros(max_iter)
    energy_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, to_numpy(u).copy()))

    for it in range(max_iter):
        optimizer.zero_grad()
        dm_u = dm_surface(u, inf_flat_T)
        phase = turb_t + dm_u
        I = far_field_intensity_metric(phase)     # differentiable forward
        I_n = I / (I.sum() + 1e-12)

        # Differentiable target loss (train graph).
        loss_match = torch.mean((I_n - target_n) ** 2)
        energy = (I_n * target_t).sum() / (target_t.sum() + 1e-12)
        loss = loss_weight_match * loss_match - loss_weight_capture * energy
        loss.backward()
        optimizer.step()

        # Logging under no_grad (not part of the training graph).
        with torch.no_grad():
            phase_eval = turb_t + dm_surface(u.detach(), inf_flat_T)
            I_eval = far_field_intensity_metric(phase_eval)
            loss_hist[it] = to_numpy(loss)
            energy_hist[it] = to_numpy(energy_in_target(I_eval, target_t))

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(u).copy()))

        if it % 200 == 0 or it == max_iter - 1:
            logger.info(
                f"Shaping[{log_name}] {it:4d}: loss={loss_hist[it]:.4e}, "
                f"energy-in-target={energy_hist[it]:.3f}"
            )

    with torch.no_grad():
        dm_u_final = dm_surface(u.detach(), inf_flat_T)
        I_final = far_field_intensity_metric(turb_t + dm_u_final)

    u_np = to_numpy(u)
    dm_np = to_numpy(dm_u_final)
    I_np = to_numpy(I_final / (I_final.sum() + 1e-12))

    return u_np, dm_np, loss_hist, energy_hist, I_np


def _shape_name(target: torch.Tensor) -> str:
    """Small heuristic label for logging (square / triangle / other)."""
    n = int(target.sum())
    side = int(round(np.sqrt(n)))
    if n == side * side:
        return f"square-{side}x{side}"
    return f"target-{n}px"