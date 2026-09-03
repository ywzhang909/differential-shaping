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
    spgd_shaping_optimization(...)        -> (u, dm_u, loss_hist, energy_hist, I_np)
    hgd_shaping_optimization(...)         -> (u, dm_u, loss_hist, energy_hist, I_np)

The first optimizer (``target_shaping_optimization``) differentiates the target
loss through the forward model with **autograd / backprop** (the differential
approach).  ``spgd_shaping_optimization`` and ``hgd_shaping_optimization``
estimate the gradient of the *same* target loss by finite-difference perturbation
(SPGD's two-sided random perturbation / H-GD's Hadamard patterns), so all three
methods optimise an identical objective and can be compared fairly.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from scipy.linalg import hadamard

from ..params import (
    N,
    n_act,
    delta_amp,
    alpha_spgd,
    alpha_hgd,
    seed_spgd,
    max_iter as _default_max_iter,
)
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
    "energy_in_target",
    "target_shaping_optimization",
    "spgd_shaping_optimization",
    "hgd_shaping_optimization",
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


def shaping_metric(
    phase: torch.Tensor,
    target_t: torch.Tensor,
    target_n: torch.Tensor,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar target-shaping loss + energy-in-target from a far-field phase.

    Shared by every shaping optimizer (backprop and finite-difference alike)
    so that Torch-GD / SPGD / H-GD optimise an *identical* objective.

    Parameters
    ----------
    phase : (N, N) torch complex or float far-field phase (piston-removed here).
    target_t : (N, N) non-negative target intensity.
    target_n : (N, N) normalised target (``target_t / sum``).

    Returns
    -------
    loss, energy, I_n : torch.Tensor
        ``loss = W_m * mean((I_n - T_n)^2) - W_c * energy``,
        ``energy = energy_in_target``, ``I_n = normalised focal intensity``.
    """
    I = far_field_intensity_metric(phase)
    I_n = I / (I.sum() + 1e-12)
    loss_match = torch.mean((I_n - target_n) ** 2)
    energy = (I_n * target_t).sum() / (target_t.sum() + 1e-12)
    loss = loss_weight_match * loss_match - loss_weight_capture * energy
    return loss, energy, I_n


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
    capture term biases energy into the compact target region.  This is the
    **backprop / differential** variant: both gradients are computed by autograd
    through the differentiable FFT forward model and applied with Adam to all
    ``n_act`` DM commands.

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
        loss, _, _ = shaping_metric(
            phase, target_t, target_n, loss_weight_match, loss_weight_capture
        )
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


# ---------------------------------------------------------------------------
# Numeric-gradient (finite-difference) shaping optimisers
# ---------------------------------------------------------------------------
def _numeric_shaping_step(
    turb_t: torch.Tensor,
    u: torch.Tensor,
    dm_u: torch.Tensor,
    delta_u: torch.Tensor,
    dm_delta: torch.Tensor,
    alpha: float,
    target_t: torch.Tensor,
    target_n: torch.Tensor,
    loss_weight_match: float,
    loss_weight_capture: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One two-sided perturbative update of the *shaping* loss.

    Estimates ``dJ/du`` by ``(J(u+δ) - J(u-δ))`` (the same estimator SPGD /
    H-GD use for the Strehl objective) but with ``J`` = the target-shaping loss,
    so the DM is steered to push conserved energy into the target shape.

    With ``dm_delta = DM(δ)`` the control update is scalar·δ, and the DM-surface
    update is the same scalar·dm_delta — exactly as in ``spgd.update_by_two_sided_perturbation``.
    """
    phase_p = turb_t + dm_u + dm_delta
    phase_m = turb_t + dm_u - dm_delta
    J_p, _, _ = shaping_metric(
        phase_p, target_t, target_n, loss_weight_match, loss_weight_capture
    )
    J_m, _, _ = shaping_metric(
        phase_m, target_t, target_n, loss_weight_match, loss_weight_capture
    )
    delta = torch.max(torch.abs(delta_u))
    scalar = -alpha * (J_p - J_m) / (2 * delta**2 + 1e-30)
    u = u + scalar * delta_u
    dm_u = dm_u + scalar * dm_delta
    return u, dm_u


def _shaping_update_loop(
    method: str,
    turb_phase,
    inf_flat,
    target_t: torch.Tensor,
    target_n: torch.Tensor,
    delta_patterns: torch.Tensor,
    dm_deltas: torch.Tensor,
    alpha: float,
    max_iter: int,
    loss_weight_match: float,
    loss_weight_capture: float,
    label: str,
    snapshot_indices,
    snapshot_store,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared inner loop for SPGD / H-GD shaping (numeric gradient).

    Operates entirely on float32 torch tensors in the hot loop (one
    ``shaping_metric`` per perturbed phase), matches the ``target_shaping_optimization``
    return contract ``(u, dm_u, loss_hist, energy_hist, I_np)``.
    """
    turb_t = to_torch(turb_phase)
    inf_flat_t = to_torch(inf_flat)
    inf_flat_T = inf_flat_t.t().contiguous()  # (N*N, n_act)

    u = torch.zeros(n_act, dtype=_DTYPE)
    dm_u = torch.zeros((N, N), dtype=_DTYPE)

    loss_hist = np.zeros(max_iter)
    energy_hist = np.zeros(max_iter)

    snapshot_iters = set(snapshot_indices) if snapshot_indices is not None else set()
    if snapshot_store is not None and 0 in snapshot_iters:
        snapshot_store.append((0, to_numpy(u).copy()))

    for it in range(max_iter):
        u, dm_u = _numeric_shaping_step(
            turb_t, u, dm_u, delta_patterns[it], dm_deltas[it], alpha,
            target_t, target_n, loss_weight_match, loss_weight_capture,
        )
        # Record current loss + energy under no_grad (metric only, not the
        # perturbative estimate) for comparison with the backprop run.
        with torch.no_grad():
            loss_now, energy, _ = shaping_metric(
                turb_t + dm_u, target_t, target_n,
                loss_weight_match, loss_weight_capture,
            )
            loss_hist[it] = to_numpy(loss_now)
            energy_hist[it] = to_numpy(energy)

        if snapshot_store is not None and (it + 1) in snapshot_iters:
            snapshot_store.append((it + 1, to_numpy(u).copy()))

        if it % 200 == 0 or it == max_iter - 1:
            logger.info(
                f"{method}-shaping[{label}] {it:4d}: "
                f"loss={loss_hist[it]:.4e}, energy-in-target={energy_hist[it]:.3f}"
            )

    with torch.no_grad():
        dm_u_final = dm_surface(u, inf_flat_T)
        _, _, I_n = shaping_metric(
            turb_t + dm_u_final, target_t, target_n,
            loss_weight_match, loss_weight_capture,
        )
        I_np = to_numpy(I_n)

    return to_numpy(u), to_numpy(dm_u_final), loss_hist, energy_hist, I_np


def _prepare_target(
    target: np.ndarray | torch.Tensor, label: str | None
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Normalise a target pattern -> (target_t, target_n, log_name)."""
    target_t = (
        to_torch(target)
        if isinstance(target, np.ndarray)
        else target.to(dtype=_DTYPE, device=_DEVICE)
    )
    target_t = torch.clamp(target_t, min=0.0)
    target_n = target_t / (target_t.sum() + 1e-12)
    return target_t, target_n, label or _shape_name(target_t)


def spgd_shaping_optimization(
    turb_phase,
    inf_flat,
    target: np.ndarray | torch.Tensor,
    max_iter: int = _default_max_iter,
    alpha: float = alpha_spgd,
    seed: int = seed_spgd,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
    label: str | None = None,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the far-field spot via SPGD (stochastic two-sided perturbation).

    Identical target-shaping objective to ``target_shaping_optimization`` but the
    gradient is estimated by random two-sided perturbation of the n_act DM
    commands (not autograd), matching the classic SPGD scheme.  Returns the same
    ``(u, dm_u, loss_hist, energy_hist, I_np)`` contract.

    ``seed`` is accepted for API parity but the perturbation patterns are driven
    by the module-level ``seed_spgd`` (the same RNG as the Strehl-correction loop).
    """
    inf_flat_t = to_torch(inf_flat)

    target_t, target_n, log_name = _prepare_target(target, label)

    # Precompute SPGD perturbation patterns (same RNG as the correction loop).
    gen = torch.Generator()
    gen.manual_seed(seed_spgd)
    patterns = (
        2
        * torch.randint(0, 2, (max_iter, n_act), generator=gen).to(dtype=_DTYPE)
        - 1
    )
    delta_patterns = patterns * delta_amp
    dm_deltas = (delta_patterns @ inf_flat_t).reshape(max_iter, N, N)

    return _shaping_update_loop(
        "SPGD", turb_phase, inf_flat, target_t, target_n,
        delta_patterns, dm_deltas, alpha, max_iter,
        loss_weight_match, loss_weight_capture, log_name,
        snapshot_indices, snapshot_store,
    )


def hgd_shaping_optimization(
    turb_phase,
    inf_flat,
    target: np.ndarray | torch.Tensor,
    max_iter: int = _default_max_iter,
    alpha: float = alpha_hgd,
    loss_weight_match: float = 1.0,
    loss_weight_capture: float = 0.15,
    label: str | None = None,
    snapshot_indices: list[int] | None = None,
    snapshot_store: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the far-field spot via H-GD (Hadamard perturbation patterns).

    Same target-shaping objective and two-sided estimating as SPGD but the
    perturbation patterns are orthogonal Hadamard rows (cycled), matching the
    H-GD correction loop.  Returns the same 5-tuple contract.
    """
    inf_flat_t = to_torch(inf_flat)

    target_t, target_n, log_name = _prepare_target(target, label)

    H = hadamard(n_act, dtype=np.float64)
    patterns = H[:, 1:].T.copy()  # skip the all-one (piston-like) column
    patterns_t = torch.from_numpy(patterns.astype(np.float32))
    delta_patterns = patterns_t * delta_amp
    dm_deltas = (delta_patterns @ inf_flat_t).reshape(n_act - 1, N, N)

    # Cycle through the (n_act-1) Hadamard patterns like the H-GD correction loop.
    if max_iter <= n_act - 1:
        delta_patterns_used = delta_patterns[:max_iter]
        dm_deltas_used = dm_deltas[:max_iter]
    else:
        reps = (max_iter + n_act - 2) // (n_act - 1)
        delta_patterns_used = delta_patterns.repeat(reps, 1)[:max_iter]
        dm_deltas_used = dm_deltas.repeat(reps, 1, 1)[:max_iter]

    return _shaping_update_loop(
        "H-GD", turb_phase, inf_flat, target_t, target_n,
        delta_patterns_used, dm_deltas_used, alpha, max_iter,
        loss_weight_match, loss_weight_capture, log_name,
        snapshot_indices, snapshot_store,
    )