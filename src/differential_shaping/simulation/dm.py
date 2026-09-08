"""
Deformable mirror actuator grid and Gaussian influence functions.

Provides generate_actuator_grid and generate_influence_functions used
by the optimization modules to build DM surface shapes.

Converted to canonical PyTorch float32 (CPU) for the simulation physics layer.
"""

import torch

from differential_shaping.params import (
    D,
    N,
    act_spacing,
    n_act,
    n_act_x,
    n_act_y,
    sigma_inf,
)

from .pupil import XX_t, YY_t, pupil_float_t, pupil_mask

_DEVICE = torch.device("cpu")
_DTYPE = torch.float32


def generate_actuator_grid() -> tuple[torch.Tensor, torch.Tensor]:
    margin = act_spacing / 2
    xs = torch.linspace(
        -D / 2 + margin, D / 2 - margin, n_act_x, dtype=_DTYPE, device=_DEVICE
    )
    ys = torch.linspace(
        -D / 2 + margin, D / 2 - margin, n_act_y, dtype=_DTYPE, device=_DEVICE
    )
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    return X.ravel(), Y.ravel()


def generate_influence_functions() -> torch.Tensor:
    act_x, act_y = generate_actuator_grid()
    inf = torch.zeros((n_act, N, N), dtype=_DTYPE, device=_DEVICE)
    for i, (x0, y0) in enumerate(zip(act_x, act_y)):
        r2 = (XX_t - x0) ** 2 + (YY_t - y0) ** 2
        z = torch.exp(-r2 / (2 * sigma_inf**2)) * pupil_float_t
        z[pupil_mask] -= z[pupil_mask].mean()
        z[~pupil_mask] = 0.0
        z /= z.abs().max() + 1e-15
        inf[i] = z
    return inf
