# -*- coding: utf-8 -*-
"""
Minimal Adam optimiser on JAX arrays.

A stateless, pure-function Adam step used by the backprop (Torch-GD) optimiser.
It is intentionally dependency-free: ``jax_adam_step`` is a pure function of
(parameters, first-moment, second-moment, step) -> updated tuple, so it can be
called inside a plain Python loop (or wrapped in ``jax.jit``) without any
optimizer-class state.  Matches ``torch.optim.Adam`` (betas 0.9/0.999,
eps 1e-8, with bias correction).
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["AdamState", "jax_adam_init", "jax_adam_step"]

_BETA1 = 0.9
_BETA2 = 0.999
_EPS = 1e-8


class AdamState:
    """First/second moment buffers plus the step counter for one parameter set."""

    def __init__(self, shape: tuple[int, ...]):
        m = jnp.zeros(shape, dtype=jnp.float32)
        v = jnp.zeros(shape, dtype=jnp.float32)
        self.m = m
        self.v = v
        self.t = 0

    def update(self, param: jax.Array, grad: jax.Array, lr: float):
        """Apply one Adam step in place; returns the updated parameter."""
        self.t += 1
        self.m = _BETA1 * self.m + (1.0 - _BETA1) * grad
        self.v = _BETA2 * self.v + (1.0 - _BETA2) * (grad * grad)
        m_hat = self.m / (1.0 - _BETA1**self.t)
        v_hat = self.v / (1.0 - _BETA2**self.t)
        return param - lr * m_hat / (jnp.sqrt(v_hat) + _EPS)


def jax_adam_init(shape: tuple[int, ...]) -> AdamState:
    """Create a zero-initialised Adam state for a parameter of the given shape."""
    return AdamState(shape)


def jax_adam_step(
    state: AdamState, param: jax.Array, grad: jax.Array, lr: float
) -> jax.Array:
    """One pure Adam step; returns the updated parameter (and updates state)."""
    return state.update(param, grad, lr)
