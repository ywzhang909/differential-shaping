# -*- coding: utf-8 -*-
"""
JAX / chromatix backend.

GPU-accelerated re-implementation of the simulation physics and the AO
optimisers on JAX arrays, built on the ``chromatix`` functional layer
(``fx.fft`` / ``fx.generic_field`` / ``fx.df_lens``) for the differentiable
FFT-propagation core and on raw JAX (``jax.grad`` / ``jax.jit`` / ``jax.vmap``
/ ``jax.pmap``) for the project-specific DM / SLM / GS / SPGD logic that
``chromatix`` does not model natively.

The physics is coherent Fraunhofer far-field diffraction:

    E = pupil * exp(i * piston_rm(phase))     # pupil-plane field
    I = |fftshift(fft2(E))|^2                 # focal-plane intensity

which maps directly onto ``fx.fft(E, shift=True)`` (verified numerically
identical to ``torch.fft.fftshift(fft2(E))`` to float32 precision, no extra
scaling).

Modules
-------
    pupil_jax     : jnp grid / pupil mask / I0 (JAX version of ``pupil.py``)
    turbulence_jax: Kolmogorov phase screen (JAX version of ``turbulence.py``)
    dm_jax        : DM actuators + influence functions (JAX version of ``dm.py``)
    optics_jax    : far-field intensity + metrics (JAX/chromatix core)
    slm_jax       : pixelated SLM with fill factor (JAX version of ``slm.py``)
    jax_adam      : minimal Adam optimiser on JAX arrays
    shaping_jax   : target shaping (JAX version of ``shaping.py``)
    gs_jax        : Gerchberg-Saxton (JAX version of ``gs.py``)
    spgd_jax      : SPGD (JAX version of ``spgd.py``)
    hgd_jax       : H-GD (JAX version of ``hgd.py``)
    torch_gd_jax  : autograd Torch-GD via jax.grad (JAX version)

The backend reuses ``differential_shaping.params`` for all shared constants so
the JAX and torch pipelines stay parameter-identical.
"""

# Import submodules lazily so the package is importable incrementally.
# Each submodule is a self-contained JAX port of a torch module; importing one
# only pulls its own dependencies.
from . import pupil_jax  # noqa: F401
from . import turbulence_jax  # noqa: F401
from . import dm_jax  # noqa: F401
from . import optics_jax  # noqa: F401

# Remaining modules (slm_jax, jax_adam, shaping_jax, gs_jax, spgd_jax,
# hgd_jax, torch_gd_jax) are imported lazily via __getattr__ so that the
# core simulation is usable even before the optimiser ports land.
_LAZY = {
    "slm_jax": ".slm_jax",
    "jax_adam": ".jax_adam",
    "shaping_jax": ".shaping_jax",
    "gs_jax": ".gs_jax",
    "spgd_jax": ".spgd_jax",
    "hgd_jax": ".hgd_jax",
    "torch_gd_jax": ".torch_gd_jax",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return importlib.import_module(_LAZY[name], __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return [*_LAZY, "pupil_jax", "turbulence_jax", "dm_jax", "optics_jax"]
