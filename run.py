# -*- coding: utf-8 -*-
"""
Entry point for the H-GD / SPGD wavefront-sensorless AO simulation.

Orchestrates the simulation pipeline:
  1. Generate a Kolmogorov-like turbulence phase screen.
  2. Build the deformable mirror influence functions.
  3. Run SPGD and H-GD optimizers to correct the wavefront.
  4. Plot and save the comparison figure.
"""

from pathlib import Path

import numpy as np
from loguru import logger

from differential_shaping import params
from differential_shaping.simulation import (
    pupil,
    generate_turbulence_phase,
    generate_influence_functions,
    compute_metrics,
)
from differential_shaping.optimization import spgd_optimization, hgd_optimization
from differential_shaping.visualization import plot_results


def main() -> None:
    logger.info("Generate turbulence phase screen...")
    turb_phase = generate_turbulence_phase(
        params.N, params.pixel_size, params.r0, params.target_phase_rms, params.seed_phase
    )
    logger.info(f"Actual pupil RMS = {np.std(turb_phase[pupil]):.6f} rad")

    J_init, SR_init, _ = compute_metrics(turb_phase)
    logger.info(f"Initial: J={J_init:.3f} pix, SR={SR_init:.4f}")

    inf_funcs = generate_influence_functions()
    inf_flat = inf_funcs.reshape(params.n_act, -1)
    logger.info(
        f"Actuators: {params.n_act}; spacing={params.act_spacing*1e3:.3f} mm; "
        f"influence width={params.sigma_inf*1e3:.3f} mm"
    )

    logger.info("\n===== SPGD =====")
    u_spgd, dm_spgd, J_spgd, SR_spgd = spgd_optimization(turb_phase, inf_flat)

    logger.info("\n===== H-GD =====")
    u_hgd, dm_hgd, J_hgd, SR_hgd = hgd_optimization(turb_phase, inf_flat)

    phase_spgd = turb_phase + dm_spgd
    phase_hgd = turb_phase + dm_hgd
    J_spgd_end, SR_spgd_end, _ = compute_metrics(phase_spgd)
    J_hgd_end, SR_hgd_end, _ = compute_metrics(phase_hgd)

    logger.info("\n===== Performance comparison =====")
    logger.info(f"Initial:    J={J_init:.3f} pix, SR={SR_init:.4f}")
    logger.info(f"SPGD final: J={J_spgd_end:.3f} pix, SR={SR_spgd_end:.4f}")
    logger.info(f"H-GD final: J={J_hgd_end:.3f} pix, SR={SR_hgd_end:.4f}")
    logger.info(f"SPGD: ΔJ={J_init - J_spgd_end:.3f} pix, ΔSR={SR_spgd_end - SR_init:.4f}")
    logger.info(f"H-GD: ΔJ={J_init - J_hgd_end:.3f} pix, ΔSR={SR_hgd_end - SR_init:.4f}")

    out_path = Path(__file__).parent / "H_GD_phase_spot_fixed_result.png"
    plot_results(
        turb_phase, phase_spgd, phase_hgd,
        J_init, SR_init,
        J_spgd, SR_spgd,
        J_hgd, SR_hgd,
        out_path,
    )
    logger.info(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()