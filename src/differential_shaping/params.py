# -*- coding: utf-8 -*-
"""
System, algorithm, and display parameters for H-GD / SPGD AO simulation.

All constants are extracted from the original monolithic script to enable
modular imports across simulation / optimization / visualization packages.
"""

# ------------------------- system parameters -------------------------
wavelength = 532e-9          # wavelength (m)
pixel_size = 10e-6           # pupil-plane sampling pitch (m)
N = 128                      # pupil grid size
D = N * pixel_size           # pupil diameter / computational aperture width (m)
f = 0.1                      # focal length (m)
r0 = 0.00006                 # Fried parameter (m); target_rms below sets final phase strength

target_phase_rms = 1.0       # pupil RMS of the generated turbulence phase, rad

# DM: 16 x 16 square actuator array
n_act_x = 16
n_act_y = 16
n_act = n_act_x * n_act_y
act_spacing = D / n_act_x
sigma_inf = act_spacing * 0.65

# Algorithm parameters
max_iter = 2000
delta_amp = 0.05             # perturbation amplitude, rad
alpha_spgd = 0.02
alpha_hgd = 0.04
seed_phase = 42
seed_spgd = 123

# Display parameters
pad_factor_show = 4           # zero-padding factor only for focal-plane visualization
spot_half_width_lamD = 14     # show +/- this many lambda*f/D units
