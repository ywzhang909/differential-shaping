# -*- coding: utf-8 -*-
"""
H-GD / SPGD wavefront-sensorless AO simulation

本版主要修正：
1. 相位屏频域合成的离散尺度：频域采样间隔为 df=1/D，不是 pupil 采样间隔 dx；
   NumPy 的 ifft2 自带 1/N^2，因此合成后乘 N^2。这样归一化前的相位 RMS 不会落到
   1e-12 量级，也不会被 eps 改坏。
2. 相位屏归一化使用检查式除法，不用 “std + 1e-12” 这种会改变目标 RMS 的写法。
3. 远场光斑显示使用零填充 FFT，提高焦平面采样；显示时按无像差峰值统一归一化，
   不再每幅图按自己的最大值归一化。
4. 相位屏/残余相位显示加入 pupil mask、origin='lower'、equal aspect 和对称色标。
5. 优化循环中缓存 DM 面形；SPGD/H-GD 的扰动面形预计算，避免每次重复 tensordot。
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

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

# ------------------------- grids and pupil -------------------------
grid_1d = (np.arange(N) - N / 2) * pixel_size
XX, YY = np.meshgrid(grid_1d, grid_1d)
pupil = (XX**2 + YY**2) <= (D / 2) ** 2
pupil_float = pupil.astype(float)
extent_pupil_mm = [-D / 2 * 1e3, D / 2 * 1e3, -D / 2 * 1e3, D / 2 * 1e3]


def remove_piston(phase: np.ndarray) -> np.ndarray:
    """Remove piston inside the pupil and set the outside-pupil area to 0."""
    out = np.array(phase, dtype=float, copy=True)
    out[pupil] -= np.mean(out[pupil])
    out[~pupil] = 0.0
    return out


# ------------------------- turbulence phase screen -------------------------
def generate_turbulence_phase(
    N: int,
    pixel_size: float,
    r0: float,
    target_rms: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Kolmogorov-like phase screen and scale pupil RMS to target_rms.

    Discrete spectral synthesis note:
    - frequency spacing is df = 1 / (N * pixel_size) = 1 / D;
    - numpy.ifft2 contains 1 / N^2, so multiply the inverse transform by N^2.
    """
    rng = np.random.default_rng(seed)
    D_local = N * pixel_size
    df = 1.0 / D_local

    fx = np.fft.fftfreq(N, d=pixel_size)
    fy = np.fft.fftfreq(N, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    fr = np.sqrt(FX**2 + FY**2)

    PSD = np.zeros_like(fr)
    mask = fr > 0
    PSD[mask] = 0.023 * r0 ** (-5.0 / 3.0) * fr[mask] ** (-11.0 / 3.0)

    # fftfreq returns the ordering expected by ifft2; do not fftshift/ifftshift this spectrum.
    noise = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    phase_fft = np.sqrt(PSD) * noise * df
    phase = np.real(np.fft.ifft2(phase_fft)) * (N**2)

    phase = remove_piston(phase)
    raw_rms = np.std(phase[pupil])
    if raw_rms <= np.finfo(float).tiny:
        raise RuntimeError("Generated phase screen has near-zero RMS; check PSD/sampling parameters.")
    phase *= target_rms / raw_rms
    return phase


# ------------------------- deformable mirror -------------------------
def generate_actuator_grid() -> tuple[np.ndarray, np.ndarray]:
    margin = act_spacing / 2
    xs = np.linspace(-D / 2 + margin, D / 2 - margin, n_act_x)
    ys = np.linspace(-D / 2 + margin, D / 2 - margin, n_act_y)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def generate_influence_functions() -> np.ndarray:
    act_x, act_y = generate_actuator_grid()
    inf = np.zeros((n_act, N, N), dtype=np.float64)
    for i, (x0, y0) in enumerate(zip(act_x, act_y)):
        r2 = (XX - x0) ** 2 + (YY - y0) ** 2
        z = np.exp(-r2 / (2 * sigma_inf**2)) * pupil_float
        z[pupil] -= np.mean(z[pupil])
        z[~pupil] = 0.0
        z /= np.max(np.abs(z)) + 1e-15
        inf[i] = z
    return inf


# ------------------------- focal-plane metrics used by optimizer -------------------------
I0 = np.abs(np.fft.fftshift(np.fft.fft2(pupil_float))) ** 2
I0_peak = np.max(I0)
yy_metric, xx_metric = np.indices((N, N))


def far_field_intensity_metric(phase: np.ndarray) -> np.ndarray:
    E = pupil_float * np.exp(1j * remove_piston(phase))
    return np.abs(np.fft.fftshift(np.fft.fft2(E))) ** 2


def compute_metrics(phase: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return J = centroid mean radius in unpadded pixels, SR, and intensity."""
    I = far_field_intensity_metric(phase)
    total = np.sum(I) + 1e-30
    cx = np.sum(xx_metric * I) / total
    cy = np.sum(yy_metric * I) / total
    r_pix = np.sqrt((xx_metric - cx) ** 2 + (yy_metric - cy) ** 2)
    J_mr = np.sum(r_pix * I) / total
    SR = np.max(I) / (I0_peak + 1e-30)
    return J_mr, SR, I


# ------------------------- padded focal-plane display -------------------------
def far_field_intensity_padded(phase: np.ndarray, pad_factor: int = pad_factor_show) -> np.ndarray:
    """Zero-padded Fraunhofer intensity for visualization only."""
    M = N * pad_factor
    s = (M - N) // 2
    E = pupil_float * np.exp(1j * remove_piston(phase))
    Epad = np.zeros((M, M), dtype=np.complex128)
    Epad[s : s + N, s : s + N] = E
    return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Epad)))) ** 2


def crop_center(img: np.ndarray, half_width_pix: int) -> tuple[np.ndarray, list[float]]:
    M = img.shape[0]
    c = M // 2
    a = max(0, c - half_width_pix)
    b = min(M, c + half_width_pix + 1)
    crop = img[a:b, a:b]
    # x/y in units of lambda*f/D.  One unpadded FFT pixel corresponds to lambda*f/D.
    extent = [
        (a - c) / pad_factor_show,
        (b - 1 - c) / pad_factor_show,
        (a - c) / pad_factor_show,
        (b - 1 - c) / pad_factor_show,
    ]
    return crop, extent


# ------------------------- optimization -------------------------
def update_by_two_sided_perturbation(
    turb_phase: np.ndarray,
    u: np.ndarray,
    dm_u: np.ndarray,
    delta_u: np.ndarray,
    dm_delta: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize J with a two-sided perturbation estimate.

    Because dm_delta = DM(delta_u), the control update is scalar * delta_u,
    so the DM surface update is the same scalar * dm_delta.
    """
    delta = np.max(np.abs(delta_u))
    J_p, _, _ = compute_metrics(turb_phase + dm_u + dm_delta)
    J_m, _, _ = compute_metrics(turb_phase + dm_u - dm_delta)
    scalar = -alpha * (J_p - J_m) / (2 * delta**2 + 1e-30)
    u = u + scalar * delta_u
    dm_u = dm_u + scalar * dm_delta
    return u, dm_u


def spgd_optimization(turb_phase: np.ndarray, inf_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed_spgd)
    patterns = (2 * rng.integers(0, 2, size=(max_iter, n_act)) - 1).astype(np.float64)
    delta_patterns = patterns * delta_amp

    # Precompute perturbation DM shapes in one BLAS multiplication.
    dm_deltas = (delta_patterns @ inf_flat).reshape(max_iter, N, N)

    u = np.zeros(n_act)
    dm_u = np.zeros((N, N))
    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)
    for it in range(max_iter):
        u, dm_u = update_by_two_sided_perturbation(
            turb_phase, u, dm_u, delta_patterns[it], dm_deltas[it], alpha_spgd
        )
        J_hist[it], SR_hist[it], _ = compute_metrics(turb_phase + dm_u)
        if it % 500 == 0:
            print(f"SPGD {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}")
    return u, dm_u, J_hist, SR_hist


def hgd_optimization(turb_phase: np.ndarray, inf_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H = hadamard(n_act, dtype=np.float64)
    # Skip the all-one column; it is mainly a piston-like common mode after influence-function projection.
    patterns = H[:, 1:].T.copy()
    delta_patterns = patterns * delta_amp
    dm_deltas = (delta_patterns @ inf_flat).reshape(n_act - 1, N, N)

    u = np.zeros(n_act)
    dm_u = np.zeros((N, N))
    J_hist = np.zeros(max_iter)
    SR_hist = np.zeros(max_iter)
    for it in range(max_iter):
        k = it % (n_act - 1)
        u, dm_u = update_by_two_sided_perturbation(
            turb_phase, u, dm_u, delta_patterns[k], dm_deltas[k], alpha_hgd
        )
        J_hist[it], SR_hist[it], _ = compute_metrics(turb_phase + dm_u)
        if it % 500 == 0:
            print(f"H-GD {it:4d}: J={J_hist[it]:.3f} pix, SR={SR_hist[it]:.4f}")
    return u, dm_u, J_hist, SR_hist


# ------------------------- plotting -------------------------
def plot_results(
    turb_phase: np.ndarray,
    phase_spgd: np.ndarray,
    phase_hgd: np.ndarray,
    J_init: float,
    SR_init: float,
    J_spgd: np.ndarray,
    SR_spgd: np.ndarray,
    J_hgd: np.ndarray,
    SR_hgd: np.ndarray,
    out_path: Path,
) -> None:
    J_spgd_end, SR_spgd_end, _ = compute_metrics(phase_spgd)
    J_hgd_end, SR_hgd_end, _ = compute_metrics(phase_hgd)

    I0_pad = far_field_intensity_padded(np.zeros_like(turb_phase))
    I0_pad_peak = np.max(I0_pad)
    spots = [
        ("Initial", turb_phase, J_init, SR_init),
        ("SPGD", phase_spgd, J_spgd_end, SR_spgd_end),
        ("H-GD", phase_hgd, J_hgd_end, SR_hgd_end),
    ]
    half_pix = int(round(spot_half_width_lamD * pad_factor_show))

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.05], height_ratios=[1, 1])

    last_im = None
    for col, (name, phase, Jv, SRv) in enumerate(spots):
        ax = fig.add_subplot(gs[0, col])
        I_rel = far_field_intensity_padded(phase) / (I0_pad_peak + 1e-30)
        I_db, extent = crop_center(10 * np.log10(np.maximum(I_rel, 1e-8)), half_pix)
        last_im = ax.imshow(I_db, extent=extent, origin="lower", cmap="jet", vmin=-40, vmax=0)
        ax.set_title(f"{name}\nJ={Jv:.2f} pix, SR={SRv:.3f}")
        ax.set_xlabel(r"$x/(\lambda f/D)$")
        ax.set_ylabel(r"$y/(\lambda f/D)$")
        ax.set_aspect("equal")
    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(last_im, cax=cax, label="Intensity / ideal peak (dB)")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(J_spgd, label="SPGD")
    ax.plot(J_hgd, label="H-GD")
    ax.axhline(J_init, color="k", linestyle="--", label="Initial")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("J = mean radius (pixels)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(SR_spgd, label="SPGD")
    ax.plot(SR_hgd, label="H-GD")
    ax.axhline(SR_init, color="k", linestyle="--", label="Initial")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SR")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # Initial phase screen display
    ax = fig.add_subplot(gs[1, 2])
    phase_show = np.ma.array(remove_piston(turb_phase), mask=~pupil)
    lim = np.max(np.abs(phase_show))
    im_phase = ax.imshow(
        phase_show,
        extent=extent_pupil_mm,
        origin="lower",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )
    ax.set_title(f"Initial phase screen\nRMS={np.std(turb_phase[pupil]):.3f} rad")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(im_phase, ax=ax, fraction=0.046, pad=0.04, label="rad")

    # SPGD residual phase display
    ax = fig.add_subplot(gs[1, 3])
    residual_spgd = remove_piston(phase_spgd)
    residual_spgd_show = np.ma.array(residual_spgd, mask=~pupil)
    lim_spgd = max(np.max(np.abs(residual_spgd_show)), 1e-6)
    im_res_spgd = ax.imshow(
        residual_spgd_show,
        extent=extent_pupil_mm,
        origin="lower",
        cmap="RdBu_r",
        vmin=-lim_spgd,
        vmax=lim_spgd,
        interpolation="nearest",
    )
    ax.set_title(f"SPGD residual phase\nRMS={np.std(residual_spgd[pupil]):.3f} rad")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(im_res_spgd, ax=ax, fraction=0.046, pad=0.04, label="rad")

    # H-GD residual phase display
    ax = fig.add_subplot(gs[1, 4])
    residual = remove_piston(phase_hgd)
    residual_show = np.ma.array(residual, mask=~pupil)
    lim = max(np.max(np.abs(residual_show)), 1e-6)
    im_res = ax.imshow(
        residual_show,
        extent=extent_pupil_mm,
        origin="lower",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )
    ax.set_title(f"H-GD residual phase\nRMS={np.std(residual[pupil]):.3f} rad")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(im_res, ax=ax, fraction=0.046, pad=0.04, label="rad")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------- main -------------------------
def main() -> None:
    print("Generate turbulence phase screen...")
    turb_phase = generate_turbulence_phase(N, pixel_size, r0, target_phase_rms, seed_phase)
    print(f"Actual pupil RMS = {np.std(turb_phase[pupil]):.6f} rad")

    J_init, SR_init, _ = compute_metrics(turb_phase)
    print(f"Initial: J={J_init:.3f} pix, SR={SR_init:.4f}")

    inf_funcs = generate_influence_functions()
    inf_flat = inf_funcs.reshape(n_act, -1)
    print(f"Actuators: {n_act}; spacing={act_spacing*1e3:.3f} mm; influence width={sigma_inf*1e3:.3f} mm")

    print("\n===== SPGD =====")
    u_spgd, dm_spgd, J_spgd, SR_spgd = spgd_optimization(turb_phase, inf_flat)

    print("\n===== H-GD =====")
    u_hgd, dm_hgd, J_hgd, SR_hgd = hgd_optimization(turb_phase, inf_flat)

    phase_spgd = turb_phase + dm_spgd
    phase_hgd = turb_phase + dm_hgd
    J_spgd_end, SR_spgd_end, _ = compute_metrics(phase_spgd)
    J_hgd_end, SR_hgd_end, _ = compute_metrics(phase_hgd)

    print("\n===== Performance comparison =====")
    print(f"Initial:    J={J_init:.3f} pix, SR={SR_init:.4f}")
    print(f"SPGD final: J={J_spgd_end:.3f} pix, SR={SR_spgd_end:.4f}")
    print(f"H-GD final: J={J_hgd_end:.3f} pix, SR={SR_hgd_end:.4f}")
    print(f"SPGD: ΔJ={J_init - J_spgd_end:.3f} pix, ΔSR={SR_spgd_end - SR_init:.4f}")
    print(f"H-GD: ΔJ={J_init - J_hgd_end:.3f} pix, ΔSR={SR_hgd_end - SR_init:.4f}")

    out_path = Path(__file__).with_name("H_GD_phase_spot_fixed_result.png")
    plot_results(turb_phase, phase_spgd, phase_hgd, J_init, SR_init, J_spgd, SR_spgd, J_hgd, SR_hgd, out_path)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
