"""
flow_common.py -- shared Madelung utilities for the potential-flow pipeline

  * smooth_boundary_window : cosine-taper window for FFT periodicity
  * madelung_encode        : velocity potential -> unit-density wavefunction
  * madelung_decode        : wavefunction -> velocity field (probability current)

Defaults: hbar = mass = 1.
"""
import numpy as np


def smooth_boundary_window(XX, YY, L, taper_ratio=0.12):
    """Cosine-tapered window for FFT periodicity."""
    w = L * taper_ratio
    win_x = np.ones_like(XX)
    win_y = np.ones_like(XX)
    mask_l = XX < -L + w
    mask_r = XX > L - w
    win_x[mask_l] = 0.5 * (1 - np.cos(np.pi * (XX[mask_l] + L) / w))
    win_x[mask_r] = 0.5 * (1 - np.cos(np.pi * (L - XX[mask_r]) / w))
    mask_b = YY < -L + w
    mask_t = YY > L - w
    win_y[mask_b] = 0.5 * (1 - np.cos(np.pi * (YY[mask_b] + L) / w))
    win_y[mask_t] = 0.5 * (1 - np.cos(np.pi * (L - YY[mask_t]) / w))
    return win_x * win_y


def madelung_encode(phi, rho=1.0, hbar=1.0):
    """Encode a (unit-density) velocity potential as the phase of a wavefunction."""
    return np.sqrt(rho) * np.exp(1j * phi / hbar)


def madelung_decode(psi, hbar=1.0, mass=1.0, dx=1.0, dy=1.0):
    """Decode the velocity field from the quantum probability current
    v = (hbar/m) Im(psi* grad psi) / |psi|^2."""
    ny, nx = psi.shape
    psi_xp = np.roll(psi, -1, axis=1)
    psi_xm = np.roll(psi, 1, axis=1)
    psi_yp = np.roll(psi, -1, axis=0)
    psi_ym = np.roll(psi, 1, axis=0)
    j_x = np.imag(np.conj(psi) * (psi_xp - psi_xm) / (2 * dx))
    j_y = np.imag(np.conj(psi) * (psi_yp - psi_ym) / (2 * dy))
    j_x[:, 0] = np.imag(np.conj(psi[:, 0]) * (psi[:, 1] - psi[:, 0]) / dx)
    j_x[:, -1] = np.imag(np.conj(psi[:, -1]) * (psi[:, -1] - psi[:, -2]) / dx)
    j_y[0, :] = np.imag(np.conj(psi[0, :]) * (psi[1, :] - psi[0, :]) / dy)
    j_y[-1, :] = np.imag(np.conj(psi[-1, :]) * (psi[-1, :] - psi[-2, :]) / dy)
    rho = np.abs(psi) ** 2 + 1e-15
    return (hbar / mass) * j_x / rho, (hbar / mass) * j_y / rho
