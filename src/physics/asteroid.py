import numpy as np

from config import (
    h, c, k, WAVELENGTH_UM,
    LENS_AREA, TRANSMISSION, QE, AST_DISTANCE, AST_TEMPERATURE, AST_DIAMETER
)


def calculate_asteroid_signal(
    diameter = AST_DIAMETER, distance=AST_DISTANCE, temperature=AST_TEMPERATURE, emissivity=1.0):
    lam = WAVELENGTH_UM * 1e-6

    # Planck spectral radiance: W m^-2 sr^-1 m^-1
    B_lam = (
        2 * h * c**2 / lam**5
        / np.expm1(h * c / (lam * k * temperature))
    )

    # Apparent solid angle of a spherical asteroid
    omega_ast = np.pi * (diameter / 2)**2 / distance**2

    # Flux density per wavelength, W m^-2 µm^-1
    F_lam = emissivity * B_lam * omega_ast * 1e-6

    # Detected electron spectrum: electrons/s/µm
    electron_spectrum = (
            F_lam * LENS_AREA * TRANSMISSION * QE * lam / (h * c)
    )

    # Total detected electron rate integrated over wavelength: electrons/s
    electron_rate = np.trapezoid(electron_spectrum, WAVELENGTH_UM)

    print("Asteroid electron rate:", electron_rate, "electrons/s")
    return electron_rate