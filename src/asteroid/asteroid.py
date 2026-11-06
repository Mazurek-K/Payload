import numpy as np

from config import (
    h, c, k, WAVELENGTH_UM,
    LENS_AREA, TRANSMISSION, QE,
    AST_DISTANCE, AST_TEMPERATURE, AST_DIAMETER, EMISSIVITY
)
from src.asteroid.config import EPSILON_F


def calculate_asteroid_signal(
    diameter=AST_DIAMETER,
    distance=AST_DISTANCE,
    temperature=AST_TEMPERATURE,
    emissivity=EMISSIVITY,
    epsilon_f= EPSILON_F
):
    lam = WAVELENGTH_UM * 1e-6

    # Planck spectral radiance: W m^-2 sr^-1 m^-1
    B_lam = (
        2 * h * c**2 / lam**5
        / np.expm1(h * c / (lam * k * temperature))
    )

    # Apparent solid angle of the asteroid
    omega_ast = np.pi * (diameter / 2)**2 / distance**2

    # Spectral flux density: W m^-2 µm^-1
    F_lam = emissivity * B_lam * omega_ast * 1e-6

    # Detected count-rate spectrum: electrons/s/µm
    dC_dlambda = (
        F_lam
        * LENS_AREA
        * TRANSMISSION
        * QE
        * lam / (h * c)
        * epsilon_f
    )

    # Total detected count rate: electrons/s on the sensor overall
    C = np.trapezoid(dC_dlambda, WAVELENGTH_UM)

    print("Asteroid count rate C:", C, "electrons/s")
    return C