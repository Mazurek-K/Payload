from helpers import airy_disk, airy_disk2D, get_optical_factors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from config import (
    h, c, k, WAVELENGTH_UM, AU_2_M,
     TRANSMISSION, QE,PIXEL_PITCH, FOCAL_LENGTH,
    AST_DISTANCE, AST_TEMPERATURE, AST_DIAMETER, EMISSIVITY, APERTURE
)
from src.asteroid.config import EPSILON_F


def calculate_asteroid_signal(
    diameter=AST_DIAMETER,
    distance=AST_DISTANCE,
    temperature=AST_TEMPERATURE,
    emissivity=EMISSIVITY,
    epsilon_f= EPSILON_F,
    aperture = APERTURE,
    focal_length = FOCAL_LENGTH,
    per_pixel = False,
    per_pixel2D = False,
    plot = False
):
    if per_pixel and per_pixel2D:
        raise ValueError("Choose between per_pixel and per_pixel2D")

    lens_area, _ = get_optical_factors(aperture, focal_length)
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
        * lens_area
        * TRANSMISSION
        * QE
        * lam / (h * c)
        * epsilon_f
    )

    # Total detected count rate: electrons/s on the sensor overall

    if per_pixel:
        n_pix_source,_  = airy_disk(aperture = aperture, focal_length = focal_length)
        dC_dlambda_perpix = dC_dlambda / n_pix_source
        electron_rate_perpix = np.trapezoid(dC_dlambda_perpix, WAVELENGTH_UM)
        return electron_rate_perpix

    if per_pixel2D:
        hs = airy_disk2D( aperture = aperture, focal_length = focal_length)

        # Distribute the spectral signal across the 2D pixel masks
        dC_dlambda_2d = (
                dC_dlambda[:, np.newaxis, np.newaxis]* np.array(hs) )

        # Integrate over wavelength
        electron_rate_2d = np.trapezoid(
            dC_dlambda_2d,
            WAVELENGTH_UM,
            axis=0
        )

        if plot:
            fig, ax = plt.subplots()
            im = ax.imshow(
                electron_rate_2d,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                vmin=0,
                vmax=np.max(electron_rate_2d)
            )

            n_pix = electron_rate_2d.shape[0]
            ax.set_xticks(np.arange(-0.5, n_pix, 1))
            ax.set_yticks(np.arange(-0.5, n_pix, 1))
            ax.grid(color="red", linewidth=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Electron rate [electrons/s]", fontsize= 16)
            plt.show()
        return electron_rate_2d
    else:
        C = np.trapezoid(dC_dlambda, WAVELENGTH_UM)
        return C



def plot_asteroid_parameter_space(
    threshold = 1e2,
    diameter_range = (20, 50),
    distance_range = (0.03*149597870700, 0.16*149597870700),
    n_diameters=100,
    n_distances=100,
    log_scale=True,
):
    # Generate parameter ranges
    diameters = np.linspace(
        diameter_range[0],
        diameter_range[1],
        n_diameters
    )

    distances = np.linspace(
        distance_range[0],
        distance_range[1],
        n_distances
    )

    if np.any(diameters <= 0) or np.any(distances <= 0):
        raise ValueError("Diameters and distances must be positive.")

    # Calculate reference signal using existing function


    # Create 2D parameter grid
    D_grid, R_grid = np.meshgrid(diameters, distances, indexing="ij")

    # # Scale reference count rate
    electron_rate_perpix = np.zeros_like(D_grid)

    for i, diameter in enumerate(diameters):
        for j, distance in enumerate(distances):
            electron_rate_perpix[i, j] = calculate_asteroid_signal(
                diameter=diameter,
                distance=distance,
                per_pixel= True
            )

    plt.figure(figsize=(10, 7))

    distances_AU = distances/AU_2_M

    if log_scale:
        mesh = plt.pcolormesh(
            distances_AU,
            diameters,
            electron_rate_perpix,
            shading="auto",
            cmap="inferno",
            norm=LogNorm()
        )
    else:
        mesh = plt.pcolormesh(
            distances,
            diameters,
            electron_rate_perpix,
            shading="auto",
            cmap="inferno"
        )

    cbar = plt.colorbar(mesh)
    cbar.set_label("Electron rate (electrons/s/pix)")

    plt.xlabel("Asteroid distance (same units as AST_DISTANCE) [AU] ")
    plt.ylabel("Asteroid diameter (same units as AST_DIAMETER)")
    plt.title("Asteroid electron rate")

    plt.tight_layout()
    plt.show()

    return diameters, distances, electron_rate_perpix
