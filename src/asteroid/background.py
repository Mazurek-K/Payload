import csv
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from config import (
    CSV_FILE, COMPONENT, WAVELENGTH_UM, RA0, DEC0,
    PIXEL_ROW, h, c, LENS_AREA, OMEGA_PIX, TRANSMISSION, QE
)


def calculate_background():
    rows = []

    with open(CSV_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        required = {"ra_deg", "dec_deg", "wavelength_um", COMPONENT}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            wl = float(row["wavelength_um"])
            if np.any(np.isclose(wl, WAVELENGTH_UM, atol=1e-6, rtol=0)):
                rows.append((
                    float(row["ra_deg"]),
                    float(row["dec_deg"]),
                    wl,
                    float(row[COMPONENT])
                ))

    if not rows:
        raise ValueError("No data found for selected wavelengths.")

    data = np.asarray(rows)
    ra_values, dec_values, wl_values, values = data.T

    ra = np.sort(np.unique(ra_values))
    dec = np.sort(np.unique(dec_values))
    wavelength = np.sort(np.unique(wl_values))

    sky = np.full((len(wavelength), len(dec), len(ra)), np.nan)
    ri = np.searchsorted(ra, ra_values)
    di = np.searchsorted(dec, dec_values)
    wi = np.searchsorted(wavelength, wl_values)
    sky[wi, di, ri] = values

    # Interpolate background at the selected sky position
    intensity = np.array([
        RegularGridInterpolator(
            (dec, ra), sky[i],
            bounds_error=False, fill_value=np.nan
        )((DEC0, RA0)).item()
        for i in range(len(wavelength))
    ])

    if np.any(~np.isfinite(intensity)):
        raise ValueError(
            "Background interpolation returned NaN; check RA0, DEC0 "
            "and the completeness of the IRSA grid."
        )

    # Approximate a spatially uniform background over the sensor
    intensity_fov = np.broadcast_to(
        intensity[:, None, None],
        (len(wavelength), PIXEL_ROW, PIXEL_ROW)
    )

    lam = wavelength * 1e-6
    I_nu = intensity_fov * 1e-20  # W m^-2 Hz^-1 sr^-1
    I_lam = I_nu * c / lam[:, None, None]**2 * 1e-6

    photon_spectrum = ( I_lam * lam[:, None, None] / (h * c)* LENS_AREA * OMEGA_PIX * TRANSMISSION) #photons / (s * pixel * µm)

    photon_rate = np.trapezoid(photon_spectrum, wavelength, axis=0)
    electron_rate = photon_rate * QE # [electrons/s/pixel]

    print("Wavelengths (µm):", wavelength)
    print("Background (MJy/sr):", intensity)
    print("Mean background:", np.nanmean(electron_rate), "electrons/s/pixel")

    return electron_rate

