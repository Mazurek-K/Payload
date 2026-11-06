
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
np.set_printoptions(linewidth=200)

from config import (
    HDF5_FILE,
    COMPONENT,
    WAVELENGTH_UM,
    ECLIPTIC_LON0,
    ECLIPTIC_LAT0,
    PIXEL_ROW,
    h,
    c,
    LENS_AREA,
    OMEGA_PIX,
    TRANSMISSION,
    QE,
    SR_TO_ARCSEC2,
    EPSILON_F
)


def calculate_background():

    with h5py.File(HDF5_FILE, "r") as file:

        lon = file["coordinates/ecliptic_longitude_deg"][:]
        lat = file["coordinates/ecliptic_latitude_deg"][:]
        all_wavelengths = file["spectral/wavelength_um"][:]

        component_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in file["background/component_names"][:]
        ]
        all_values = file["background/values"][:]

    # Allow short component names as well as the full HDF5 names.
    component_aliases = {
        "zodiacal": "zodiacal_MJy_sr",
        "ISM": "ISM_MJy_sr",
        "stars": "stars_MJy_sr",
        "CIB": "CIB_MJy_sr",
        "total": "total_MJy_sr",
    }

    selected_component = component_aliases.get(COMPONENT, COMPONENT)

    if selected_component not in component_names:
        raise ValueError( f"Component '{COMPONENT}' not found in HDF5 file. "f"Available components: {component_names}")
    component_index = component_names.index(selected_component)

    # Array axes:
    # (ecliptic longitude, ecliptic latitude, wavelength, component)
    sky = all_values[:, :, :, component_index]

    wavelength_indices = []
    for requested_wavelength in WAVELENGTH_UM:
        matches = np.where(
            np.isclose(all_wavelengths,requested_wavelength,atol=1e-6,rtol=0,))[0]

        if len(matches) == 0:
            raise ValueError(
                f"Wavelength {requested_wavelength} µm "
                "is not available in the HDF5 file."
            )
        wavelength_indices.append(matches[0])

    wavelength_indices = np.asarray(wavelength_indices)
    wavelength = all_wavelengths[wavelength_indices]
    sky = sky[:, :, wavelength_indices]


    # Extract 1D coordinate axes
    lon_1d = lon[:, 0].astype(float)
    lat_1d = lat[0, :].astype(float)

    # Unwrap longitude (only if the original values cross 360 -> 0)
    lon_1d = np.rad2deg(np.unwrap(np.deg2rad(lon_1d)))

    # Map longitude to -180..180
    lon_1d = (lon_1d + 180) % 360 - 180

    # Sort coordinates and reorder sky
    lon_order = np.argsort(lon_1d)
    lat_order = np.argsort(lat_1d)

    lon_1d = lon_1d[lon_order]
    lat_1d = lat_1d[lat_order]

    sky_sorted = sky[lon_order, :, :][:, lat_order, :]

    # Interpolate all wavelengths simultaneously
    interpolator = RegularGridInterpolator(
        (lon_1d, lat_1d),
        sky_sorted,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Position at which to interpolate
    position = np.array([[ECLIPTIC_LON0, ECLIPTIC_LAT0]])

    # Result contains one value per wavelength
    intensity = interpolator(position)[0]

    print("Interpolated values:", intensity )
    print("Number of wavelengths:", len(intensity))


    if np.any(~np.isfinite(intensity)):
        raise ValueError(
            "Background interpolation returned NaN. "
            "Check ECLIPTIC_LON0, ECLIPTIC_LAT0 "
            "and the completeness of the HDF5 grid."
        )

    # --------------------------------------------------
    # Convert spectral intensity to photon rate
    # --------------------------------------------------
    lam = wavelength * 1e-6 # convert to  [µm -> m]
    I_nu = intensity * 1e-20  #  [MJy/sr -> W m^-2 Hz^-1 sr^-1]
    I_lam = (I_nu * c/ lam ** 2 * 1e-6 ) # W m^-2 sr^-1 µm^-1

    photon_spectrum = (
            I_lam
            * lam
            / (h * c)
            * LENS_AREA
            * OMEGA_PIX
            * TRANSMISSION
    )  # photons / (s * pixel * µm)

    # --------------------------------------------------
    # Integrate over wavelength
    # --------------------------------------------------
    photon_rate = np.trapezoid(
        photon_spectrum,
        wavelength,
    )

    electron_rate = photon_rate * QE  # electrons/s/pixel
    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("Selected component:", selected_component)
    print("Ecliptic position:", ECLIPTIC_LON0, ECLIPTIC_LAT0)
    print("Wavelengths (µm):", wavelength)
    print("Background (MJy/sr):", intensity)
    print("Background:", electron_rate, "electrons/s/pixel")

    return electron_rate