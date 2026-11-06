import time
import requests
import numpy as np
import h5py
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from astropy.coordinates import (
    BarycentricMeanEcliptic,
    FK5,
    SkyCoord,
)
from astropy.time import Time
import astropy.units as u


URL = (
    "https://irsa.ipac.caltech.edu/"
    "cgi-bin/BackgroundModel/nph-bgmodel"
)

# Observation date: 21 March 2026
YEAR = 2026
DAY_OF_YEAR = 80
OBSERVATION_DATE = "2026-03-21"

# Grid is centred on the vernal equinox in Ecliptic J2000.
SUN_ECLIPTIC_LON_DEG = 0.0
SUN_ECLIPTIC_LAT_DEG = 0.0

# IRSA observer settings.
OBSLOCIN = 0
OBSVERSION = 4

GRID_HALF_WIDTH_DEG = 65.0
GRID_STEP_DEG = 5.0

# Spectral grid: 8 to 12 microns, in 1-micron increments.
WAVELENGTHS_UM = np.arange(8.0, 12.01, 1)

OUTPUT_HDF5 = "irsa_background_ecliptic_8_12um.h5"
OUTPUT_PLOT = "irsa_background_ecliptic_8_12um_spectrum.png"

COMPONENTS = [
    "zodiacal_MJy_sr",
    "ISM_MJy_sr",
    "stars_MJy_sr",
    "CIB_MJy_sr",
    "total_MJy_sr",
]


def make_ecliptic_grid(
    centre_lon_deg,
    centre_lat_deg,
    half_width_deg=65.0,
    step_deg=5.0,
):
    """Create a rectangular Ecliptic J2000 longitude/latitude grid.

    Returns 2-D coordinate arrays and the corresponding 2-D
    Equatorial J2000 coordinates used for IRSA requests.
    """
    if step_deg <= 0:
        raise ValueError("step_deg must be positive.")
    if half_width_deg < 0:
        raise ValueError("half_width_deg cannot be negative.")
    if not -90.0 <= centre_lat_deg <= 90.0:
        raise ValueError("Ecliptic latitude must be between -90 and 90.")

    offsets = np.arange(-half_width_deg,half_width_deg + step_deg * 1e-9, step_deg, )

    lon_offsets, lat_offsets = np.meshgrid(offsets, offsets, indexing="ij")

    lon_grid = (centre_lon_deg + lon_offsets) % 360.0
    lat_grid = centre_lat_deg + lat_offsets

    valid = (lat_grid >= -90.0) & (lat_grid <= 90.0)
    if not np.all(valid):
        raise ValueError(
            "This implementation expects a complete rectangular grid; "
            "reduce the grid extent near an ecliptic pole."
        )

    # Define the grid in the mean ecliptic and equinox of J2000.
    ecliptic = SkyCoord(
        lon=lon_grid * u.deg,
        lat=lat_grid * u.deg,
        frame=BarycentricMeanEcliptic(equinox=Time("J2000")),
    )

    # IRSA Background Model's documented coordinate example uses
    # Equatorial J2000. Transform each ecliptic point before querying.
    equatorial = ecliptic.transform_to(FK5(equinox=Time("J2000")))

    return {
        "ecliptic_lon_deg": lon_grid,
        "ecliptic_lat_deg": lat_grid,
        "ra_deg": equatorial.ra.deg,
        "dec_deg": equatorial.dec.deg,
    }


def read_value(stats, tag):
    text = stats.findtext(tag)
    if text is None:
        raise ValueError(f"Missing '{tag}' in IRSA response.")
    return float(text.split()[0])


def query_irsa(session, ra_deg, dec_deg, wavelength_um):
    """Query IRSA using the Equatorial J2000 equivalent of a grid point."""
    params = {
        "locstr": f"{ra_deg:.8f} {dec_deg:.8f} equ J2000",
        "year": YEAR,
        "day": DAY_OF_YEAR,
        "obslocin": OBSLOCIN,
        "obsverin": OBSVERSION,
        "ido_viewin": 0,
        "wavelength": float(wavelength_um),
    }

    response = session.get(URL, params=params, timeout=45)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    if root.attrib.get("status") != "ok":
        raise RuntimeError(
            f"IRSA query failed for RA={ra_deg:.5f}°, "
            f"Dec={dec_deg:.5f}°, wavelength={wavelength_um:.2f} µm:\n"
            f"{response.text[:1000]}"
        )

    stats = root.find(".//statistics")
    if stats is None:
        raise ValueError(
            "No statistics element in IRSA response:\n"
            f"{response.text[:1000]}"
        )

    return [
        read_value(stats, "zody"),
        read_value(stats, "ism"),
        read_value(stats, "stars"),
        read_value(stats, "cib"),
        read_value(stats, "totbg"),
    ]


def calculate_summary(results):
    """Calculate statistics over both spatial axes.

    Input shape: (ecliptic_longitude, ecliptic_latitude,
                  wavelength, component)
    Output shape: (wavelength, component)
    """
    return {
        "mean": np.mean(results, axis=(0, 1)),
        "median": np.median(results, axis=(0, 1)),
        "min": np.min(results, axis=(0, 1)),
        "max": np.max(results, axis=(0, 1)),
    }


def save_results_hdf5(filename, grid, wavelengths, results, summary):
    """Save measurements, coordinates, summaries, and metadata to HDF5."""
    with h5py.File(filename, "w") as h5:
        h5.attrs["description"] = (
            "IRSA infrared background model on an Ecliptic J2000 grid"
        )
        h5.attrs["observation_date"] = OBSERVATION_DATE
        h5.attrs["year"] = YEAR
        h5.attrs["day_of_year"] = DAY_OF_YEAR
        h5.attrs["coordinate_system"] = "Ecliptic J2000"
        h5.attrs["query_coordinate_system"] = "Equatorial J2000 (FK5)"
        h5.attrs["grid_half_width_deg"] = GRID_HALF_WIDTH_DEG
        h5.attrs["grid_step_deg"] = GRID_STEP_DEG
        h5.attrs["observer_location_code"] = OBSLOCIN
        h5.attrs["irsa_model_version"] = OBSVERSION
        h5.attrs["data_units"] = "MJy/sr"
        h5.attrs["array_axis_order"] = (
            "ecliptic_longitude, ecliptic_latitude, wavelength, component"
        )

        coordinates = h5.create_group("coordinates")
        coordinates.create_dataset(
            "ecliptic_longitude_deg", data=grid["ecliptic_lon_deg"]
        )
        coordinates.create_dataset(
            "ecliptic_latitude_deg", data=grid["ecliptic_lat_deg"]
        )
        coordinates.create_dataset("ra_j2000_deg", data=grid["ra_deg"])
        coordinates.create_dataset("dec_j2000_deg", data=grid["dec_deg"])

        spectral = h5.create_group("spectral")
        spectral.create_dataset("wavelength_um", data=wavelengths)

        background = h5.create_group("background")
        string_type = h5py.string_dtype(encoding="utf-8")
        background.create_dataset(
            "component_names",
            data=np.asarray(COMPONENTS, dtype=object),
            dtype=string_type,
        )
        background.create_dataset(
            "values",
            data=results,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        background["values"].attrs["units"] = "MJy/sr"
        background["values"].attrs["axis_order"] = (
            "ecliptic_longitude, ecliptic_latitude, wavelength, component"
        )

        stats_group = h5.create_group("summary")
        for name, array in summary.items():
            stats_group.create_dataset(name, data=array)


def plot_spectrum(wavelengths, summary):
    component_indices = {
        name: index for index, name in enumerate(COMPONENTS)
    }
    mean_spectrum = summary["mean"]

    plt.figure(figsize=(9, 6))
    plt.plot(
        wavelengths,
        mean_spectrum[:, component_indices["zodiacal_MJy_sr"]],
        marker="o",
        label="Mean zodiacal background",
    )
    plt.plot(
        wavelengths,
        mean_spectrum[:, component_indices["total_MJy_sr"]],
        marker="s",
        label="Mean total background",
    )
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Spectral radiance (MJy/sr)")
    plt.title("IRSA background: Ecliptic J2000 grid")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.show()


def main():
    grid = make_ecliptic_grid(
        SUN_ECLIPTIC_LON_DEG,
        SUN_ECLIPTIC_LAT_DEG,
        half_width_deg=GRID_HALF_WIDTH_DEG,
        step_deg=GRID_STEP_DEG,
    )

    n_lon, n_lat = grid["ecliptic_lon_deg"].shape
    n_wavelengths = len(WAVELENGTHS_UM)
    n_components = len(COMPONENTS)
    total_requests = n_lon * n_lat * n_wavelengths

    print("=" * 64)
    print("IRSA Background Model — Ecliptic J2000 grid")
    print("=" * 64)
    print(f"Observation date: {OBSERVATION_DATE} (day {DAY_OF_YEAR})")
    print(
        "Ecliptic grid centre: "
        f"longitude={SUN_ECLIPTIC_LON_DEG:.2f}°, "
        f"latitude={SUN_ECLIPTIC_LAT_DEG:.2f}°"
    )
    print(f"Grid: {n_lon} longitudes x {n_lat} latitudes")
    print(f"Grid step: {GRID_STEP_DEG:.1f}°")
    print(f"Wavelengths: {WAVELENGTHS_UM.tolist()} µm")
    print(f"Total IRSA requests: {total_requests}")
    print("IRSA requests are sent as transformed Equatorial J2000 coordinates.")
    print("=" * 64)

    # Four-dimensional array:
    # (ecliptic longitude, ecliptic latitude, wavelength, component)
    results = np.empty(
        (n_lon, n_lat, n_wavelengths, n_components),
        dtype=np.float64,
    )

    start_time = time.time()
    completed = 0

    with requests.Session() as session:
        for lon_index in range(n_lon):
            for lat_index in range(n_lat):
                ra_deg = grid["ra_deg"][lon_index, lat_index]
                dec_deg = grid["dec_deg"][lon_index, lat_index]

                for wavelength_index, wavelength_um in enumerate(
                    WAVELENGTHS_UM
                ):
                    results[
                        lon_index, lat_index, wavelength_index, :
                    ] = query_irsa(
                        session, ra_deg, dec_deg, wavelength_um
                    )
                    completed += 1
                    time.sleep(0.15)

                elapsed = time.time() - start_time
                print(
                    f"Grid point {lon_index * n_lat + lat_index + 1:4d}/"
                    f"{n_lon * n_lat} | "
                    f"ecl lon={grid['ecliptic_lon_deg'][lon_index, lat_index]:7.2f}° | "
                    f"ecl lat={grid['ecliptic_lat_deg'][lon_index, lat_index]:7.2f}° | "
                    f"requests={completed}/{total_requests} | "
                    f"elapsed={elapsed:.1f}s"
                )

    print(
        f"\nAll IRSA queries completed in "
        f"{time.time() - start_time:.1f} seconds."
    )

    summary = calculate_summary(results)
    save_results_hdf5(
        OUTPUT_HDF5, grid, WAVELENGTHS_UM, results, summary
    )
    print(f"Saved multidimensional dataset to: {OUTPUT_HDF5}")
    print(f"Array shape: {results.shape}")
    print(
        "Axis order: (ecliptic longitude, ecliptic latitude, "
        "wavelength, component)"
    )

    plot_spectrum(WAVELENGTHS_UM, summary)


if __name__ == "__main__":
    main()
