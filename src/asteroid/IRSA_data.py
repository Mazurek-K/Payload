import time
import csv
import requests
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

URL = (
    "https://irsa.ipac.caltech.edu/"
    "cgi-bin/BackgroundModel/nph-bgmodel"
)

# Observation date: 21 March 2026
YEAR = 2026
DAY_OF_YEAR = 80

# Approximate Sun coordinates near the March equinox.
# These are illustrative, NOT precise ephemeris coordinates.
SUN_RA_DEG = 0.0
SUN_DEC_DEG = 0.0

# IRSA observer settings.
# Verify these values against the IRSA documentation.
OBSLOCIN = 0
OBSVERSION = 4

# Grid extends +/-45 degrees from the assumed Sun coordinates.
GRID_HALF_WIDTH_DEG = 65.0
GRID_STEP_DEG = 5.0

# Spectral grid: 8 to 12 microns, in 0.5-micron increments.
WAVELENGTHS_UM = np.arange(8.0, 12.01, 1)

# Output files.
OUTPUT_CSV = "irsa_background_8_12um_grid_new.csv"
OUTPUT_SUMMARY_CSV = "irsa_background_8_12um_summary.csv"
OUTPUT_PLOT = "irsa_background_8_12um_spectrum.png"

# Background components returned by IRSA.
COMPONENTS = [
    "zodiacal_MJy_sr",
    "ISM_MJy_sr",
    "stars_MJy_sr",
    "CIB_MJy_sr",
    "total_MJy_sr",
]

def sun_centred_grid(
    sun_ra_deg,
    sun_dec_deg,
    half_width_deg=45.0,
    step_deg=5.0,
):
    """
    Generate a square grid in equatorial RA and Dec.

    RA and Dec each extend +/- half_width_deg from the
    supplied central coordinates.

    RA is wrapped into [0, 360) degrees.

    Returns a dictionary of NumPy arrays:
        ra_deg
        dec_deg

    Note:
        This is a coordinate grid, not a grid with uniform
        angular spacing on the celestial sphere.
    """

    if step_deg <= 0:
        raise ValueError("step_deg must be positive.")

    if half_width_deg < 0:
        raise ValueError("half_width_deg cannot be negative.")

    if not -90.0 <= sun_dec_deg <= 90.0:
        raise ValueError("sun_dec_deg must be between -90 and 90.")

    # Include the upper endpoint when it falls on the grid.
    offsets = np.arange(
        -half_width_deg,
        half_width_deg + step_deg * 1e-9,
        step_deg,
    )

    ra_offsets, dec_offsets = np.meshgrid(
        offsets,
        offsets,
        indexing="xy",
    )

    # Offset coordinates from the central position.
    ra_grid = (sun_ra_deg + ra_offsets) % 360.0
    dec_grid = sun_dec_deg + dec_offsets

    # Discard declinations outside the physical range.
    valid = (
        (dec_grid >= -90.0)
        & (dec_grid <= 90.0)
    )

    return {
        "ra_deg": ra_grid[valid].ravel(),
        "dec_deg": dec_grid[valid].ravel(),
    }


# ============================================================
# Parse IRSA response
# ============================================================

def read_value(stats, tag):
    """
    Read a numerical value from an XML statistics element.
    """

    text = stats.findtext(tag)

    if text is None:
        raise ValueError(
            f"Missing '{tag}' in IRSA response."
        )

    return float(text.split()[0])


def query_irsa(session, ra_deg, dec_deg, wavelength_um):
    """
    Query IRSA for one sky direction and one wavelength.
    """

    params = {
        "locstr": (
            f"{ra_deg:.6f} {dec_deg:.6f} equ J2000"
        ),
        "year": YEAR,
        "day": DAY_OF_YEAR,
        "obslocin": OBSLOCIN,
        "obsverin": OBSVERSION,
        "ido_viewin": 0,
        "wavelength": float(wavelength_um),
    }

    response = session.get(
        URL,
        params=params,
        timeout=45,
    )

    response.raise_for_status()

    root = ET.fromstring(response.text)

    if root.attrib.get("status") != "ok":
        raise RuntimeError(
            f"IRSA query failed for "
            f"RA={ra_deg:.3f}°, "
            f"Dec={dec_deg:.3f}°, "
            f"wavelength={wavelength_um:.1f} µm:\n"
            f"{response.text[:1000]}"
        )

    stats = root.find(".//statistics")

    if stats is None:
        raise ValueError(
            "No statistics element in IRSA response:\n"
            f"{response.text[:1000]}"
        )

    return {
        "zodiacal_MJy_sr": read_value(stats, "zody"),
        "ISM_MJy_sr": read_value(stats, "ism"),
        "stars_MJy_sr": read_value(stats, "stars"),
        "CIB_MJy_sr": read_value(stats, "cib"),
        "total_MJy_sr": read_value(stats, "totbg"),
    }


# ============================================================
# Save individual measurements
# ============================================================

def save_results_csv(filename, grid, wavelengths, results):
    """
    Save all measurements to CSV.

    results shape:
        (number_of_directions,
         number_of_wavelengths,
         number_of_components)
    """
    fieldnames = [
        "ra_deg",
        "dec_deg",
        "wavelength_um",
        *COMPONENTS,
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        n_directions = len(grid["ra_deg"])

        for direction_index in range(n_directions):
            for wavelength_index, wavelength in enumerate(
                wavelengths
            ):

                row = {
                    "ra_deg": float(
                        grid["ra_deg"][direction_index]
                    ),
                    "dec_deg": float(
                        grid["dec_deg"][direction_index]
                    ),
                    "wavelength_um": float(wavelength),
                }

                for component_index, component in enumerate(
                    COMPONENTS
                ):
                    row[component] = float(
                        results[
                            direction_index,
                            wavelength_index,
                            component_index,
                        ]
                    )

                writer.writerow(row)


# ============================================================
# Statistical summary using NumPy
# ============================================================

def calculate_summary(results):
    """
    Calculate statistics across sky directions.

    Input shape:
        (directions, wavelengths, components)

    Output arrays have shape:
        (wavelengths, components)
    """

    return {
        "mean": np.mean(results, axis=0),
        "median": np.median(results, axis=0),
        "min": np.min(results, axis=0),
        "max": np.max(results, axis=0),
    }


# ============================================================
# Plot mean spectrum
# ============================================================

def plot_spectrum(wavelengths, summary):
    """
    Plot the mean zodiacal and total background spectra
    across all sampled sky directions.
    """

    component_indices = {
        component: index
        for index, component in enumerate(COMPONENTS)
    }

    mean_spectrum = summary["mean"]

    zodiacal_mean = mean_spectrum[
        :,
        component_indices["zodiacal_MJy_sr"],
    ]

    total_mean = mean_spectrum[
        :,
        component_indices["total_MJy_sr"],
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        wavelengths,
        zodiacal_mean,
        marker="o",
        label="Mean zodiacal background",
    )

    plt.plot(
        wavelengths,
        total_mean,
        marker="s",
        label="Mean total background",
    )

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Spectral radiance (MJy/sr)")

    plt.title(
        "IRSA background: square RA/Dec grid "
        "around the approximate Sun position"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=200)

    print(f"Saved spectrum plot to: {OUTPUT_PLOT}")

    plt.show()



def main():
    grid = sun_centred_grid(
        SUN_RA_DEG,
        SUN_DEC_DEG,
        half_width_deg=GRID_HALF_WIDTH_DEG,
        step_deg=GRID_STEP_DEG,
    )

    n_directions = len(grid["ra_deg"])
    n_wavelengths = len(WAVELENGTHS_UM)
    n_components = len(COMPONENTS)

    total_requests = n_directions * n_wavelengths

    print("=" * 60)
    print("IRSA Background Model")
    print("=" * 60)

    print(f"Observation date: 21 March {YEAR}")
    print(
        "Central coordinates: "
        f"RA={SUN_RA_DEG:.2f}°, "
        f"Dec={SUN_DEC_DEG:.2f}°"
    )

    print(
        "Grid half-width: "
        f"{GRID_HALF_WIDTH_DEG:.1f}°"
    )

    print(f"Grid step: {GRID_STEP_DEG:.1f}°")
    print(f"Sky directions: {n_directions}")
    print(f"Wavelengths: {n_wavelengths}")
    print(f"Total IRSA requests: {total_requests}")

    print(
        f"RA range: {SUN_RA_DEG - GRID_HALF_WIDTH_DEG:.1f}° "
        f"to {SUN_RA_DEG + GRID_HALF_WIDTH_DEG:.1f}° "
        "(wrapped to [0, 360))"
    )

    print(
        f"Dec range: "
        f"{SUN_DEC_DEG - GRID_HALF_WIDTH_DEG:.1f}° "
        f"to {SUN_DEC_DEG + GRID_HALF_WIDTH_DEG:.1f}°"
    )

    print("=" * 60)

    results = np.empty(
        (
            n_directions,
            n_wavelengths,
            n_components,
        ),
        dtype=np.float64,
    )

    start_time = time.time()
    completed_requests = 0

    with requests.Session() as session:

        for direction_index in range(n_directions):

            ra_deg = grid["ra_deg"][direction_index]
            dec_deg = grid["dec_deg"][direction_index]

            for wavelength_index, wavelength_um in enumerate(
                WAVELENGTHS_UM
            ):

                values = query_irsa(
                    session,
                    ra_deg,
                    dec_deg,
                    wavelength_um,
                )

                results[
                    direction_index,
                    wavelength_index,
                    :,
                ] = [
                    values[component]
                    for component in COMPONENTS
                ]

                completed_requests += 1
                # Be polite to the service.
                time.sleep(0.15)

            elapsed = time.time() - start_time
            print(
                f"Direction {direction_index + 1:3d}/"
                f"{n_directions} | "
                f"RA={ra_deg:7.2f}° | "
                f"Dec={dec_deg:6.2f}° | "
                f"Requests={completed_requests}/{total_requests} | "
                f"Elapsed={elapsed:.1f}s"
            )

    print(
        f"\nAll IRSA queries completed in "
        f"{time.time() - start_time:.1f} seconds."
    )

    # --------------------------------------------------------
    # Save individual measurements
    # --------------------------------------------------------

    save_results_csv(
        OUTPUT_CSV,
        grid,
        WAVELENGTHS_UM,
        results,
    )

    print(f"Saved measurements to: {OUTPUT_CSV}")

    # --------------------------------------------------------
    # Calculate and save spectral summary
    # --------------------------------------------------------
    summary = calculate_summary(results)
    print(f"Saved summary to: {OUTPUT_SUMMARY_CSV}")
    plot_spectrum(
        WAVELENGTHS_UM,
        summary,
    )


if __name__ == "__main__":
    main()
