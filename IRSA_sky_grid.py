import time
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# IRSA Background Model service
URL = "https://irsa.ipac.caltech.edu/cgi-bin/BackgroundModel/nph-bgmodel"

# ------------------------------------------------------------
# Observation setup
# ------------------------------------------------------------
# Approximate spring-equinox geometry:
# the Sun is near RA=0 deg, Dec=0 deg around the March equinox.
# This is a convenient coordinate-system test, NOT an accurate
# ephemeris. Update these values for a real observation date.
YEAR = 2026
DAY_OF_YEAR = 80  # approximately 21 March 2026
SUN_RA_DEG = 0.0
SUN_DEC_DEG = 0.0

# Provisional Earth observer setting. This is not an L1-specific
# geometry; use the IRSA-supported observer option appropriate to
# your needs and verify its definition in the IRSA documentation.
OBSLOCIN = 0
OBSVERSION = 4

# Sky grid: elongation from 20 to 40 degrees, all around the Sun.
ELONGATIONS_DEG = np.arange(20.0, 40.1, 5.0)
POSITION_ANGLES_DEG = np.arange(0.0, 360.0, 30.0)

# Coarse spectral grid for initial testing.
WAVELENGTHS_UM = np.arange(8.0, 12.01, 0.5)

OUTPUT_CSV = "irsa_background_8_12um_grid.csv"
OUTPUT_SUMMARY_CSV = "irsa_background_8_12um_summary.csv"


def sun_centred_grid(sun_ra_deg, sun_dec_deg, elongations_deg, position_angles_deg):
    """Convert Sun-centred angular offsets into equatorial RA/Dec."""
    ra0 = np.radians(sun_ra_deg)
    dec0 = np.radians(sun_dec_deg)

    rows = []
    for elongation in elongations_deg:
        sep = np.radians(elongation)
        for pa_deg in position_angles_deg:
            pa = np.radians(pa_deg)

            dec = np.arcsin(
                np.sin(dec0) * np.cos(sep)
                + np.cos(dec0) * np.sin(sep) * np.cos(pa)
            )
            ra = ra0 + np.arctan2(
                np.sin(pa) * np.sin(sep) * np.cos(dec0),
                np.cos(sep) - np.sin(dec0) * np.sin(dec)
            )

            rows.append({
                "elongation_deg": float(elongation),
                "position_angle_deg": float(pa_deg),
                "ra_deg": float(np.degrees(ra) % 360.0),
                "dec_deg": float(np.degrees(dec)),
            })
    return pd.DataFrame(rows)


def read_value(stats, tag):
    text = stats.findtext(tag)
    if text is None:
        raise ValueError(f"Missing '{tag}' in IRSA response.")
    return float(text.split()[0])


def query_irsa(session, ra_deg, dec_deg, wavelength_um):
    params = {
        "locstr": f"{ra_deg:.6f} {dec_deg:.6f} equ J2000",
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
            f"IRSA query failed for RA={ra_deg:.3f}, Dec={dec_deg:.3f}, "
            f"wavelength={wavelength_um:.1f} um:\n{response.text[:1000]}"
        )

    stats = root.find(".//statistics")
    if stats is None:
        raise ValueError(f"No statistics element in response:\n{response.text[:1000]}")

    return {
        "zodiacal_MJy_sr": read_value(stats, "zody"),
        "ISM_MJy_sr": read_value(stats, "ism"),
        "stars_MJy_sr": read_value(stats, "stars"),
        "CIB_MJy_sr": read_value(stats, "cib"),
        "total_MJy_sr": read_value(stats, "totbg"),
    }


def main():
    grid = sun_centred_grid(
        SUN_RA_DEG, SUN_DEC_DEG,
        ELONGATIONS_DEG, POSITION_ANGLES_DEG
    )
    print(f"Sky directions: {len(grid)}")
    print(f"Wavelengths: {len(WAVELENGTHS_UM)}")
    print(f"Total IRSA requests: {len(grid) * len(WAVELENGTHS_UM)}")

    results = []
    with requests.Session() as session:
        for direction_index, direction in grid.iterrows():
            for wavelength_um in WAVELENGTHS_UM:
                values = query_irsa(
                    session,
                    direction["ra_deg"],
                    direction["dec_deg"],
                    wavelength_um,
                )
                results.append({
                    **direction.to_dict(),
                    "wavelength_um": float(wavelength_um),
                    **values,
                })
                # Be polite to the service; increase this if requests are throttled.
                time.sleep(0.15)

            print(
                f"Completed direction {direction_index + 1}/{len(grid)} "
                f"(elongation={direction['elongation_deg']:.0f} deg, "
                f"PA={direction['position_angle_deg']:.0f} deg)"
            )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved all results to: {OUTPUT_CSV}")

    # Mean spectrum across sampled directions (not a full-sky average).
    summary = (
        df.groupby("wavelength_um")[
            ["zodiacal_MJy_sr", "ISM_MJy_sr", "stars_MJy_sr",
             "CIB_MJy_sr", "total_MJy_sr"]
        ]
        .agg(["mean", "median", "min", "max"])
    )
    summary.columns = [f"{component}_{stat}" for component, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"Saved spectral summary to: {OUTPUT_SUMMARY_CSV}")

    # Plot mean spectrum over the sampled directions.
    means = df.groupby("wavelength_um").mean(numeric_only=True)
    plt.plot(
        means.index, means["zodiacal_MJy_sr"],
        marker="o", label="Mean zodiacal background"
    )
    plt.plot(
        means.index, means["total_MJy_sr"],
        marker="s", label="Mean total background"
    )
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Spectral radiance (MJy/sr)")
    plt.title("IRSA background, sampled 20°–40° around the Sun")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("irsa_background_8_12um_spectrum.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()