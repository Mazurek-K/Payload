import csv
import numpy as np
import matplotlib.pyplot as plt


CSV_FILE = "../outputs/irsa_background_8_12um_grid.csv"
sr_to_arcsec2 = (648000/np.pi)**2

# Select wavelength in microns
SELECTED_WAVELENGTH_UM = 8.0

# Select component:
# "zodiacal_MJy_sr"
# "ISM_MJy_sr"
# "stars_MJy_sr"
# "CIB_MJy_sr"
# "total_MJy_sr"
COMPONENT = "total_MJy_sr"

# -----------------------------
# Read CSV
# -----------------------------
ra_values = []
dec_values = []
intensity_values = []
wavelength_values = []

with open(CSV_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    required_columns = {
        "ra_deg",
        "dec_deg",
        "wavelength_um",
        COMPONENT,
    }

    missing = required_columns - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"Missing CSV columns: {', '.join(sorted(missing))}"
        )

    for row in reader:
        wavelength = float(row["wavelength_um"])

        if np.isclose(
            wavelength,
            SELECTED_WAVELENGTH_UM,
            atol=1e-6,
            rtol=0
        ):
            ra_values.append(float(row["ra_deg"]))
            dec_values.append(float(row["dec_deg"]))
            intensity_values.append(float(row[COMPONENT]))
            wavelength_values.append(wavelength)

if not intensity_values:
    raise ValueError(
        f"No rows found for wavelength "
        f"{SELECTED_WAVELENGTH_UM} µm."
    )

ra = np.asarray(ra_values)
dec = np.asarray(dec_values)
intensity = np.asarray(intensity_values)

# Centre the RA axis on the circular mean of the data
ra_rad = np.deg2rad(ra)
centre_ra = np.rad2deg(
    np.arctan2(
        np.mean(np.sin(ra_rad)),
        np.mean(np.cos(ra_rad))
    )
) % 360.0
# Wrap RA differences into the range [-180, 180)
ra_plot = ((ra - centre_ra + 180.0) % 360.0) - 180.0

threshold = 100.0
mask = intensity < threshold
intensity = intensity[mask]
ra_plot = ra_plot[mask]
dec = dec[mask]
intensity_arcsec2 = intensity*sr_to_arcsec2

# -----------------------------
# RA x Dec scatter plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(
    ra_plot,
    dec,
    c=intensity_arcsec2,
    cmap="viridis",
    s=65,
    marker="s"
)
colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label(
    f"{COMPONENT} (MJy/arcsec2)"
)

ax.set_xlabel("Right Ascension (degrees)")
ax.set_ylabel("Declination (degrees)")

ax.set_title(
    f"IRSA Background at {SELECTED_WAVELENGTH_UM:.1f} µm\n"
    f"{COMPONENT}"
)

ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#######################################################################

# This assumes the RA/Dec points form a regular rectangular grid.
ra_unique = np.sort(np.unique(ra_plot))
dec_unique = np.sort(np.unique(dec))
RA, DEC = np.meshgrid(ra_unique, dec_unique)
intensity_grid = np.full(RA.shape, np.nan)

for r, d, value in zip(ra_plot, dec, intensity_arcsec2):
    i = np.argmin(np.abs(ra_unique - r))
    j = np.argmin(np.abs(dec_unique - d))
    intensity_grid[j, i] = value

fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(
    RA,
    DEC,
    intensity_grid,
    levels=20,
    cmap="viridis"
)

lines = ax.contour(
    RA,
    DEC,
    intensity_grid,
    levels=10,
    colors="black",
    linewidths=0.6,
    alpha=0.6
)

ax.clabel(
    lines,
    inline=True,
    fontsize=8,
    fmt="%.2g"
)

# Colorbar.
colorbar = plt.colorbar(contour, ax=ax)
colorbar.set_label(f"{COMPONENT} (MJy/arcsec²)")

# Axis labels and title.
# ax.set_xlabel("Right Ascension (degrees)")
# ax.set_ylabel("Declination (degrees)")

ax.set_xlabel(
    "Right Ascension (degrees)",
    fontsize=20,
)

ax.set_ylabel(
    "Declination (degrees)",
    fontsize=20,
)
ax.set_title(
    f"IRSA Background at {SELECTED_WAVELENGTH_UM:.1f} µm\n"
    f"{COMPONENT}",
    fontsize = 20
)

ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()