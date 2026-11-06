import h5py
import numpy as np
import matplotlib.pyplot as plt
from config import SR_TO_ARCSEC2

HDF5_FILE = "../outputs/irsa_background_ecliptic_8_12um.h5"

# Select wavelength in microns
SELECTED_WAVELENGTH_UM = 8.0

# Select component:
# "zodiacal_MJy_sr"
# "ISM_MJy_sr"
# "stars_MJy_sr"
# "CIB_MJy_sr"
# "total_MJy_sr"
COMPONENT = "total_MJy_sr"

# Ecliptic longitude around which the plot is centred
CENTRE_LON_DEG = 0.0

# Intensity threshold in MJy/sr
THRESHOLD = 100.0

# Conversion: square arcseconds in one steradian
sr_to_arcsec2 = SR_TO_ARCSEC2

# -------------------------------------------------
# Read HDF5
# --------------------------------------------------

with h5py.File(HDF5_FILE, "r") as file:

    lon_grid = file["coordinates/ecliptic_longitude_deg"][:]
    lat_grid = file["coordinates/ecliptic_latitude_deg"][:]
    wavelengths = file["spectral/wavelength_um"][:]
    component_names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in file["background/component_names"][:]
    ]
    # Shape:
    # (ecliptic longitude, ecliptic latitude,
    #  wavelength, component)
    results = file["background/values"][:]

# --------------------------------------------------
# Select component
# --------------------------------------------------

if COMPONENT not in component_names:
    raise ValueError(
        f"Component '{COMPONENT}' not found.\n"
        f"Available components: {component_names}"
    )

component_index = component_names.index(COMPONENT)

# --------------------------------------------------
# Select wavelength
# --------------------------------------------------
wavelength_matches = np.where(np.isclose( wavelengths, SELECTED_WAVELENGTH_UM, atol=1e-6, rtol=0,))[0]

if len(wavelength_matches) == 0:
    raise ValueError(
        f"Wavelength {SELECTED_WAVELENGTH_UM} µm " "not found in the HDF5 file.")

wavelength_index = wavelength_matches[0]
# --------------------------------------------------
# Extract 2D background map
# --------------------------------------------------

# Shape:
# (ecliptic longitude, ecliptic latitude)
intensity = results[:, :, wavelength_index, component_index]

# Convert the longitude coordinates to offsets around
# the selected central longitude. This avoids a jump
# from 295 degrees to 0 degrees in the plot.
lon_plot = ((lon_grid - CENTRE_LON_DEG + 180.0) % 360.0) - 180.0

# Apply intensity threshold
mask = intensity < THRESHOLD
intensity_masked = np.where(mask, intensity, np.nan)

# Convert MJy/sr to MJy/arcsec^2
intensity_arcsec2 = intensity_masked / sr_to_arcsec2
intensity_arcsec2_mili = intensity_arcsec2* 1e9

# --------------------------------------------------
# Prepare coordinates for plotting
# --------------------------------------------------

# The HDF5 grid has axis order:
# (longitude, latitude)
#
# Transpose the arrays for plotting with latitude
# on the vertical axis and longitude on the horizontal.

LON = lon_plot
LAT = lat_grid

# Meshgrid for contour plotting
LON_mesh, LAT_mesh = np.meshgrid(LON[:, 0],LAT[0, :],indexing="ij",)

# --------------------------------------------------
# 1. Ecliptic longitude x latitude scatter plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(
    lon_plot.flatten(),
    lat_grid.flatten(),
    c=intensity_arcsec2_mili.flatten(),
    cmap="viridis",
    s=65,
    marker="s",
)

colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label(
    f"(mJy/arcsec²)"
)

ax.set_xlabel("Ecliptic longitude offset (degrees)")
ax.set_ylabel("Ecliptic latitude (degrees)")

ax.set_title(
    f"IRSA Background at {SELECTED_WAVELENGTH_UM:.1f} µm\n"

)

ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Filled contour plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

contour = ax.contourf(
    LON_mesh,
    LAT_mesh,
    intensity_arcsec2_mili,
    levels=20,
    cmap="viridis",
)

lines = ax.contour(
    LON_mesh,
    LAT_mesh,
    intensity_arcsec2_mili,
    levels=10,
    colors="black",
    linewidths=0.6,
    alpha=0.6,
)

ax.clabel(
    lines,
    inline=True,
    fontsize=8,
    fmt="%.2g",
)

# Colorbar
colorbar = plt.colorbar(contour, ax=ax)
colorbar.set_label( f"(mJy/arcsec²)")

# Axis labels
ax.set_xlabel("Ecliptic longitude offset (degrees)",fontsize=20,)

ax.set_ylabel( "Ecliptic latitude (degrees)",fontsize=20,)

ax.set_title( f"IRSA Background at {SELECTED_WAVELENGTH_UM:.1f} µm\n"
    f"",fontsize=20,)

ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()