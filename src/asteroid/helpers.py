import numpy as np
import matplotlib.pyplot as plt
import h5py
from config import ( WAVELENGTH_UM, APERTURE, PIXEL_PITCH, HDF5_FILE, FOCAL_LENGTH)
from scipy.integrate import quad
from matplotlib.patches import Circle


def airy_disk():
    d_airy = 2 * 1.22e-6 * WAVELENGTH_UM / APERTURE
    d_airy_focal = FOCAL_LENGTH * d_airy
    n_pix = np.pi * d_airy_focal**2 / (4 * PIXEL_PITCH**2)

    return n_pix, d_airy
def circle_pixel_area(x0, y0, pixel_pitch, radius):
    """
    Calculate the intersection area between a square pixel
    and a circle centred at (0, 0).
    """

    x1 = x0 - pixel_pitch / 2
    x2 = x0 + pixel_pitch / 2
    y1 = y0 - pixel_pitch / 2
    y2 = y0 + pixel_pitch / 2

    # Clip integration range to circle
    xa = max(x1, -radius)
    xb = min(x2, radius)

    if xa >= xb:
        return 0.0

    def vertical_intersection(x):
        y_circle = np.sqrt(max(0, radius**2 - x**2))

        upper = min(y2, y_circle)
        lower = max(y1, -y_circle)

        return max(0, upper - lower)

    area, _ = quad(vertical_intersection, xa, xb, epsabs=1e-12)

    return area


def airy_disk2D(plot = True):

    hs = []

    # Maximum grid size, based on the largest wavelength
    d_airy_max = 2 * 1.22e-6 * np.max(WAVELENGTH_UM)
    d_airy_focal_max = FOCAL_LENGTH * d_airy_max

    n_pix = int(np.ceil(d_airy_focal_max / PIXEL_PITCH))
    n_pix = n_pix + 1 if n_pix % 2 == 0 else n_pix

    pix_middle = n_pix // 2

    # Grid coordinates (same for all wavelengths)
    x = (np.arange(n_pix) - pix_middle) * PIXEL_PITCH
    y = (np.arange(n_pix) - pix_middle) * PIXEL_PITCH

    for wavelength in WAVELENGTH_UM:

        # Calculate radius for this wavelength
        d_airy = 2 * 1.22e-6 * wavelength
        d_airy_focal = FOCAL_LENGTH * d_airy
        radius = d_airy_focal / 2

        pixel_array = np.zeros((n_pix, n_pix))

        # Calculate fractional pixel coverage
        for i in range(n_pix):
            for j in range(n_pix):

                area = circle_pixel_area(
                    x[j], y[i], PIXEL_PITCH, radius
                )

                pixel_array[i, j] = area / PIXEL_PITCH**2

        pixel_array = pixel_array/(np.sum(pixel_array))
        hs.append(pixel_array)

        if plot:
            # Plot
            fig, ax = plt.subplots()

            ax.imshow(
                pixel_array,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                vmin=0,
                vmax=np.max(pixel_array)
            )

            ax.set_xticks(np.arange(-0.5, n_pix, 1))
            ax.set_yticks(np.arange(-0.5, n_pix, 1))
            ax.grid(color="red", linewidth=0.5)


            circle = Circle(
                (pix_middle, pix_middle),
                radius / PIXEL_PITCH,
                fill=False,
                edgecolor="blue",
                linewidth=1.5
            )

            ax.add_patch(circle)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            ax.set_title(f"Airy disk, λ = {wavelength:.1f} µm")

            plt.show()

    return hs


def plot_lat_long(data, title="-", mask=False, cmap="viridis"):

    with h5py.File(HDF5_FILE, "r") as file:
        lon = file["coordinates/ecliptic_longitude_deg"][:]
        lat = file["coordinates/ecliptic_latitude_deg"][:]

    lon = lon[:, 0].astype(float)
    lat = lat[0, :].astype(float)

    lon = (np.rad2deg(np.unwrap(np.deg2rad(lon))) + 180) % 360 - 180

    lon_order = np.argsort(lon)
    lat_order = np.argsort(lat)

    lon = lon[lon_order]
    lat = lat[lat_order]
    data = data[lon_order, :][:, lat_order]

    if mask :
        mask = (np.sqrt(lon[:, None]**2 + lat[None, :]**2) > 25)
        data =  np.where(mask, data, np.nan)

    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(lon, lat, data.T, shading="auto", cmap="inferno")
    plt.colorbar(mesh, label=title)
    plt.xlabel("Ecliptic longitude (deg)")
    plt.ylabel("Ecliptic latitude (deg)")
    plt.tight_layout()
    plt.show()