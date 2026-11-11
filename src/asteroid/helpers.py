import numpy as np
import matplotlib.pyplot as plt
import h5py

from config import ( WAVELENGTH_UM, APERTURE, PIXEL_PITCH, HDF5_FILE, FOCAL_LENGTH)


def airy_disk():
    d_airy = 2 * 1.22e-6 * WAVELENGTH_UM[0] / APERTURE
    d_airy_focal = FOCAL_LENGTH * d_airy

    n_pix = np.pi * d_airy_focal**2 / (4 * PIXEL_PITCH**2)

    return n_pix, d_airy


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