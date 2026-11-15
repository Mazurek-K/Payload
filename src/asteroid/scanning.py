import numpy as np
import matplotlib.pyplot as plt
from config import (HDF5_FILE)
import h5py
from scipy.interpolate import RegularGridInterpolator
from helpers import plot_interpolated_map
from matplotlib.path import Path
np.set_printoptions(linewidth=200)

def translate(coords, vector):
    return np.array(coords) + np.array(vector)

def rotate(coords, angle):
    theta = np.radians(angle)  # angle in degrees
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return np.array(coords) @ R.T

def scan_pattern(fov1, fov2, radius, overlap=0.2):

    dimensions = np.array([
        [fov1/2, -fov2/2],
        [fov1/2,  fov2/2],
        [-fov1/2, fov2/2],
        [-fov1/2, -fov2/2]
    ])

    radii = [radius, radius + fov2, radius + 2*fov2]

    all_images = []
    pointings_per_level = []

    plt.figure()

    for r in radii:

        angle = np.degrees(2 * np.arctan(fov1 / (2 * r)))
        angle_step = angle * (1 - overlap)
        n = int(np.ceil(360 / angle_step))
        pointings_per_level.append(n)

        for i in range(n):

            image = rotate(
                translate(dimensions, [0, r + fov2/2]),
                i * 360/n
            )

            all_images.append(image)

            closed_image = np.vstack([image, image[0]])
            plt.plot(closed_image[:, 0], closed_image[:, 1], color='blue')

        print(f"Radius: {r}°")
        print(f"Number of pointings: {n}")
        print(f"Angular step: {360/n:.2f}°\n")

        circle = plt.Circle((0, 0), r, fill=False, color='red')
        plt.gca().add_patch(circle)

    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

    all_images = np.array(all_images)

    print(f"Total number of pointings: {len(all_images)}")
    print(all_images)
    return all_images, pointings_per_level


def exposure_times(time_map):
    with h5py.File(HDF5_FILE, "r") as file:
        lon = file["coordinates/ecliptic_longitude_deg"][:]
        lat = file["coordinates/ecliptic_latitude_deg"][:]

    lon_1d = lon[:, 0].astype(float)
    lat_1d = lat[0, :].astype(float)

    # Unwrap longitude (only if the original values cross 360 -> 0)
    lon_1d = np.rad2deg(np.unwrap(np.deg2rad(lon_1d)))
    lon_1d = (lon_1d + 180) % 360 - 180
    lon_order = np.argsort(lon_1d)
    lat_order = np.argsort(lat_1d)
    lon_1d = lon_1d[lon_order]
    lat_1d = lat_1d[lat_order]

    lon_fine = np.linspace(lon_1d[0], lon_1d[-1], 270)
    lat_fine = np.linspace(lat_1d[0], lat_1d[-1], 270)

    xv, yv = np.meshgrid(lon_fine, lat_fine)
    time_map = time_map[np.ix_(lon_order, lat_order)]

    interpolator = RegularGridInterpolator(
        (lon_1d, lat_1d),
        time_map,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    points = np.column_stack((xv.ravel(), yv.ravel()))
    time_map_fine = interpolator(points).reshape(xv.shape)
    plot_interpolated_map(xv,yv, time_map_fine)




    return time_map_fine



