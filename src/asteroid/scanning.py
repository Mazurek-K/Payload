import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

from config import (HDF5_FILE)
import h5py
from scipy.interpolate import RegularGridInterpolator
from helpers import plot_interpolated_map
from matplotlib.path import Path
np.set_printoptions(linewidth=200)

from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np

def translate(coords, vector):
    return np.array(coords) + np.array(vector)

def rotate(coords, angle):
    theta = np.radians(angle)  # angle in degrees
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return np.array(coords) @ R.T

def scan_pattern(fov1, fov2, radius, overlap=0.2, n_radii = 4):

    dimensions = np.array([
        [fov1/2, -fov2/2],
        [fov1/2,  fov2/2],
        [-fov1/2, fov2/2],
        [-fov1/2, -fov2/2]
    ])

    radii = []
    for i in range(0, n_radii):
        radii.append(radius+i*fov2)
    # radii = [radius, radius + fov2, radius + 2*fov2, radius+3*fov2]

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


        circle = plt.Circle((0, 0), r, fill=False, color='red')
        plt.gca().add_patch(circle)

    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

    all_images = np.array(all_images)

    return all_images


def exposure_frames(time_map, fov1, fov2, min_angle = 25, plot = True):

    #INterpolate first to get better resolution, like done in "background.py"
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
    # print("points shape: ", points.shape)
    time_map_fine = interpolator(points).reshape(xv.shape)
    plot_interpolated_map(xv,yv, time_map_fine)

    images = scan_pattern(fov1, fov2, min_angle,n_radii=8)

    exposures = np.zeros(len(images))
    for i, image in enumerate(images):
        polygon = Path(image)
        grid_points = np.column_stack((xv.ravel(), yv.ravel()))
        mask = polygon.contains_points(grid_points, radius=1e-9).reshape(xv.shape)
        inside_data = time_map_fine[mask]
        exposures[i] = np.max(inside_data) if inside_data.size > 0 else np.nan


    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        collection = PolyCollection(
            images,
            array=exposures,  # Maps each rectangle to its exposure value
            cmap="viridis",  # Color map (try 'plasma', 'inferno', or 'turbo')
            edgecolors="black",  # Optional: outlines each rectangle
            linewidths=0.4,  # Thin borders so overlapping rectangles stay readable
            alpha=0.8,  # Slight transparency to see overlapping footprints
        )
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_aspect("equal")
        cbar = fig.colorbar(collection, ax=ax, label="Exposure Time [s]")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Exposure Time foro each frame")
        plt.tight_layout()
        plt.show()



        t_plot = np.cumsum(exposures)/3600
        fig, ax = plt.subplots()
        ax.plot(t_plot, exposures)
        ax.set(xlabel='time [h]', ylabel='exposure time [s]', title = 'Exposure length over scanning pattern')
        ax.grid()
        plt.show()

    return exposures













