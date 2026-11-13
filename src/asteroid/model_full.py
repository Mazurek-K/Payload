from background import calculate_background_local, calculate_background_global
from asteroid import calculate_asteroid_signal, plot_asteroid_parameter_space
from config import ( DARK_CURRENT, READ_NOISE, WELL_CAPACITY, HDF5_FILE)
from helpers import airy_disk, plot_lat_long, airy_disk2D
import numpy as np
from scipy.optimize import minimize
import h5py
from src.asteroid.config import FOCAL_LENGTH, PIXEL_PITCH

np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt

def signal_to_noise(background_ele_map, asteroid_ele, time):

    signal = asteroid_ele * time
    background = background_ele_map * time
    dark = DARK_CURRENT * time
    read = READ_NOISE**2

    snr = signal / np.sqrt(signal + background + dark + read)
    return snr

def overexposed(background_ele_map, time):
    background = background_ele_map * time
    dark = DARK_CURRENT * time
    total_ele_perpix = background + dark
    overexposed = total_ele_perpix >= WELL_CAPACITY

    return overexposed


def optimize_lens():

    # time = 5
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

    def snr_def(ap, fl, time):
        background_ele_map = calculate_background_global(plot=False, aperture=0.5, focal_length=1)
        asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True, aperture=0.5, focal_length=1)
        snr_map = signal_to_noise(background_ele_map, np.max(asteroid_ele2D), 10)
        overexposed_map = overexposed(background_ele_map, 10)

        snr_map1 = snr_map[lon_order, :][:, lat_order]
        mask1 = (np.sqrt(lon[:, None] ** 2 + lat[None, :] ** 2) > 25)
        overexposed_map1 = np.where(mask1, overexposed_map, 0)

        mask2 = ~overexposed_map
        snr_map2 = np.where(mask2, snr_map, 0)

        overexposed_sum = np.sum(overexposed_map1)
        exposed_sum = np.sum(snr_map2)

        return exposed_sum-10*overexposed_sum

    def objective(x):
        ap, fl, t = x
        return -snr_def(ap, fl, t)

    x0 = [0.5, 1, 5]

    bounds = [
        (0.1, 0.5),
        (0.3, 2.5),
        (2.0, 30.0)
    ]

    # result = minimize(
    #     objective,
    #     x0=x0,
    #     method='L-BFGS-B',
    #     bounds=bounds
    # )
    result = minimize(
        objective,
        x0=x0,
        method='Powell',
        bounds=bounds,
        options={
            'maxiter': 10000,
            'xtol': 1e-6,
            'ftol': 1e-6,
            'disp': True
        }
    )

    print("Optimal aperture:", result.x[0])
    print("Optimal focal length:", result.x[1])
    print("Optimal exposure time:", result.x[2])
    print("Maximum SNR objective:", -result.fun)
    print("Success:", result.success)
    print("Message:", result.message)

    return result.x

def main():

    background_ele_map = calculate_background_global() # [e-/s/pix] {27x27}
    asteroid_ele1D = calculate_asteroid_signal(per_pixel=True) # [e-/s/pix] {1}
    asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True, plot=True)

    snr_map = signal_to_noise(background_ele_map, np.max(asteroid_ele2D), 30)
    overexposed_map = overexposed(background_ele_map,30)

    plot_lat_long(snr_map, "SNR")
    plot_lat_long(overexposed_map, "OVEREXPOSED")
    plot_asteroid_parameter_space()

    ap_opt, fl_opt, time_opt =  optimize_lens()

    fov_rad1 = 2*np.arctan((2048*18e-6)/(2*fl_opt))
    fov_rad2 = 2*np.arctan((2*2048*18e-6)/(2*fl_opt))
    print("FOVs: ")
    print(np.rad2deg(fov_rad1))
    print(np.rad2deg(fov_rad2))



    # with h5py.File(HDF5_FILE, "r") as file:
    #     lon = file["coordinates/ecliptic_longitude_deg"][:]
    #     lat = file["coordinates/ecliptic_latitude_deg"][:]
    # lon = lon[:, 0].astype(float)
    # lat = lat[0, :].astype(float)
    # lon = (np.rad2deg(np.unwrap(np.deg2rad(lon))) + 180) % 360 - 180
    # lon_order = np.argsort(lon)
    # lat_order = np.argsort(lat)
    # lon = lon[lon_order]
    # lat = lat[lat_order]
    #
    # background_ele_map = calculate_background_global(plot=False, aperture=0.5, focal_length=1)
    # asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True, aperture=0.5, focal_length=1)
    # snr_map = signal_to_noise(background_ele_map, np.max(asteroid_ele2D), 10)
    # overexposed_map = overexposed(background_ele_map, 10)
    #
    # snr_map1 = snr_map[lon_order, :][:, lat_order]
    # mask1 = (np.sqrt(lon[:, None] ** 2 + lat[None, :] ** 2) > 25)
    # overexposed_map1 = np.where(mask1, overexposed_map, 0)
    #
    # mask2 = ~overexposed_map
    # snr_map2 = np.where(mask2, snr_map, 0)
    #
    # overexposed_sum = np.sum(overexposed_map1)
    # exposed_sum = np.sum(snr_map2)
    #
    # plot_lat_long(overexposed_map1)
    # plot_lat_long(snr_map2)


if __name__ == "__main__":
    main()
