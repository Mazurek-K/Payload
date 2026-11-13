from background import calculate_background_local, calculate_background_global
from asteroid import calculate_asteroid_signal, plot_asteroid_parameter_space
from config import ( DARK_CURRENT, READ_NOISE, WELL_CAPACITY, HDF5_FILE)
from helpers import airy_disk, plot_lat_long, airy_disk2D
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import h5py
from src.asteroid.config import FOCAL_LENGTH, PIXEL_PITCH, APERTURE

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

    def snr_def(ap, fl, time):
        background_ele_map = calculate_background_global(plot=False, aperture=ap, focal_length=fl)
        asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True, aperture=ap, focal_length=fl)
        snr_map = signal_to_noise(background_ele_map, np.max(asteroid_ele2D), time)
        overexposed_map = overexposed(background_ele_map, time)

        mask2 = ~overexposed_map
        snr_map2 = np.where(mask2, snr_map, 0)
        snr_map3 = np.where(snr_map2 > 5, 1, 0)

        exposed_sum = np.sum(snr_map3)
        return exposed_sum

    def objective(x):
        ap, fl, t = x
        return -snr_def(ap, fl, t)

    x0 = [0.2, 1, 10]
    bounds = [
        (0.1, 0.5),
        (0.3, 2.5),
        (2.0, 30.0)
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy='best1bin',
        maxiter=200,
        popsize=25,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        disp=True,
        seed=42
    )

    print("Optimal aperture:", result.x[0])
    print("Optimal focal length:", result.x[1])
    print("Optimal exposure time:", result.x[2])
    print("Maximum SNR objective:", -result.fun)
    print("Success:", result.success)
    print("Message:", result.message)

    return result.x



def optimize_lens2():
    def snr_def(ap, fl):
        background_ele_map = calculate_background_global(plot=False,aperture=ap,focal_length=fl)
        asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True,aperture=ap,focal_length=fl)
        asteroid_ele2D = np.max(asteroid_ele2D)
        # Exposure time bounds
        t_min = 2.0
        t_max = 30.0

        # Calculate maximum exposure time per pixel
        total_background = background_ele_map + DARK_CURRENT
        t_opt_map = 0.8 * WELL_CAPACITY / total_background
        t_opt_map = np.clip(t_opt_map, t_min, t_max)

        valid_pixels = total_background * t_min < WELL_CAPACITY
        snr_map = signal_to_noise(background_ele_map, asteroid_ele2D, t_opt_map)
        snr_map = np.where(valid_pixels, snr_map, 0)
        exposed_sum = np.sum(snr_map > 5)
        mean_snr = np.mean(snr_map[snr_map > 5]) if exposed_sum > 0 else 0

        return mean_snr

    def objective(x):
        ap, fl = x
        return -snr_def(ap, fl)

    x0 = [0.2, 1]
    bounds = [
        (0.1, 0.5),
        (0.3, 2.5)
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy='best1bin',
        maxiter=200,
        popsize=25,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        disp=True,
        seed=42
    )

    print("Optimal aperture:", result.x[0])
    print("Optimal focal length:", result.x[1])
    print("Maximum SNR objective:", -result.fun)
    print("Success:", result.success)
    print("Message:", result.message)

    return result.x
def main():

    # ap_opt, fl_opt, time_opt =  optimize_lens()
    # ap, fl, time = ap_opt, fl_opt, time_opt
    ap, fl = optimize_lens2()

    background_ele_map = calculate_background_global(plot=True, aperture=ap, focal_length=fl) # [e-/s/pix] {27x27}
    asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True, aperture=ap, focal_length=fl, plot=True)

    time = 0.6 * WELL_CAPACITY / background_ele_map
    snr_map = signal_to_noise(background_ele_map, np.max(asteroid_ele2D), time)
    overexposed_map = overexposed(background_ele_map,time)

    plot_lat_long(snr_map, "SNR", mask=True)
    plot_lat_long(overexposed_map, "OVEREXPOSED")
    plot_lat_long(time, "time", mask=True)
    plot_asteroid_parameter_space()
    fov_rad1 = 2 * np.arctan((2048 * 18e-6) / (2 * fl))
    fov_rad2 = 2 * np.arctan((2 * 2048 * 18e-6) / (2 * fl))
    print("FOVs: ")
    print(np.rad2deg(fov_rad1))
    print(np.rad2deg(fov_rad2))



if __name__ == "__main__":
    main()
