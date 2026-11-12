from background import calculate_background_local, calculate_background_global
from asteroid import calculate_asteroid_signal, plot_asteroid_parameter_space
from config import ( DARK_CURRENT, READ_NOISE, WELL_CAPACITY)
from helpers import airy_disk, plot_lat_long, airy_disk2D
import numpy as np
from scipy.optimize import minimize

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

    time = 10

    def snr_def(ap, fl):

        background_ele_map = calculate_background_global(plot=False, aperture=ap, focal_length=fl)
        asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True, aperture=ap, focal_length=fl)
        snr_map = signal_to_noise(background_ele_map, np.max(asteroid_ele2D), time)
        overexposed_map = overexposed(background_ele_map, time)
        mask = ~overexposed_map

        return np.sum(snr_map[mask])

    def objective(x):
        ap, fl = x
        return -snr_def(ap, fl)

    x0 = [0.5, 1]

    bounds = [
        (0.1, 0.7),
        (0.3, 2.5)
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
            'maxiter': 1000,
            'xtol': 1e-5,
            'ftol': 1e-5,
            'disp': True
        }
    )

    print("Optimal aperture:", result.x[0])
    print("Optimal focal length:", result.x[1])
    print("Maximum SNR objective:", -result.fun)
    print("Success:", result.success)
    print("Message:", result.message)

    return result.x

def main():

    background_ele_map = calculate_background_global() # [e-/s/pix] {27x27}
    asteroid_ele1D = calculate_asteroid_signal(per_pixel=True) # [e-/s/pix] {1}
    asteroid_ele2D = calculate_asteroid_signal(per_pixel2D=True, plot=True)

    snr_map = signal_to_noise(background_ele_map, np.max(asteroid_ele2D), 10)
    overexposed_map = overexposed(background_ele_map,10)

    plot_lat_long(snr_map, "SNR")
    plot_lat_long(overexposed_map, "OVEREXPOSED")
    plot_asteroid_parameter_space()

    optimize_lens()




if __name__ == "__main__":
    main()
