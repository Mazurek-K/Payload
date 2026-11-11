from background import calculate_background_local, calculate_background_global
from asteroid import calculate_asteroid_signal, plot_asteroid_parameter_space
from config import ( DARK_CURRENT, READ_NOISE, WELL_CAPACITY)
from helpers import airy_disk, plot_lat_long
import numpy as np
np.set_printoptions(linewidth=200)

def signal_to_noise(background_ele_map, asteroid_ele, time):
    n_pix, _ = airy_disk()

    signal = asteroid_ele * n_pix * time
    background = background_ele_map * n_pix * time
    dark = n_pix * DARK_CURRENT * time
    read = n_pix * READ_NOISE**2

    snr = signal / np.sqrt(signal + background + dark + read)

    total_ele_perpix = (asteroid_ele + background_ele_map + DARK_CURRENT) * time

    overexposed = total_ele_perpix >= WELL_CAPACITY

    return snr, overexposed


def main():

    background_ele_map = calculate_background_global() # [e-/s/pix] {27x27}
    asteroid_ele = calculate_asteroid_signal(per_pixel=True) # [e-/s/pix] {1}
    snr_map, overexposed_map = signal_to_noise(background_ele_map, asteroid_ele, 5)
    plot_lat_long(snr_map, "SNR")
    plot_lat_long(overexposed_map, "OVEREXPOSED")
    plot_asteroid_parameter_space()


if __name__ == "__main__":
    main()
