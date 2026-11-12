from background import calculate_background_local, calculate_background_global
from asteroid import calculate_asteroid_signal, plot_asteroid_parameter_space
from config import ( DARK_CURRENT, READ_NOISE, WELL_CAPACITY)
from helpers import airy_disk, plot_lat_long, airy_disk2D
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt

def signal_to_noise(background_ele_map, asteroid_ele, time):

    signal = asteroid_ele * time
    background = background_ele_map * time
    dark = DARK_CURRENT * time
    read = READ_NOISE**2

    snr = signal / np.sqrt(signal + background + dark + read)

    total_ele_perpix = signal + background + dark
    overexposed = total_ele_perpix >= WELL_CAPACITY

    return snr, overexposed


def main():

    background_ele_map = calculate_background_global() # [e-/s/pix] {27x27}
    asteroid_ele = calculate_asteroid_signal(per_pixel=True) # [e-/s/pix] {1}
    snr_map, overexposed_map = signal_to_noise(background_ele_map, asteroid_ele, 15)
    plot_lat_long(snr_map, "SNR")
    plot_lat_long(overexposed_map, "OVEREXPOSED")
    plot_asteroid_parameter_space()

    asteroid_signal = calculate_asteroid_signal(per_pixel2D=True)
    fig, ax = plt.subplots()
    im = ax.imshow(
        asteroid_signal,
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=np.max(asteroid_signal)
    )

    n_pix = asteroid_signal.shape[0]
    ax.set_xticks(np.arange(-0.5, n_pix, 1))
    ax.set_yticks(np.arange(-0.5, n_pix, 1))
    ax.grid(color="red", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Electron rate [electrons/s]")
    plt.show()

if __name__ == "__main__":
    main()
