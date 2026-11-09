from background import calculate_background_local, calculate_background_global
from asteroid import calculate_asteroid_signal, plot_asteroid_parameter_space
from config import ( WAVELENGTH_UM, APERTURE, PIXEL_PITCH)
import numpy as np
np.set_printoptions(linewidth=200)

def main():
    # Background rate: electrons/s/pixel
    # background_rate = calculate_background_local()

    # Total asteroid signal: electrons/s
    # asteroid_rate = calculate_asteroid_signal()

    # Number of pixels covering the asteroid's detected image.  PSF
    # d_airy = 2 * 1.22e-6*WAVELENGTH_UM[0]/APERTURE
    # n_pix_source = ((np.pi*d_airy**2)/4) / (PIXEL_PITCH**2) # just rough assume that A_pixels = A_airy


    calculate_background_global()
    plot_asteroid_parameter_space()

if __name__ == "__main__":
    main()
