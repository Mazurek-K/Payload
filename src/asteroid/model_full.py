# from background import calculate_background
# from asteroid import calculate_asteroid_signal
#
#
# def main():
#     background_rate = calculate_background()
#     print("\n ------------------------------------------------------------")
#     asteroid_signal = calculate_asteroid_signal()
#
#     print(background_rate.mean())
#
# if __name__ == "__main__":
#     main()

######################

from background import calculate_background
from asteroid import calculate_asteroid_signal
from config import ( WAVELENGTH_UM, APERTURE, PIXEL_PITCH)
import numpy as np

def main():
    # Background rate: electrons/s/pixel
    background_rate = calculate_background()
    mean_background = background_rate.mean()

    # Total asteroid signal: electrons/s
    asteroid_rate = calculate_asteroid_signal()

    # Number of pixels covering the asteroid's detected image.  PSF
    d_airy = 2.44e-6*WAVELENGTH_UM[0]/APERTURE
    n_pix_source = ((np.pi*d_airy**2)/4) / (PIXEL_PITCH**2)
    print("N pix source: ",n_pix_source)


    # Background contribution over those pixels: electrons/s
    background_under_asteroid = mean_background * n_pix_source

    print(f"Mean background: {mean_background:.2f} electrons/s/pixel")
    print(f"Asteroid signal: {asteroid_rate:.6g} electrons/s")
    print(
        f"Background over {n_pix_source} pixels: "
        f"{background_under_asteroid:.6g} electrons/s"
    )
    print(
        "Asteroid/background ratio:",
        asteroid_rate / background_under_asteroid
    )

if __name__ == "__main__":
    main()
