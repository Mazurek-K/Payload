from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Open the FITS file
file_path = "output/run_20261020_164007/detector_image.fits"

with fits.open(file_path) as hdul:
    hdul.info()  # Show the available HDUs
    data = hdul[0].data

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(data, cmap="gray", origin="lower")
plt.colorbar(label="Pixel value")
plt.title("FITS Image")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.show()