import numpy as np
import matplotlib.pyplot as plt

# Synthetic infrared image
size = 128
background = 1000       # Background photons per pixel
asteroid_signal = 500   # Additional photons at the source

image = np.full((size, size), background, dtype=float)

# Add a simple point source
image[size // 2, size // 2] += asteroid_signal

# Display
plt.figure(figsize=(7, 6))
plt.imshow(image, cmap="inferno", origin="lower")
plt.colorbar(label="Photon counts")
plt.title("Synthetic asteroid on infrared background")
plt.xlabel("Pixel")
plt.ylabel("Pixel")
plt.show()