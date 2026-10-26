import numpy as np
import matplotlib.pyplot as plt
# Sun's equatorial coordinates (example values; update for your date)
sun_ra_deg = 180.0
sun_dec_deg = 0.0

# Define the observing region
elongations_deg = np.arange(20, 41, 5)
position_angles_deg = np.arange(0, 360, 10)

# Generate every combination of elongation and position angle
elongation, position_angle = np.meshgrid(
    elongations_deg, position_angles_deg
)

# Convert to radians
ra0 = np.radians(sun_ra_deg)
dec0 = np.radians(sun_dec_deg)
sep = np.radians(elongation.ravel())
pa = np.radians(position_angle.ravel())

# Spherical coordinate transformation
dec = np.arcsin(
    np.sin(dec0) * np.cos(sep)
    + np.cos(dec0) * np.sin(sep) * np.cos(pa)
)

ra = ra0 + np.arctan2(
    np.sin(pa) * np.sin(sep) * np.cos(dec0),
    np.cos(sep) - np.sin(dec0) * np.sin(dec)
)

# Convert to degrees
ra_deg = np.degrees(ra) % 360
dec_deg = np.degrees(dec)

# Print coordinates for IRSA
for e, p, r, d in zip(
    elongation.ravel(),
    position_angle.ravel(),
    ra_deg,
    dec_deg
):
    print(
        f"Elongation={e:4.1f}°, "
        f"PA={p:5.1f}°, RA={r:8.3f}°, Dec={d:8.3f}°"
    )
plt.scatter(ra_deg, dec_deg)
plt.show()