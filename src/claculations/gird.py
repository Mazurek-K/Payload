import csv
import numpy as np
CSV_FILE = "../outputs/irsa_background_8_12um_grid.csv"
COMPONENT = "total_MJy_sr"
SELECTED_WAVELENGTH_UM = [8, 9, 10, 11, 12]

ra_values = []
dec_values = []
wavelength_values = []
intensity_values = []

with open(CSV_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    required_columns = {
        "ra_deg",
        "dec_deg",
        "wavelength_um",
        COMPONENT,
    }

    missing = required_columns - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    for row in reader:
        wavelength = float(row["wavelength_um"])

        if np.any(np.isclose(
            wavelength, SELECTED_WAVELENGTH_UM,
            atol=1e-6, rtol=0
        )):
            ra_values.append(float(row["ra_deg"]))
            dec_values.append(float(row["dec_deg"]))
            wavelength_values.append(wavelength)
            intensity_values.append(float(row[COMPONENT]))

# Convert to arrays
ra_values = np.asarray(ra_values)
dec_values = np.asarray(dec_values)
wavelength_values = np.asarray(wavelength_values)
intensity_values = np.asarray(intensity_values)

if intensity_values.size == 0:
    raise ValueError("No data found for the selected wavelengths.")


ra = np.sort(np.unique(ra_values))
dec = np.sort(np.unique(dec_values))
wavelength = np.sort(np.unique(wavelength_values))
intensity_tensor = np.full(
    (len(wavelength), len(dec), len(ra)),
    np.nan
)

# Map coordinates to array indices
ra_idx = np.searchsorted(ra, ra_values)
dec_idx = np.searchsorted(dec, dec_values)
wl_idx = np.searchsorted(wavelength, wavelength_values)

# Fill tensor
intensity_tensor[wl_idx, dec_idx, ra_idx] = intensity_values

print("RA shape:", ra.shape)
print("Dec shape:", dec.shape)
print("Wavelengths (µm):", wavelength)
print("Intensity tensor shape:", intensity_tensor.shape)
print("Intensity units: MJy/sr")

# Example: background map at 8 µm
i = np.argmin(np.abs(wavelength - 8.0))
background_8um = intensity_tensor[i]

print("8 µm map shape:", background_8um.shape)
print("Example intensity:", intensity_tensor[i, 0, 0], "MJy/sr")
print("Coordinates:", ra[0], "deg RA,", dec[0], "deg Dec")


# Telescope and detector
pixel_pitch = 30e-6       # m
aperture = 0.4            # m
lens_area = np.pi * aperture**2 / 4  # m²
focal_len = 0.8           # m
theta_pix = 206265 * pixel_pitch / focal_len  # arcsec/pixel
omega_pix = (theta_pix / 206265)**2           # sr/pixel

T, QE = 0.7, 0.7          # assumed transmission and quantum efficiency

# intensity_tensor: (wavelength, dec, ra), in MJy/sr
# wavelength: [8, 9, 10, 11, 12] in µm
lam = wavelength * 1e-6   # m
I_nu = intensity_tensor * 1e-20  # W m⁻² Hz⁻¹ sr⁻¹

# Convert spectral radiance to per µm
I_lam = I_nu * 3e8 / lam[:, None, None]**2 * 1e-6

h = 6.626e-34 # planck constant
c = 3e8 # speed of light

# Photon rate per pixel, integrated over wavelength
photon_spectrum = I_lam * (lam[:, None, None] / (h*c)) * lens_area * omega_pix * T # photons s⁻¹ µm⁻¹ pixel⁻¹
photon_rate = np.trapezoid(photon_spectrum, wavelength, axis=0)
electron_rate = photon_rate * QE  # electrons/s/pixel

print("Photon rate shape:", photon_rate.shape)
print("Electron rate shape:", electron_rate.shape)
print("Mean background:", np.nanmean(electron_rate), "electrons/s/pixel")




