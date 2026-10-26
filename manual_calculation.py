import numpy as np

# Physical constants (SI units)
h = 6.62607015e-34       # Planck constant [J s]
c = 2.99792458e8          # Speed of light [m/s]
k_B = 1.380649e-23        # Boltzmann constant [J/K]
AU = 1.495978707e11       # Astronomical unit [m]
epsilon = 0.9             # Thermal emissivity
rad_to_arcs = 180*3600/np.pi

# Spectral integration range: 8–12 micrometers
lambda_min = 8e-6         # [m]
lambda_max = 12e-6        # [m]
n_lambda = 1000
wavelength = np.linspace(lambda_min, lambda_max, n_lambda)


# Reference case
D_ref = 20.0              # [m]
Delta_ref = 0.08 * AU     # [m]
T_ref = 300.0             # [K]
D_lens = 0.3          # [m]
eta_lens = 0.8   # lens efficiency [-]

# Planck spectral radiance B_lambda [W m^-3 sr^-1]
def planck_lambda(wavelength, T):
    return (
        (2 * h * c**2) / wavelength**5
        / np.expm1(h * c / (wavelength * k_B * T))
    )

# Asteroid spectral flux density F_lambda [W m^-3]
def asteroid_spectrum(wavelength, D_ast, Delta, T, epsilon):
    B_lambda = planck_lambda(wavelength, T)
    A_projected = np.pi * (D_ast / 2)**2

    return epsilon * B_lambda * A_projected / Delta**2

# 1. Reference asteroid spectrum
F_lambda_ref = asteroid_spectrum(
    wavelength, D_ref, Delta_ref, T_ref, epsilon
)
# Integrate spectrum over 8–12 micrometers
F_band_ref = np.trapezoid(F_lambda_ref, wavelength)
A_lens = (np.pi * D_lens**2)/4
P_sensor = F_band_ref*A_lens*eta_lens


# Spectral flux density at 10 micrometers
lambda_10 = 10e-6
F_lambda_10 = asteroid_spectrum(
    np.array([lambda_10]), D_ref, Delta_ref, T_ref, epsilon
)[0]

# Convert to flux density per unit frequency
F_nu_10 = F_lambda_10 * lambda_10**2 / c

# Convert to mJy
F_nu_10_mJy = F_nu_10 / 1e-29

theta_r = 1.22*lambda_10/D_lens  # max angular precision
theta_r_arc = theta_r * rad_to_arcs

theta_ref = D_ref/Delta_ref
theta_ref_arc = theta_ref * rad_to_arcs

print("REFERENCE ASTEROID")
print(f"Diameter: {D_ref:.1f} m")
print(f"Distance: {Delta_ref / AU:.3f} AU")
print(f"Temperature: {T_ref:.1f} K")
print(f"Flux density at 10 um: {F_nu_10_mJy:.4f} mJy")
print(f"Integrated 8-12 um flux: {F_band_ref:.4e} W/m^2")
print(f"Power on the sensor: {P_sensor:.4e} W")
print(" --------------------------------------------------- ")
print(f"Min angle to measure due to diffraction: {theta_r_arc:.5f} arcsecond")
print(f"Asteroid  measure: {theta_ref_arc:.5f} arcsecond")

