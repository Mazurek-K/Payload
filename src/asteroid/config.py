from pathlib import Path
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CSV_FILE = PROJECT_ROOT / "outputs" / "irsa_background_8_12um_grid.csv"
HDF5_FILE = "../outputs/irsa_background_ecliptic_8_12um.h5"
COMPONENT = "total"

# Physical constants
h = 6.62607015e-34
c = 299792458
k = 1.380649e-23  # Boltzmann constant, J/K
AU_2_M = 149597870700

# Scaling etc
SR_TO_ARCSEC = (648000 / np.pi)
SR_TO_ARCSEC2 = SR_TO_ARCSEC** 2

# Wavelengths
# WAVELENGTH_UM = np.array([8, 9, 10, 11, 12])
WAVELENGTH_UM = np.arange(8.0, 12.01, 1)

# Telescope and detector
PIXEL_PITCH = 30e-6
APERTURE = 0.5
FOCAL_LENGTH = 0.8
TRANSMISSION = 0.7
QE = 0.7
PIXEL_ROW = 480

# Sky position and asteroid
EMISSIVITY = 1.
EPSILON_F = 1. # surface ratio of incoming light on sensor
# RA0, DEC0 = 25.0, 30.0
ECLIPTIC_LON0 = 30.0
ECLIPTIC_LAT0 = 20.0
AST_DIAMETER = 50 # [m]
AST_DISTANCE = 0.01 * AU_2_M
AST_TEMPERATURE = 300 # [K]

# Derived quantities
LENS_AREA = np.pi * APERTURE**2 / 4
THETA_PIX = SR_TO_ARCSEC * PIXEL_PITCH / FOCAL_LENGTH  # [arcsec]
OMEGA_PIX = (THETA_PIX / SR_TO_ARCSEC)**2   # [sr]









