# detector.py

from pathlib import Path
import numpy as np
import pyxel

# Path to YAML file (same directory as this file)
CONFIG_FILE = Path(__file__).resolve().with_name("detector_config.yaml")

# Variables shared with the Pyxel custom model
_INPUT_IMAGE = None
_EXPOSURE_TIME = None
_OUTPUT_IMAGE = None


def pyxel_detector_noise(
    detector,
    dark_current=0.05,
    read_noise=15.0,
    full_well=80000.0,
    seed=42,
):
    """
    Custom Pyxel model.

    Input:
        Expected signal in electrons/pixel.

    Applies:
        - Poisson shot noise
        - Dark current shot noise
        - Gaussian read noise
        - Full-well saturation

    Output:
        Noisy electron image.
    """

    global _INPUT_IMAGE, _EXPOSURE_TIME, _OUTPUT_IMAGE

    if _INPUT_IMAGE is None:
        raise RuntimeError("No input electron image provided.")

    rng = np.random.default_rng(seed)

    signal = np.asarray(_INPUT_IMAGE, dtype=float)

    if np.any(signal < 0):
        raise ValueError("Expected electron counts cannot be negative.")

    # 1. Signal shot noise
    signal_noisy = rng.poisson(signal)

    # 2. Dark current shot noise
    dark_mean = dark_current * _EXPOSURE_TIME

    dark_noisy = rng.poisson(
        dark_mean,
        size=signal.shape
    )

    # 3. Total collected electrons
    collected = signal_noisy + dark_noisy

    # 4. Full-well saturation
    collected = np.minimum(collected, full_well)

    # 5. Readout noise
    readout = rng.normal(
        loc=0.0,
        scale=read_noise,
        size=signal.shape
    )

    measured = collected + readout

    # 6. Output clipping
    measured = np.clip(measured, 0, full_well)

    _OUTPUT_IMAGE = measured.copy()

    # Initialize Pyxel charge bucket
    detector.charge.array[:] = collected.astype(float)

    # Initialize Pyxel image bucket (required for image output)
    # Assumption: 1 electron = 1 ADU
    detector.image.array = measured.astype(np.uint16)
    return None


def simulate_detector(
    electron_image,
    exposure_time,
    seed=42,
):
    """
    Simulate detector noise using ESA Pyxel.

    Parameters
    ----------
    electron_image : np.ndarray
        Expected number of signal electrons per pixel.
    exposure_time : float
        Exposure time in seconds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy detector image in electrons.
    """

    global _INPUT_IMAGE, _EXPOSURE_TIME, _OUTPUT_IMAGE

    electron_image = np.asarray(electron_image, dtype=float)

    if electron_image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    if exposure_time <= 0:
        raise ValueError("Exposure time must be positive.")

    _INPUT_IMAGE = electron_image
    _EXPOSURE_TIME = float(exposure_time)
    _OUTPUT_IMAGE = None

    # Load Pyxel configuration
    config = pyxel.load(str(CONFIG_FILE))

    # Run Pyxel
    config.run()

    if _OUTPUT_IMAGE is None:
        raise RuntimeError(
            "Pyxel did not execute the custom detector model."
        )

    result = _OUTPUT_IMAGE.copy()

    # Clear global state
    _INPUT_IMAGE = None
    _EXPOSURE_TIME = None
    _OUTPUT_IMAGE = None

    return result