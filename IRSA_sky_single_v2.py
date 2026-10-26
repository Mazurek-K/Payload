import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# IRSA background model service
url = (
    "https://irsa.ipac.caltech.edu/"
    "cgi-bin/BackgroundModel/nph-bgmodel"
)

# Illustrative test configuration
params = {
    "locstr": "180.0 0.0 equ J2000",
    "year": 2026,
    "day": 180,
    "obslocin": 0,
    "obsverin": 4,
    "ido_viewin": 0,
}

wavelengths_um = np.arange(8.0, 12.01, 0.5)
results = []

for wavelength in wavelengths_um:
    query = params.copy()
    query["wavelength"] = float(wavelength)

    response = requests.get(
        url,
        params=query,
        timeout=30
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)

    if root.attrib.get("status") != "ok":
        raise RuntimeError(
            f"IRSA query failed at {wavelength} um: "
            f"{response.text}"
        )

    stats = root.find(".//statistics")

    def read_value(tag):
        text = stats.findtext(tag)
        return float(text.split()[0])

    results.append({
        "wavelength_um": wavelength,
        "zodiacal_MJy_sr": read_value("zody"),
        "ISM_MJy_sr": read_value("ism"),
        "stars_MJy_sr": read_value("stars"),
        "CIB_MJy_sr": read_value("cib"),
        "total_MJy_sr": read_value("totbg"),
    })

df = pd.DataFrame(results)

print(df.to_string(index=False))

df.to_csv(
    "irsa_background_8_12um.csv",
    index=False
)

plt.plot(
    df["wavelength_um"],
    df["zodiacal_MJy_sr"],
    marker="o",
    label="Zodiacal"
)

plt.plot(
    df["wavelength_um"],
    df["total_MJy_sr"],
    marker="s",
    label="Total background"
)

plt.xlabel("Wavelength (µm)")
plt.ylabel("Spectral radiance (MJy/sr)")
plt.title("IRSA background model: 8–12 µm")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()