import numpy as np

from optiland import optic

lens = optic.Optic(name="TMA")

# set aperture
lens.set_aperture(aperture_type="EPD", value=10)

# set fields
lens.fields.set_type(field_type="angle")
lens.fields.add(y=0)
lens.fields.add(y=+1.5)
lens.fields.add(y=-1.5)

# set wavelengths
lens.wavelengths.add(value=0.486)
lens.wavelengths.add(value=0.587, is_primary=True)
lens.wavelengths.add(value=0.656)

# add surfaces
lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)

# these are optimized coefficients
c1 = np.array(
    [
        [2.05790534e-05, -2.04095791e-05, 4.55045366e-05, 1.47183545e-05],
        [3.64010850e-08, -9.18266382e-06, 3.12457789e-07, -3.43010757e-08],
        [-1.18212287e-04, 2.97963986e-06, -1.31354947e-06, 7.34797747e-07],
        [-1.09979245e-06, 8.55358588e-07, -3.94583506e-07, -5.01115270e-08],
    ],
)
lens.surfaces.add(
    index=1,
    radius=-99.9773,
    thickness=-20.0003,
    conic=2.35645e-06,
    material="mirror",
    rx=-0.261801,
    is_stop=True,
    surface_type="polynomial",
    coefficients=c1,
)

c2 = np.array(
    [
        [-2.05802662e-05, 2.05376130e-05, -2.91883756e-05, 4.01274602e-05],
        [-2.94473576e-08, -1.63268440e-06, -5.35655902e-06, -1.50920309e-05],
        [4.87006021e-05, -1.70308923e-05, 2.12362562e-05, 1.51092539e-05],
        [-1.04657810e-05, -2.25376153e-05, -8.02314506e-06, 3.02373985e-06],
    ],
)
lens.surfaces.add(
    index=2,
    radius=-100.019,
    thickness=20.0003,
    conic=-2.21455e-07,
    material="mirror",
    rx=-0.174532,
    dy=-11.5,
    surface_type="polynomial",
    coefficients=c2,
)

c3 = np.array(
    [
        [2.05787912e-05, -2.05604065e-05, 2.54616632e-05, -3.62841214e-05],
        [6.24577550e-10, 1.09936457e-07, -2.69199040e-06, -1.05068667e-05],
        [-3.28005504e-05, 1.88336844e-05, -3.34058539e-05, 4.93261709e-06],
        [6.09659324e-07, -1.58589636e-06, -1.16163570e-05, -2.26243673e-05],
    ],
)
lens.surfaces.add(
    index=3,
    radius=-99.9774,
    thickness=-20.0003,
    conic=5.32063e-08,
    material="mirror",
    rx=-0.0174547,
    dy=-15,
    surface_type="polynomial",
    coefficients=c3,
)

lens.surfaces.add(index=4, dy=-19.3)


lens.draw3D()