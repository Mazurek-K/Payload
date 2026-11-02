import numpy as np

from optiland import optic

class TessarLens(optic.Optic):
    """Tessar Lens f/4.5

    Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 63
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
        self.surfaces.add(index=1, radius=1.3329, thickness=0.2791, material="N-SK15")
        self.surfaces.add(index=2, radius=-9.9754, thickness=0.2054)
        self.surfaces.add(
            index=3,
            radius=-2.0917,
            thickness=0.09,
            material=("F2", "schott"),
        )
        self.surfaces.add(index=4, radius=1.2123, thickness=0.0709)
        self.surfaces.add(index=5, radius=np.inf, thickness=0.1534, is_stop=True)
        self.surfaces.add(index=6, radius=-7.5205, thickness=0.09, material="K10")
        self.surfaces.add(index=7, radius=1.3010, thickness=0.3389, material="N-SK15")
        self.surfaces.add(index=8, radius=-1.5218, thickness=3.4025)
        self.surfaces.add(index=9)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=4.5)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=10)
        self.fields.add(y=20.5)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)

lens = TessarLens()
lens.draw3D(num_rays=5)