import numpy as np
from indices import IndicesClassEncoderEq, NORMP

band_indexes = list(range(1, 12))
encoder = IndicesClassEncoderEq([NORMP], band_indexes)

INDICES = [
    encoder.getIndex(13),
    encoder.getIndex(54),
    encoder.getIndex(76),
]

MINMAX = [
    [-0.3045226812229239, 0.10015644019435427],
    [-0.6560889572250587, -0.19370390259716894],
    [-0.7410045649390549, -0.3036949073831795],
]

def convert(s2):
    rgb = np.empty((s2.shape[0], s2.shape[1], 3))

    s2[~np.isfinite(s2)] = 0
    nf = (s2 == 0).all(axis=2)
    s2[nf] = 1 # all features will result in 0 value for all same bands

    for i in range(3):
        val = INDICES[i].getValue(s2.swapaxes(0, 2).swapaxes(1, 2))
        rgb[:, :, i] = (val - MINMAX[i][0]) / (MINMAX[i][1] - MINMAX[i][0]) 

    db_norm = np.clip(255 * rgb, 0, 255).astype("uint8")
    return db_norm

NORM_MEAN = (0, 0, 0)
NORM_STD = (1, 1, 1)
NODATA_RGB = list(convert(np.zeros((1, 1, 12)))[0, 0, :])
