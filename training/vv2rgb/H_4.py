import numpy as np
from indices import IndicesClassEncoderEq, NORMP4

band_indexes = list(range(1, 12))
encoder = IndicesClassEncoderEq([NORMP4], band_indexes)

INDICES = [
    encoder.getIndex(6225),
    encoder.getIndex(12489),
    encoder.getIndex(3949),
]

MINMAX = [
    [-0.32973090916364195, -0.13489027816390695],
    [0.2929443742807403, 0.610055909891059],
    [-4.391900454727976, -1.1768749375667924],
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
