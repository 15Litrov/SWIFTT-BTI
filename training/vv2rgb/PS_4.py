import numpy as np
from indices import IndicesClassEncoderEq, NORMP4

band_indexes = list(range(1, 12))
encoder = IndicesClassEncoderEq([NORMP4], band_indexes)

INDICES = [
    encoder.getIndex(1102),
    encoder.getIndex(6428),
    encoder.getIndex(526),
]

MINMAX = [
    [-0.1424391223336886, 0.04857338087802458],
    [0.31516640955479613, 0.5646789301855831],
    [0.14275228134018864, 0.5470525149502415],
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
