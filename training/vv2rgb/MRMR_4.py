import numpy as np
from indices import IndicesClassEncoderEq, NORMP4

band_indexes = list(range(1, 12))
encoder = IndicesClassEncoderEq([NORMP4], band_indexes)

INDICES = [
    encoder.getIndex(144),
    encoder.getIndex(14545),
    encoder.getIndex(14567),
]

MINMAX = [
    [-0.1322701702045459, 0.3117338424794585],
    [0.14179907925942703, 0.5118421860347533],
    [-1.682014739790682, -0.3215144021424001],
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
    db_norm[nf, :] = 128
    return db_norm

NORM_MEAN = (0, 0, 0)
NORM_STD = (1, 1, 1)
NODATA_RGB = [128, 128, 128]#list(convert(np.zeros((1, 1, 12)))[0, 0, :])
