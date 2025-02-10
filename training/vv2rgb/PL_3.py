import numpy as np
from indices import IndicesClassEncoderEq, HueSimp

band_indexes = list(range(1, 12))
encoder = IndicesClassEncoderEq([HueSimp], band_indexes)

INDICES = [
    encoder.getIndex(1263),
    encoder.getIndex(147),
    encoder.getIndex(950),
]

MINMAX = [
    [-0.004426019758488539, 0.004360103527477899],
    [-0.00405730392405531, 0.0010753600360298154],
    [-0.0025704015757799237, 0.00359378885992534],
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
