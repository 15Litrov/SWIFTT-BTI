import numpy as np
from indices import IndicesClassEncoderEq, HueSimp

band_indexes = list(range(1, 12))
encoder = IndicesClassEncoderEq([HueSimp], band_indexes)

INDICES = [
    encoder.getIndex(144),
    encoder.getIndex(164),
    encoder.getIndex(1274),
]

MINMAX = [
    [-0.0001988100073957444, -4.90000942945934e-07],
    [-0.00025439994701594097, 0.0019416993488381067],
    [-0.012856470103411677, 0.0039511999869823455],
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
