import numpy as np
import cv2


NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (1, 1, 1)
NODATA_RGB = [0, 0, 0]

def convert(s2):
    rgb = np.empty((s2.shape[0], s2.shape[1], 3))
    nf = (s2 == 0).all(axis=2)

    indices = np.array([[10, 5, 4], [9, 2, 3], [5, 10, 6]], dtype="int")
    args = [-250, -250, 250]

    for i in range(3):
        ind = indices[i]
        val = (2 * s2[:, :, ind[0]] - s2[:, :, ind[1]] - s2[:, :, ind[2]]) * (s2[:, :, ind[1]] - s2[:, :, ind[2]])
        val = np.clip(np.arctan(args[i] * val) / np.pi + 0.5, 0, 1)
        val[nf] = 0
        rgb[:, :, i] = val

    db_norm = (255 * rgb).astype("uint8")

    # db_norm = cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    db_norm[nf, :] = 0
    return db_norm