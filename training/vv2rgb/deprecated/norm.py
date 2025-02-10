import numpy as np
import cv2


NORM_MEAN = (0, 0, 0)
NORM_STD = (1, 1, 1)
NODATA_RGB = [0, 0, 0]

def convert(s2):
    rgb = np.empty((s2.shape[0], s2.shape[1], 3))
    nf = (s2 == 0).all(axis=2)

    indices = np.array([[2, 3], [10, 9], [11, 10]], dtype="int")
    args = [-250, -250, 250]

    for i in range(3):
        ind = indices[i]
        val = (s2[:, :, ind[0]] - s2[:, :, ind[1]]) / (s2[:, :, ind[0]] + s2[:, :, ind[1]])
        val[nf] = 0
        rgb[:, :, i] = val

    db_norm = np.clip(255 * (1 + rgb), 0, 255).astype("uint8")

    # db_norm = cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    db_norm[nf, :] = 0
    return db_norm