import numpy as np
import cv2


NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (1, 1, 1)
NODATA_RGB = [0, 0, 0]

def convert(s2):
    rgb = np.empty((s2.shape[0], s2.shape[1], 3))

    indices = np.array([[10, 11, 5], [9, 4, 11], [9, 5, 3]], dtype="int")

    for i in range(3):
        ind = indices[i]
        rgb[:, :, i] = (2 * s2[:, :, ind[0]] - s2[:, :, ind[1]] - s2[:, :, ind[2]]) * (s2[:, :, ind[1]] - s2[:, :, ind[2]])

    # db_norm = cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    db_norm = (255 * (np.arctan(30 * rgb) + 0.5 * np.pi) / np.pi).astype("uint8")
    return db_norm