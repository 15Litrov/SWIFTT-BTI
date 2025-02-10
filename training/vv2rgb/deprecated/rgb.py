import numpy as np
import cv2


NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (1, 1, 1)
NODATA_RGB = [0, 0, 0]

def convert(s2):
    rgb = np.empty((s2.shape[0], s2.shape[1], 3))
    rgb[:, :, 0] = s2[:, :, 3]
    rgb[:, :, 1] = s2[:, :, 2]
    rgb[:, :, 2] = s2[:, :, 1]

    db_norm = cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return db_norm