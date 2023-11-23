import numpy as np
import math

from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import correlation

def bhattacharyya_distance(setA, setB):
    setA = setA[np.isfinite(setA)]
    setB = setB[np.isfinite(setB)]

    a_min = np.min(setA)
    a_max = np.max(setA)
    b_min = np.min(setB)
    b_max = np.max(setB)

    r_min = np.max([a_min, b_min])
    r_max = np.min([a_max, b_max])

    resolution = int(np.sqrt(len(setB)))

    h1, _ = np.histogram(setA, resolution, [r_min, r_max])
    h2, _ = np.histogram(setB, resolution, [r_min, r_max])

    h1 = h1 / len(setA)
    h2 = h2 / len(setB)

    return -math.log(np.sum(np.sqrt(h1*h2)))

def pearson_independency(d1, d2):
    corr, _ = pearsonr(d1, d2)
    dep = 1 - abs(corr)
    if not math.isfinite(dep):
        dep = 0

    return dep

def spearman_independency(d1, d2):
    corr, _ = spearmanr(d1, d2, nan_policy="omit")
    dep = 1 - abs(corr)
    if not math.isfinite(dep):
        dep = 0

    return dep

def distance_independency(d1, d2):
    corr = correlation(d1, d2)
    dep = 1 - abs(corr)
    if not math.isfinite(dep):
        dep = 0

    return dep