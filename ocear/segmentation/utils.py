import numpy as np
from scipy.ndimage import label, measurements, morphology


def _area(slice):
    return np.prod([max(x.stop - x.start, 0) for x in slice[:2]])


def estimate_scale(image):
    """
    Takes the binary image generated with ocear.preprocess.binarize
    """
    image = 1 - image
    labels, n = label(image)
    objects = sorted(measurements.find_objects(labels), key=_area)
    scalemap = np.zeros(image.shape)
    for obj in objects:
        if np.amax(scalemap[obj]) > 0:
            continue
        scalemap[obj] = _area(obj) ** 0.5
    return np.median(scalemap[(scalemap > 3) & (scalemap < 100)])
