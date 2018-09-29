import numpy as np
from scipy.ndimage import label, measurements


def _area(obj):
    return np.prod([max(x.stop - x.start, 0) for x in obj[:2]])


def _width(obj):
    """
    `obj` is a tuple of slices denoting the x and y coordinates on the
    original image.
    """
    return obj[1].stop - obj[1].start


def _height(obj):
    """
    `obj` is a tuple of slices denoting the x and y coordinates on the
    original image.
    """
    return obj[0].stop - obj[0].start


def estimate_scale(image):
    """
    Takes the binary image generated with ocear.preprocess.binarize
    """
    labels, n = label(image)
    objects = sorted(measurements.find_objects(labels), key=_area)
    scalemap = np.zeros(image.shape)
    for obj in objects:
        if np.amax(scalemap[obj]) > 0:
            continue
        scalemap[obj] = _area(obj) ** 0.5
    return np.median(scalemap[(scalemap > 3) & (scalemap < 100)])


def remove_horizontal_lines(image, scale, maxsize=10):
    labels, _ = label(image)
    for i, obj in enumerate(measurements.find_objects(labels)):
        if _width(obj) > maxsize * scale:
            # set all really wide objects to 0
            labels[obj][labels[obj] == i + 1] = 0
    return np.array(labels != 0, 'B')
