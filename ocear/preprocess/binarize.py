import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.stats import scoreatpercentile

from ocear.preprocess.utils import clip_borders

ESCALE = 1.0


def binarize(image):
    """
    Rescale the grayscale image to the high and low values in order to
    increase contrast and cast the pixel values to either 0 or 1 (white or
    black).
    """
    img = clip_borders(image)
    v = img - gaussian_filter(img, 20.0)
    v = gaussian_filter(v ** 2, 20.0) ** 0.5
    v = (v > 0.3) * np.amax(v)
    v = binary_dilation(v, np.ones((50, 1)))
    v = binary_dilation(v, np.ones((1, 50)))
    img = img[v]

    lo = scoreatpercentile(img.ravel(), 5)
    hi = scoreatpercentile(img.ravel(), 90)

    image = image - lo
    image = image / (hi - lo)
    image = np.clip(image, 0, 1)
    image = (image > 0.5) * 1
    return image
