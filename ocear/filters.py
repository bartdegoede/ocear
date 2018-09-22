import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage.filters import percentile_filter


def normalize(image):
    """
    Scale pixel values between 0.0 and 1.0
    """
    if image is None or np.max(image) == np.min(image):
        raise Exception('No valid image provided')
    img = image - np.min(image)
    return img / np.max(img)


def flatten(image, zoom=0.5, percentile=85.0, filter_range=20):
    """
    Flatten image by local whitelevels.
    """
    img = interpolation.zoom(image, zoom)
    img = percentile_filter(image, percentile, (filter_range, 2))
    img = percentile_filter(img, percentile, (2, filter_range))
    img = interpolation.zoom(img, 1.0 / zoom)

    # NOTE: np.minimum computes element-wise minimum:
    # >>> np.minimum([2, 3, 4], [1, 5, 2])
    # array([1, 3, 2])
    width, height = np.minimum(image.shape, img.shape)
    return np.clip(image[:width, :height] - img[:width, :height] + 1, 0, 1)
