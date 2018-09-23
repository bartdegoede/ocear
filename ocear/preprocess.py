import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage.filters import percentile_filter

MAX_SKEW = 3
SKEW_STEPS = 32


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


def _skew_angle(image):
    """
    Estimate skew angle where the horizontal variance in pixel intensity is
    highest; the higher the variance, the "straighter up" the letters should
    stand.
    """
    angles = np.linspace(-MAX_SKEW, MAX_SKEW, SKEW_STEPS + 1)
    estimates = []
    for angle in angles:
        variance = np.mean(
            interpolation.rotate(image, angle, order=0, mode='constant'), axis=1
        ).var()
        estimates.append((variance, angle))
    return max(estimates)[1]


def skew(image):
    """
    Rotate image by an estimated skew.
    """
    height, width = image.shape
    # ignore 10% of border for skew estimation; maybe make configurable
    _h, _w = int(0.1 * height), int(0.1 * width)

    # increase contrast for better skew estimation
    img = np.amax(image) - image
    img = img - np.amin(img)
    # estimate skew angle
    angle = _skew_angle(img[_h : height - _h, _w : width - _w])
    img = interpolation.rotate(img, angle, reshape=False)
    return np.amax(img) - img
