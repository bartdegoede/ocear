import numpy as np
from scipy.ndimage import interpolation

from ocear.preprocess.utils import clip_borders

MAX_SKEW = 3
SKEW_STEPS = 32


def _skew_angle(image):
    """
    Estimate skew angle where the horizontal variance in pixel intensity is
    highest; the higher the variance, the "straighter up" the letters should
    stand.
    """
    estimates = []
    for angle in np.linspace(-MAX_SKEW, MAX_SKEW, SKEW_STEPS + 1):
        variance = np.mean(
            interpolation.rotate(image, angle, order=0, mode='constant'),
            axis=1
        ).var()
        estimates.append((variance, angle))
    return max(estimates)[1]


def skew(image):
    """
    Rotate image by an estimated skew.
    """
    # increase contrast for better skew estimation
    img = np.amax(image) - image
    img = img - np.amin(img)
    # estimate skew angle
    angle = _skew_angle(clip_borders(img))
    img = interpolation.rotate(img, angle, reshape=False)
    return np.amax(img) - img
