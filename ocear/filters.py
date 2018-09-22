import numpy as np


def normalize(image):
    """
    Scale pixel values between 0.0 and 1.0
    """
    if image is None or np.max(image) == np.min(image):
        raise Exception('No valid image provided')
    img = image - np.min(image)
    return img / np.max(img)
