import numpy as np
from .utils import estimate_scale, remove_horizontal_lines


def segment(image):
    scale = estimate_scale(1 - image)
    image = np.array(image, 'B')
    image = remove_horizontal_lines(image, scale)
    return image
