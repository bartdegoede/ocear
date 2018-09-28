import numpy as np
from .utils import estimate_scale


def segment(image):
    scale = estimate_scale(image)
    print(scale)
    print(np.array(image, 'B'))
    return np.array(image, 'B')
