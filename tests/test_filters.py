import pytest
import numpy as np

from ocear.filters import normalize


def test_normalize(img):
    with pytest.raises(Exception):
        normalize()
        normalize(np.array([[1.0, 1.0]]))
    normalized_img = normalize(img)
    assert np.max(normalized_img) <= 1.0
    assert np.min(normalized_img) >= 0.0
    assert img.shape == normalized_img.shape
