import pytest
import numpy as np

from ocear.preprocess import normalize, flatten, skew, binarize, utils


def test_normalize(img):
    with pytest.raises(Exception):
        normalize()
        normalize(np.array([[1.0, 1.0]]))
    normalized_img = normalize(img)
    assert np.max(normalized_img) <= 1.0
    assert np.min(normalized_img) >= 0.0
    assert img.shape == normalized_img.shape


def test_flatten(img):
    flattened_img = flatten(img)
    assert np.max(flattened_img) <= 1.0
    assert np.min(flattened_img) >= 0.0
    assert img.shape == flattened_img.shape


def test_skew(img):
    skewed_img = skew(img)
    assert img.shape == skewed_img.shape


def test_binarize(img):
    binarized_img = binarize(img)
    assert img.shape == binarized_img.shape
    # all values in the image have to be either 0 or 1
    assert (np.equal(binarized_img, 0) | np.equal(binarized_img, 1)).all()
