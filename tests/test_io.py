import pytest
import numpy as np
import imageio

from ocear import io


def test_load():
    with pytest.raises(Exception):
        io.load_image('i-dont-exist.jpg')
    img = io.load_image('tests/fixtures/example.jpg')
    assert type(img) == imageio.core.util.Array
    assert img.shape == (1200, 1600)
