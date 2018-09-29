import pytest
import imageio

from ocear import io


@pytest.fixture
def img():
    return io.load_image('tests/fixtures/example.jpg')


@pytest.fixture
def binarized_img():
    return imageio.imread('tests/fixtures/binarized.jpg')
