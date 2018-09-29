from ocear.segmentation import estimate_scale
from ocear.segmentation.utils import _width as width
from ocear.segmentation.utils import _height as height


def test_estimate_scale(binarized_img):
    assert 0.0 <= estimate_scale(binarized_img) <= 100.0


def test_width():
    obj = (slice(142, 169), slice(301, 311))
    assert width(obj) == 10


def test_height():
    obj = (slice(142, 169), slice(301, 311))
    assert height(obj) == 27
