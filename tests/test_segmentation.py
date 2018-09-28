from ocear.segmentation import estimate_scale


def test_estimate_scale(binarized_img):
    assert 0.0 <= estimate_scale(1-binarized_img) <= 100.0
