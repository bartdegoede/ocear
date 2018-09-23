import random

from ocear.preprocess.utils import clip_borders


def test_clip_borders(img):
    h, w = img.shape
    for clip_percentage in [0.0, 0.1, 1.0]:
        _h, _w = int(clip_percentage * h), int(clip_percentage * w)
        _img = img[_h : h - _h, _w : w - _w]
        clipped_img = clip_borders(img, clip_percentage=clip_percentage)
        assert clipped_img.shape[0] == _img.shape[0]
        assert clipped_img.shape[1] == _img.shape[1]
