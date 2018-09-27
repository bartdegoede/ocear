def clip_borders(image, clip_percentage=0.1):
    height, width = image.shape
    _h, _w = int(clip_percentage * height), int(clip_percentage * width)
    return image[_h:height - _h, _w:width - _w]
