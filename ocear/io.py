from pathlib import Path
import imageio


def load_image(image_path):
    """
    Load grayscale image
    """
    path = Path(image_path)
    if not path.is_file():
        raise Exception('No image found at "{}"'.format(image_path))
    img = imageio.imread(path)
    return img[:, :, 1]
