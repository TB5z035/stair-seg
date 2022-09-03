import cv2
import numpy as np
from PIL import Image


def arr2Image(arr: np.ndarray, normalize=True) -> Image.Image:
    # Convert float to uint8
    im = np.copy(arr)
    normalize = True if arr.dtype != np.uint8 else normalize
    if normalize:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # TODO: rgb images
    return Image.fromarray(im.astype(np.uint8))
