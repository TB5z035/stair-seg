import cv2
import numpy as np
from PIL import Image


def arr2Image(arr: np.ndarray) -> Image.Image:
    # Convert float to uint8
    im = np.copy(arr)
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # TODO: rgb images
    return Image.fromarray(arr.astype(np.uint8))
