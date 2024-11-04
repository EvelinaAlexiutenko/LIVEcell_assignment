import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from cfg import GLOBAL_MEAN, GLOBAL_STD, SIZE


def normalize_images(image, mean=GLOBAL_MEAN, std=GLOBAL_STD):
    return (image - mean) / std


def preprocess_image(image, size=SIZE, mean=GLOBAL_MEAN, std=GLOBAL_STD):
    # Load image and resize
    image = image.resize((size, size))
    image = np.array(image)

    # Normalize the image
    image = normalize_images(image, mean, std)
    return image
