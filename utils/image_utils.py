# utils/image_utils.py

import cv2
import os
import re
from src.constants import IMAGE_SIZE, COLOR_SPACES


def read_and_resize_image(image_path):
    """
    Reads an image from the given path and resizes it to IMAGE_SIZE.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    resized_image = cv2.resize(image, IMAGE_SIZE)
    return resized_image


def convert_image_to_color_spaces(image):
    """
    Converts the given image into different color spaces.
    """
    color_space_images = {}
    color_space_images[COLOR_SPACES["RGB"]] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_space_images[COLOR_SPACES["HSV"]] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_space_images[COLOR_SPACES["LAB"]] = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return color_space_images


def natural_sort_key(filename):
    """
    Returns a sort key for natural sorting.

    Args:
        filename (str): Filename to extract the numeric part for sorting.

    Returns:
        A tuple that can be used to sort filenames in natural order.
    """
    base = os.path.splitext(filename)[0]  # Remove extension
    return [
        int(part) if part.isdigit() else part for part in re.split("([0-9]+)", base)
    ]
