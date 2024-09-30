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
    Converts the given image into different color spaces defined in COLOR_SPACES.

    Args:
        image (np.ndarray): The input image in BGR format.

    Returns:
        dict: A dictionary mapping color space names to their corresponding images.
    """
    color_space_images = {}

    for color_space in COLOR_SPACES:
        if color_space == "BGR":
            color_space_images[COLOR_SPACES["BGR"]] = image  # BGR is the original image
        elif color_space == "RGB":
            color_space_images[COLOR_SPACES["RGB"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )
        elif color_space == "HSV":
            color_space_images[COLOR_SPACES["HSV"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2HSV
            )
        elif color_space == "HLS":
            color_space_images[COLOR_SPACES["HLS"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2HLS
            )
        elif color_space == "LAB":
            color_space_images[COLOR_SPACES["LAB"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2Lab
            )
        elif color_space == "YUV":
            color_space_images[COLOR_SPACES["YUV"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2YUV
            )
        elif color_space == "YCrCb":
            color_space_images[COLOR_SPACES["YCrCb"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2YCrCb
            )
        elif color_space == "XYZ":
            color_space_images[COLOR_SPACES["XYZ"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2XYZ
            )
        elif color_space == "Gray":
            color_space_images[COLOR_SPACES["Gray"]] = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

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
