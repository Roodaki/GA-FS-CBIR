# src/retrieve_image.py

import os
import numpy as np
from utils.image_utils import read_and_resize_image, convert_image_to_color_spaces
from utils.histogram_utils import compute_histogram
from src.knn_image_retrieval import retrieve_similar_images, save_retrieved_images
from src.constants import IMAGE_FILE_EXTENSION, RETRIEVED_IMAGES_PATH


def process_input_image(image_path):
    """
    Process an input image (read, resize, convert to color spaces, compute histograms).

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Flattened histogram data of the image.
    """
    # Step 1: Read and resize the input image
    image = read_and_resize_image(image_path)

    # Step 2: Convert the image into multiple color spaces
    color_space_images = convert_image_to_color_spaces(image)

    # Step 3: Compute histograms for each color space
    all_histograms = []
    for color_space, color_space_image in color_space_images.items():
        histograms = compute_histogram(color_space_image, color_space)

        # Flatten and combine histograms for all channels
        for channel_histogram in histograms.values():
            all_histograms.extend(channel_histogram)

    return np.array(all_histograms)


def retrieve_similar_images_for_input(input_image_path):
    """
    Main function to process the input image and retrieve similar images.

    Args:
        input_image_path (str): Path to the input image.
    """
    # Ensure the input file exists
    if not os.path.exists(input_image_path) or not input_image_path.endswith(
        IMAGE_FILE_EXTENSION
    ):
        raise FileNotFoundError(
            f"Input image '{input_image_path}' does not exist or is not a valid image file."
        )

    # Step 1: Process the input image and extract its histogram
    input_histogram = process_input_image(input_image_path)

    # Step 2: Retrieve the most similar images using KNN
    retrieved_images = retrieve_similar_images(input_histogram)

    # Step 3: Save retrieved images to a new folder
    save_retrieved_images(retrieved_images)

    print(
        f"Image retrieval complete. Retrieved images are saved in '{RETRIEVED_IMAGES_PATH}'"
    )
