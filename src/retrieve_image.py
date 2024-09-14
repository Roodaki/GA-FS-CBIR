import os
import numpy as np
import shutil
from utils.image_utils import read_and_resize_image, convert_image_to_color_spaces
from utils.histogram_utils import compute_histogram
from src.knn_image_retrieval import retrieve_similar_images, save_retrieved_images
from src.constants import (
    IMAGE_FILE_EXTENSION,
    RETRIEVED_IMAGES_PATH,
    IMAGE_DATASET_PATH,
)


def process_image_histogram(image_path):
    """
    Process an image to compute its histogram data.

    Args:
        image_path (str): Path to the image to process.

    Returns:
        np.ndarray: Flattened histogram data of the image.
    """
    # Step 1: Read and resize the image
    image = read_and_resize_image(image_path)

    # Step 2: Convert the image into various color spaces
    color_space_images = convert_image_to_color_spaces(image)

    # Step 3: Compute histograms for each color space and channel
    combined_histogram = []
    for color_space, color_space_image in color_space_images.items():
        histograms = compute_histogram(color_space_image, color_space)
        for histogram in histograms.values():
            combined_histogram.extend(histogram)

    return np.array(combined_histogram)


def retrieve_and_save_images_for_all_dataset():
    """
    Process each image in the dataset to retrieve similar images.
    Creates a separate folder for each image's retrieved results.
    """
    # Ensure the main output directory exists
    os.makedirs(RETRIEVED_IMAGES_PATH, exist_ok=True)

    # Get the list of all image files in the dataset
    image_filenames = [
        f for f in os.listdir(IMAGE_DATASET_PATH) if f.endswith(IMAGE_FILE_EXTENSION)
    ]

    for image_filename in image_filenames:
        # Path to the current image
        image_path = os.path.join(IMAGE_DATASET_PATH, image_filename)

        # Ensure the input file exists
        if not os.path.exists(image_path):
            print(f"Image file '{image_path}' does not exist. Skipping.")
            continue

        # Step 1: Process the image and extract its histogram
        image_histogram = process_image_histogram(image_path)

        # Step 2: Retrieve similar images based on the histogram
        retrieved_image_filenames = retrieve_similar_images(image_histogram)

        # Step 3: Create a folder for the retrieved images of the current image
        retrieval_folder = os.path.join(
            RETRIEVED_IMAGES_PATH, os.path.splitext(image_filename)[0]
        )
        os.makedirs(retrieval_folder, exist_ok=True)

        # Step 4: Save the retrieved images into the respective folder
        save_retrieved_images(retrieved_image_filenames, retrieval_folder)

        print(
            f"Retrieved images for '{image_filename}' are saved in '{retrieval_folder}'"
        )


def save_retrieved_images(retrieved_image_filenames, output_folder):
    """
    Save the retrieved images in the specified folder.

    Args:
        retrieved_image_filenames (list): List of filenames of the retrieved images.
        output_folder (str): Directory to save the retrieved images.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for image_filename in retrieved_image_filenames:
        source_path = os.path.join(IMAGE_DATASET_PATH, image_filename)
        destination_path = os.path.join(output_folder, image_filename)

        if os.path.exists(source_path):
            shutil.copyfile(source_path, destination_path)
        else:
            print(f"Source image '{source_path}' not found. Skipping.")

    print(f"Saved {len(retrieved_image_filenames)} images in '{output_folder}'")
