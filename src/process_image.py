# src/process_image.py

import os
import pandas as pd
from utils.image_utils import read_and_resize_image, convert_image_to_color_spaces
from utils.histogram_utils import (
    compute_histogram,
    plot_and_save_histograms,
    append_histogram_to_csv,
)
from src.constants import (
    IMAGE_DATASET_PATH,
    HISTOGRAM_OUTPUT_PATH,
    CSV_FILENAME,
    IMAGE_FILE_EXTENSION,
)


def process_image(image_filename, csv_filename):
    """
    Processes a single image: reads, converts to color spaces, computes histograms,
    and appends the histogram to a CSV file.

    Args:
        image_filename (str): Filename of the image to process.
        csv_filename (str): Path to the CSV file to append histogram data.
    """
    try:
        image_path = os.path.join(IMAGE_DATASET_PATH, image_filename)
        image = read_and_resize_image(image_path)
        color_space_images = convert_image_to_color_spaces(image)

        # Create directories for saving histograms
        image_folder = os.path.join(
            HISTOGRAM_OUTPUT_PATH, os.path.splitext(image_filename)[0]
        )
        os.makedirs(image_folder, exist_ok=True)

        all_histograms = {}
        for color_space, color_space_image in color_space_images.items():
            # Create sub-directory for each color space
            color_space_folder = os.path.join(image_folder, color_space)
            os.makedirs(color_space_folder, exist_ok=True)

            histograms = compute_histogram(color_space_image, color_space)
            all_histograms[color_space] = histograms

            # Save each histogram as a JPG file
            plot_and_save_histograms(histograms, color_space_folder)

        # Append the histogram data to the CSV file
        append_histogram_to_csv(image_filename, all_histograms, csv_filename)

        print(f"Processed and saved histograms for {image_filename}")

    except Exception as e:
        print(f"Error processing {image_filename}: {e}")


def process_all_images():
    """
    Processes all images in the dataset directory and saves the results to a CSV file incrementally.
    """
    image_filenames = [
        f for f in os.listdir(IMAGE_DATASET_PATH) if f.endswith(IMAGE_FILE_EXTENSION)
    ]

    # CSV filename where all image histograms will be saved
    csv_filename = os.path.join(HISTOGRAM_OUTPUT_PATH, CSV_FILENAME)

    # Ensure the CSV file starts fresh (overwrite if already exists)
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    for image_filename in image_filenames:
        process_image(image_filename, csv_filename)
