# utils/histogram_utils.py

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.constants import HISTOGRAM_BINS, HISTOGRAM_OUTPUT_PATH, COLOR_SPACES


def compute_histogram(image, color_space):
    """
    Computes and flattens the histogram of an image for a given color space.
    """
    histograms = {}
    channels = cv2.split(image)

    # Generate histograms for each channel
    for idx, channel in enumerate(channels):
        hist = cv2.calcHist([channel], [0], None, [HISTOGRAM_BINS], [0, 256]).flatten()
        histograms[f"{color_space}_Channel_{idx}"] = hist

    return histograms


def plot_and_save_histograms(histograms, output_folder):
    """
    Plots histograms for each channel and saves them as JPG files in the specified folder.
    """
    for key, hist in histograms.items():
        plt.figure(figsize=(10, 4))
        plt.plot(hist)
        plt.title(key)
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Construct file path
        histogram_filename = os.path.join(output_folder, f"{key}.jpg")
        plt.savefig(histogram_filename)
        plt.close()


def append_histogram_to_csv(image_filename, histograms, csv_filename):
    """
    Appends a single image's histogram data to the CSV file.

    Args:
        image_filename (str): Name of the image being processed.
        histograms (dict): Histogram data of the image.
        csv_filename (str): The path to the CSV file.
    """
    row = {}

    # Flatten histograms for each color space and channel
    for color_space, histograms_dict in histograms.items():
        for channel_key, histogram in histograms_dict.items():
            for bin_idx in range(HISTOGRAM_BINS):
                # Updated to remove image_filename from the column name
                row[f"{color_space}_Channel_{channel_key}_Bin_{bin_idx}"] = histogram[
                    bin_idx
                ]

    # Convert the row into a DataFrame
    row_df = pd.DataFrame([row])

    # Append row to the CSV file
    if not os.path.exists(csv_filename):
        # If the CSV doesn't exist, write headers (columns)
        row_df.to_csv(csv_filename, index=False, mode="w")
    else:
        # If the CSV exists, append without headers
        row_df.to_csv(csv_filename, index=False, mode="a", header=False)
