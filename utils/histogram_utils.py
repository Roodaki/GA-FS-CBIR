# utils/histogram_utils.py

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.constants import HISTOGRAM_BINS, HISTOGRAM_2D_BINS


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


def compute_2d_histogram(channel1, channel2, color_space1, color_space2):
    """
    Computes a 2D histogram between two channels from potentially different color spaces.
    Args:
        channel1 (np.ndarray): First channel image.
        channel2 (np.ndarray): Second channel image.
        color_space1 (str): Color space of the first channel.
        color_space2 (str): Color space of the second channel.
    Returns:
        np.ndarray: Flattened 2D histogram.
    """
    hist = cv2.calcHist(
        [channel1, channel2],
        [0, 1],
        None,
        [HISTOGRAM_2D_BINS, HISTOGRAM_2D_BINS],
        [0, 256, 0, 256],
    )
    hist = hist.flatten()
    return hist


def plot_and_save_histograms(histograms, output_folder):
    """
    Plots 1D histograms for each color channel and saves them into the specified folder.
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


def plot_and_save_2d_histograms(histograms, output_folder):
    """
    Plots 2D histograms and saves them into the specified folder.
    """
    for key, hist in histograms.items():
        hist = hist.reshape(HISTOGRAM_BINS, HISTOGRAM_BINS)
        plt.figure(figsize=(10, 10))
        plt.imshow(hist, interpolation="nearest", cmap="hot")
        plt.title(key)
        plt.xlabel("Bin")
        plt.ylabel("Bin")
        plt.colorbar()
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

    # Flatten 1D histograms
    for color_space, histograms_dict in histograms.items():
        if isinstance(histograms_dict, dict):  # Ensure it's a dictionary
            for channel_key, histogram in histograms_dict.items():
                for bin_idx in range(HISTOGRAM_BINS):
                    row[f"{color_space}_Channel_{channel_key}_Bin_{bin_idx}"] = (
                        histogram[bin_idx]
                    )

    # Flatten 2D histograms
    for key, histogram in histograms.items():
        if isinstance(histogram, np.ndarray):  # Ensure it's a numpy array
            if len(histogram.shape) == 1:  # Check if it's a flattened histogram
                for bin_idx in range(HISTOGRAM_2D_BINS * HISTOGRAM_2D_BINS):
                    row[f"{key}_Bin_{bin_idx}"] = histogram[bin_idx]

    # Convert the row into a DataFrame
    row_df = pd.DataFrame([row])

    # Append row to the CSV file
    if not os.path.exists(csv_filename):
        # If the CSV doesn't exist, write headers (columns)
        row_df.to_csv(csv_filename, index=False, mode="w")
    else:
        # If the CSV exists, append without headers
        row_df.to_csv(csv_filename, index=False, mode="a", header=False)
