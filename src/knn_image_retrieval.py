# src/knn_image_retrieval.py

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import canberra  # Import Canberra distance
from src.constants import (
    CSV_FILE_PATH,
    IMAGE_DATASET_PATH,
    RETRIEVED_IMAGES_PATH,
    K_NEIGHBORS,
)


def row_index_to_filename(row_index):
    """
    Convert the row index to the corresponding image filename.

    Args:
        row_index (int): Index of the row in the CSV file.

    Returns:
        str: The corresponding image filename (e.g., "0.jpg" for row 2).
    """
    image_filename = f"{row_index - 1}.jpg"
    return image_filename


def canberra_distance(u, v):
    """
    Custom function to calculate the Canberra distance.

    Args:
        u (np.ndarray): First vector.
        v (np.ndarray): Second vector.

    Returns:
        float: Canberra distance between u and v.
    """
    return canberra(u, v)


def retrieve_similar_images(
    input_histogram, num_neighbors=K_NEIGHBORS, distance_metric=canberra_distance
):
    """
    Retrieve the most similar images using KNN based on the input image's histogram.

    Args:
        input_histogram (np.ndarray): Histogram of the input image.
        num_neighbors (int): Number of nearest neighbors to retrieve.
        distance_metric (callable): Custom distance function, e.g., Canberra distance.

    Returns:
        list of str: List of filenames for the retrieved images.
    """
    # Load precomputed histograms from the CSV file
    histogram_data = pd.read_csv(CSV_FILE_PATH)

    # Extract histogram data (ignoring the first 2 rows)
    histograms = histogram_data.iloc[1:, :].values

    # Initialize KNN model with a custom metric (Canberra distance)
    knn_model = NearestNeighbors(n_neighbors=num_neighbors, metric=distance_metric)
    knn_model.fit(histograms)

    # Find K nearest neighbors
    distances, indices = knn_model.kneighbors([input_histogram])

    # Map indices to filenames
    retrieved_filenames = [row_index_to_filename(idx) for idx in indices[0]]
    return retrieved_filenames


def save_retrieved_images(retrieved_image_filenames):
    """
    Save the retrieved images in a separate folder for visualization.

    Args:
        retrieved_image_filenames (list): List of filenames of the retrieved images.
    """
    os.makedirs(RETRIEVED_IMAGES_PATH, exist_ok=True)

    for image_filename in retrieved_image_filenames:
        source_path = os.path.join(IMAGE_DATASET_PATH, image_filename)
        destination_path = os.path.join(RETRIEVED_IMAGES_PATH, image_filename)
        shutil.copyfile(source_path, destination_path)

    print(f"Saved {len(retrieved_image_filenames)} images in '{RETRIEVED_IMAGES_PATH}'")
