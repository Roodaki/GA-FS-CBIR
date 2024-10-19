import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
import os
import shutil
from src.constants import (
    IMAGE_DATASET_PATH,
    RETRIEVED_IMAGES_PATH,
    CSV_FILE_PATH,
    IMAGE_FILE_EXTENSION,
    K_NEIGHBORS,
    LEAF_SIZE,
)


def load_histograms_from_csv(csv_file_path, method="yeo-johnson", standardize=True):
    """
    Loads all image histograms from the CSV file, skipping the header,
    and removes columns that consist only of zeros. Applies Power Transform
    to the remaining columns.

    Args:
        csv_file_path (str): Path to the CSV file containing the histograms.
        method (str): The method used for power transformation ('box-cox' or 'yeo-johnson').
        standardize (bool): Whether to standardize the transformed data.

    Returns:
        np.ndarray: Array of transformed histograms with non-zero columns only.
    """
    # Load the CSV file into a pandas DataFrame
    histograms_df = pd.read_csv(csv_file_path)

    # Remove columns that are all zeros
    non_zero_columns_df = histograms_df.loc[:, (histograms_df != 0).any(axis=0)]

    # Instantiate the Power Transformer with specified parameters
    power_transformer = PowerTransformer(method=method, standardize=standardize)

    # Fit the transformer and transform the data
    transformed_data = power_transformer.fit_transform(non_zero_columns_df)

    # Return the transformed data as a numpy array
    return transformed_data


def retrieve_similar_images(query_histogram, histograms, k=K_NEIGHBORS):
    """
    Retrieves the most similar images based on KNN.

    Args:
        query_histogram (np.ndarray): The histogram of the query image.
        histograms (np.ndarray): Histograms of all images.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        np.ndarray: Indices of the retrieved images.
    """
    # Initialize the Nearest Neighbors model
    knn = KNeighborsClassifier(
        n_neighbors=k + 1,  # +1 to account for excluding the query image
        metric="canberra",  # Using Canberra distance
        weights="distance",  # Weight neighbors by their distance
        algorithm="ball_tree",  # Or 'auto' for the best choice
        leaf_size=LEAF_SIZE,  # Adjust leaf_size as necessary
    )
    knn.fit(histograms, np.arange(histograms.shape[0]))

    # Reshape the query histogram to 2D array
    query_histogram = query_histogram.reshape(1, -1)

    # Find the k-nearest neighbors
    distances, indices = knn.kneighbors(query_histogram)

    return indices.flatten()  # Return the indices of the nearest images


def process_image_histogram(image_index, histograms):
    """
    Extracts the histogram for a specific image based on its index in the dataset.

    Args:
        image_index (int): Index of the image in the dataset.
        histograms (np.ndarray): Array of histograms of all images.

    Returns:
        np.ndarray: The histogram of the queried image.
    """
    return histograms[image_index]  # Return the histogram of the query image


def retrieve_and_save_images_for_all_dataset():
    """
    Process each image in the dataset, retrieve similar images, and save them.
    """
    # Ensure the main output directory exists
    os.makedirs(RETRIEVED_IMAGES_PATH, exist_ok=True)

    # Load histograms from the CSV file
    histograms = load_histograms_from_csv(CSV_FILE_PATH)
    print("csv file loaded.")

    # Iterate over all images in the dataset
    for i in range(histograms.shape[0]):
        # Retrieve similar images for the current image
        query_histogram = process_image_histogram(i, histograms)
        retrieved_indices = retrieve_similar_images(query_histogram, histograms)

        # Retrieve corresponding image filenames
        image_filenames = [
            f"{index}{IMAGE_FILE_EXTENSION}" for index in retrieved_indices
        ]

        # Create a folder to save the retrieved images
        retrieval_folder = os.path.join(RETRIEVED_IMAGES_PATH, str(i))
        os.makedirs(retrieval_folder, exist_ok=True)

        # Save the retrieved images in the folder
        save_retrieved_images(image_filenames, retrieval_folder)

        print(f"Retrieved images for image {i} saved in '{retrieval_folder}'")


def save_retrieved_images(retrieved_image_filenames, output_folder):
    """
    Save the retrieved images in the specified folder.

    Args:
        retrieved_image_filenames (list): List of filenames of the retrieved images.
        output_folder (str): Directory to save the retrieved images.
    """
    for image_filename in retrieved_image_filenames:
        source_path = os.path.join(IMAGE_DATASET_PATH, image_filename)
        destination_path = os.path.join(output_folder, image_filename)

        if os.path.exists(source_path):
            shutil.copyfile(source_path, destination_path)
        else:
            print(f"Source image '{source_path}' not found. Skipping.")
