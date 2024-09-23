# src/constants.py

import os

# Dataset paths
IMAGE_DATASET_PATH = "data/corel-1k/"
HISTOGRAM_OUTPUT_PATH = "data/histograms"
RETRIEVED_IMAGES_PATH = "data/retrieved_images"

# CSV file containing image histograms
CSV_FILENAME = "combined_histograms.csv"
CSV_FILE_PATH = os.path.join(HISTOGRAM_OUTPUT_PATH, CSV_FILENAME)

# Image file extension
IMAGE_FILE_EXTENSION = ".jpg"

# Image processing constants
IMAGE_SIZE = (8, 8)  # Resize images to this size for faster processing
HISTOGRAM_BINS = 8  # Number of bins for 1D histograms
HISTOGRAM_2D_BINS = 8  # Number of bins for 2D histograms

# Color spaces
COLOR_SPACES = {"RGB": "RGB", "HSV": "HSV", "LAB": "LAB"}

# KNN retrieval constants
K_NEIGHBORS = 10  # Number of images to retrieve

# Evaluation metrics constants
NUM_CLASSES = 10  # Number of classes in the dataset (0-999 with 100 per class)

# Directories
ONE_D_HISTOGRAMS_DIR = "1d_histograms"
TWO_D_HISTOGRAMS_DIR = "2d_histograms"
INTRA_COLORSPACE_DIR = "intra_colorspace"
INTER_COLORSPACE_DIR = "inter_colorspace"
