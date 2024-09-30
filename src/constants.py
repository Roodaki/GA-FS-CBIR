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

# LBP settings
LBP_RADIUS = 1  # Radius for LBP calculation
LBP_POINTS = 8 * LBP_RADIUS  # Points around the radius for LBP
LBP_BINS = 2**LBP_POINTS  # Number of bins for LBP histograms

# Color spaces
COLOR_SPACES = {
    "BGR": "BGR",
    "RGB": "RGB",
    "HSV": "HSV",
    "HLS": "HLS",
    "LAB": "LAB",
    "YUV": "YUV",
    "YCrCb": "YCrCb",
    "XYZ": "XYZ",
}

# KNN retrieval constants
K_NEIGHBORS = 10  # Number of images to retrieve
LEAF_SIZE = 30  # Leaf size for KNN (used in image retrieval)

# Evaluation metrics constants
NUM_CLASSES = 10  # Number of classes in the dataset (0-999 with 100 per class)

# Directories
ONE_D_HISTOGRAMS_DIR = "1d_histograms"
TWO_D_HISTOGRAMS_DIR = "2d_histograms"
INTRA_COLORSPACE_DIR = "intra_colorspace"
INTER_COLORSPACE_DIR = "inter_colorspace"

# Genetic Algorithm Constants
GA_POPULATION_SIZE = 50  # Number of individuals in each generation
GA_NUMBER_OF_GENERATIONS = 100  # Number of generations to evolve
GA_CROSSOVER_PROBABILITY = 0.8  # Probability of crossover between individuals
GA_MUTATION_PROBABILITY = 0.2  # Probability of mutation in individuals
GA_MUTATION_INDEPENDENCE_PROBABILITY = 0.05  # Probability for each gene to mutate
GA_PRECISION_WEIGHT = 0.5  # Higher value prioritizes precision; lower value prioritizes minimizing features
TOURNAMENT_SIZE = 3  # Tournament size for selection
CROSSOVER_INDP_PROBABILITY = 0.6  # Probability for independent crossover per gene
