# src/constants.py

# Path to Corel 1K Dataset images
IMAGE_DATASET_PATH = "data/corel-1k/"

# Path to save CSV files of histograms
HISTOGRAM_OUTPUT_PATH = "data/histograms"

# Number of bins for all histograms
HISTOGRAM_BINS = 256  # Using 256 bins for all color spaces

# Image file extension for dataset
IMAGE_FILE_EXTENSION = ".jpg"

# List of image processing constants
IMAGE_SIZE = (256, 256)  # Resize images to this size for faster processing

# Color spaces for transformation
COLOR_SPACES = {"RGB": "RGB", "HSV": "HSV", "LAB": "LAB"}

# Filename for the CSV file that stores histogram data
CSV_FILENAME = "combined_histograms.csv"
