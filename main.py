# src/main.py

from src.process_image import process_all_images
from src.knn_image_retrieval import retrieve_and_save_images_for_all_dataset
from src.evaluation import evaluate_all_retrievals

if __name__ == "__main__":
    # Process all images to compute histograms
    # process_all_images()

    # Retrieve and save images for each image in the dataset
    # retrieve_and_save_images_for_all_dataset()

    # Evaluate the retrieval performance
    evaluate_all_retrievals()
