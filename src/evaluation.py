# src/evaluation.py

import os
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from src.constants import NUM_CLASSES, RETRIEVED_IMAGES_PATH, IMAGE_DATASET_PATH


def load_ground_truth_labels():
    """
    Loads the ground truth labels for the images based on their filenames.

    Returns:
        dict: Mapping of image filenames to their class labels.
    """
    labels = {}
    for i in range(NUM_CLASSES):
        class_start = i * 100
        class_end = class_start + 100
        for j in range(class_start, class_end):
            filename = f"{j}.jpg"
            labels[filename] = i
    return labels


def compute_evaluation_metrics(retrieved_images_by_query):
    """
    Compute precision, recall, F1 score, and mean Average Precision (mAP) for the retrieval system.

    Args:
        retrieved_images_by_query (dict): A dictionary where keys are query filenames
                                          and values are lists of retrieved image filenames.

    Returns:
        dict: A dictionary with precision, recall, F1 score, and mAP.
    """
    ground_truth_labels = load_ground_truth_labels()

    # Prepare lists for metrics calculation
    y_true = []
    y_pred = []

    # Iterate through each query
    for query_filename, retrieved_filenames in retrieved_images_by_query.items():
        query_label = ground_truth_labels[query_filename]

        # Create binary vectors for ground truth and predictions
        true_labels = [
            1 if ground_truth_labels.get(img) == query_label else 0
            for img in retrieved_filenames
        ]
        predicted_labels = [1] * len(
            true_labels
        )  # Since we assume all retrieved images are positive

        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)

    # Compute precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Compute mean Average Precision (mAP)
    average_precision = average_precision_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_average_precision": average_precision,
    }


def evaluate_all_retrievals():
    """
    Evaluate retrieval results by calculating precision, recall, F1 score, and mAP.
    """
    retrieved_images_by_query = {}

    for folder_name in os.listdir(RETRIEVED_IMAGES_PATH):
        folder_path = os.path.join(RETRIEVED_IMAGES_PATH, folder_name)

        if os.path.isdir(folder_path):
            query_filename = f"{folder_name}.jpg"
            retrieved_filenames = [
                f for f in os.listdir(folder_path) if f.endswith(".jpg")
            ]
            retrieved_images_by_query[query_filename] = retrieved_filenames

    metrics = compute_evaluation_metrics(retrieved_images_by_query)

    print("Evaluation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Mean Average Precision (mAP): {metrics['mean_average_precision']:.4f}")
