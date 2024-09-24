# src/evaluation.py

import os
import csv
from src.constants import NUM_CLASSES, RETRIEVED_IMAGES_PATH
from utils.image_utils import natural_sort_key


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


def calculate_metrics(
    true_positives, false_positives, relevant_images_count, total_images
):
    """
    Calculate precision, recall, F1 score, true negatives, and false negatives.

    Args:
        true_positives (int): Number of true positive images.
        false_positives (int): Number of false positive images.
        relevant_images_count (int): Number of images in the query's class.
        total_images (int): Total number of images in the dataset.

    Returns:
        tuple: precision, recall, F1 score, true negatives, false negatives.
    """
    # False Negatives: Correct class images not retrieved
    false_negatives = relevant_images_count - true_positives

    # True Negatives: Total images not retrieved and not in the same class
    true_negatives = total_images - relevant_images_count - false_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = true_positives / relevant_images_count if relevant_images_count > 0 else 0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1_score, true_negatives, false_negatives


def evaluate_all_retrievals():
    """
    Evaluate retrieval results by calculating precision, recall, F1 score, and output CSV for debugging.
    """
    ground_truth_labels = load_ground_truth_labels()

    # Initialize data structures
    csv_data = []
    query_metrics = []

    total_images = len(ground_truth_labels)

    # Ensure folders are sorted naturally
    folder_names = sorted(os.listdir(RETRIEVED_IMAGES_PATH), key=natural_sort_key)

    for folder_name in folder_names:
        folder_path = os.path.join(RETRIEVED_IMAGES_PATH, folder_name)

        if os.path.isdir(folder_path):
            query_filename = f"{folder_name}.jpg"
            if query_filename not in ground_truth_labels:
                continue  # Skip if query filename not in labels

            query_label = ground_truth_labels[query_filename]
            retrieved_filenames = [
                f
                for f in os.listdir(folder_path)
                if f.endswith(".jpg") and f != query_filename
            ]
            num_retrieved = len(retrieved_filenames)
            relevant_images_count = sum(
                1
                for f in ground_truth_labels
                if ground_truth_labels.get(f) == query_label
            )

            # True Positives: Correct class images retrieved
            true_positives = sum(
                1
                for f in retrieved_filenames
                if ground_truth_labels.get(f) == query_label
            )
            # False Positives: Incorrect class images retrieved
            false_positives = num_retrieved - true_positives

            precision, recall, f1, true_negatives, false_negatives = calculate_metrics(
                true_positives, false_positives, relevant_images_count, total_images
            )

            # Append metrics for query image
            query_metrics.append(
                (
                    query_filename,
                    true_positives,
                    false_positives,
                    true_negatives,
                    false_negatives,
                    precision,
                    recall,
                    f1,
                )
            )

            # Append data for CSV
            csv_data.append(
                [
                    query_filename,
                    true_positives,
                    true_negatives,
                    false_positives,
                    false_negatives,
                    precision,
                    recall,
                    f1,
                ]
            )

    # Write metrics to CSV
    csv_file_path = os.path.join(RETRIEVED_IMAGES_PATH, "evaluation_metrics.csv")
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Image",
                "True Positive",
                "True Negative",
                "False Positive",
                "False Negative",
                "Precision",
                "Recall",
                "F1 Score",
            ]
        )
        writer.writerows(csv_data)

    # Calculate and print average metrics
    total_precision = total_recall = total_f1 = 0
    num_queries = len(query_metrics)

    for _, _, _, _, _, precision, recall, f1 in query_metrics:
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    avg_precision = total_precision / num_queries if num_queries > 0 else 0
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    avg_f1 = total_f1 / num_queries if num_queries > 0 else 0

    print("Evaluation Metrics:")
    print(f"Average Precision: {avg_precision * 100:.2f}%")
    print(f"Average Recall: {avg_recall * 100:.2f}%")
    print(f"Average F1 Score: {avg_f1 * 100:.2f}%")
