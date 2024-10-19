# src/evaluation.py

import os
import csv
import logging
from src.constants import NUM_CLASSES, NUM_IMAGES_PER_CLASS, RETRIEVED_IMAGES_PATH
from utils.image_utils import natural_sort_key

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_ground_truth_labels() -> dict:
    """
    Loads the ground truth labels for the images based on their filenames.

    Returns:
        dict: Mapping of image filenames to their class labels.
    """
    labels = {}
    for i in range(NUM_CLASSES):
        for j in range(NUM_IMAGES_PER_CLASS):
            filename = f"{i * NUM_IMAGES_PER_CLASS + j}.jpg"
            labels[filename] = i
    return labels


def calculate_metrics(
    true_positives: int,
    false_positives: int,
    relevant_images_count: int,
    total_images: int,
) -> tuple:
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
    true_negatives = (
        total_images - relevant_images_count - false_positives - false_negatives
    )

    # Precision: Relevant retrieved / Total retrieved
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )

    # Recall: Relevant retrieved / Total relevant in class
    recall = true_positives / relevant_images_count if relevant_images_count > 0 else 0

    # F1 Score: Harmonic mean of precision and recall
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1_score, true_negatives, false_negatives


def calculate_average_precision(
    rank_csv_path: str, query_label: int, ground_truth_labels: dict
) -> float:
    """
    Calculate Average Precision (AP) for a single query based on rank.csv.

    Args:
        rank_csv_path (str): Path to the rank.csv file.
        query_label (int): The true label for the query image.
        ground_truth_labels (dict): Mapping of image filenames to their class labels.

    Returns:
        float: Average Precision (AP) for the query.
    """
    true_positives = 0
    relevant_precision_sum = 0
    total_retrieved = 0

    # Read the rank.csv file to get the retrieval results
    try:
        with open(rank_csv_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                total_retrieved += 1
                retrieved_filename = row["Retrieved Image Filename"]
                if ground_truth_labels.get(retrieved_filename) == query_label:
                    true_positives += 1
                    relevant_precision_sum += (
                        true_positives / total_retrieved
                    )  # Precision at this rank
    except FileNotFoundError:
        logging.error(f"File {rank_csv_path} not found.")
        return 0.0

    # Calculate average precision
    ap = relevant_precision_sum / true_positives if true_positives > 0 else 0
    return ap


def evaluate_all_retrievals():
    """
    Evaluate retrieval results by calculating precision, recall, F1 score, and output CSV for debugging.
    """
    ground_truth_labels = load_ground_truth_labels()
    csv_data = []
    query_metrics = []
    average_precisions = []
    total_images = len(ground_truth_labels)

    folder_names = sorted(os.listdir(RETRIEVED_IMAGES_PATH), key=natural_sort_key)

    for folder_name in folder_names:
        folder_path = os.path.join(RETRIEVED_IMAGES_PATH, folder_name)

        if os.path.isdir(folder_path):
            query_filename = f"{folder_name}.jpg"
            if query_filename not in ground_truth_labels:
                continue  # Skip if query filename not in labels

            query_label = ground_truth_labels[query_filename]
            rank_csv_path = os.path.join(folder_path, "rank.csv")
            ap = calculate_average_precision(
                rank_csv_path, query_label, ground_truth_labels
            )
            average_precisions.append(ap)

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

            true_positives = sum(
                1
                for f in retrieved_filenames
                if ground_truth_labels.get(f) == query_label
            )
            false_positives = num_retrieved - true_positives

            precision, recall, f1, true_negatives, false_negatives = calculate_metrics(
                true_positives, false_positives, relevant_images_count, total_images
            )

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
                    ap,
                )
            )

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
                    ap,
                ]
            )

    # Write the metrics to CSV
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
                "Average Precision",
            ]
        )
        writer.writerows(csv_data)

    # Calculate and print mean metrics
    total_precision = total_recall = total_f1 = total_ap = 0
    num_queries = len(query_metrics)

    for _, _, _, _, _, precision, recall, f1, ap in query_metrics:
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_ap += ap

    mean_precision = total_precision / num_queries if num_queries > 0 else 0
    mean_recall = total_recall / num_queries if num_queries > 0 else 0
    mean_f1 = total_f1 / num_queries if num_queries > 0 else 0
    mean_ap = total_ap / num_queries if num_queries > 0 else 0

    logging.info("Evaluation Metrics:")
    logging.info(f"Mean Precision: {mean_precision * 100:.2f}%")
    logging.info(f"Mean Recall: {mean_recall * 100:.2f}%")
    logging.info(f"Mean F1 Score: {mean_f1 * 100:.2f}%")
    logging.info(f"Mean Average Precision (mAP): {mean_ap * 100:.2f}%")
