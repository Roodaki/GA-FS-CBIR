import pandas as pd
from sklearn_extra.cluster import KMedoids  # <-- The key change is here
import numpy as np


def select_prototypes_by_row_order(
    file_path,
    n_classes,
    n_images_per_class,
    n_prototypes,
    output_path="prototypes.csv",
    has_header=False,
):
    """
    Selects prototypes from a CSV file using K-Medoids clustering.
    The prototypes will be actual data points (medoids) from the original dataset.

    Args:
        file_path (str): The path to the input CSV file.
        n_classes (int): The total number of classes in the dataset.
        n_images_per_class (int): The number of images (rows) for each class.
        n_prototypes (int): The number of prototypes (K) to select for each class.
        output_path (str): The path to save the resulting CSV file of prototypes.
        has_header (bool): Set to True if the CSV file has a header row, False otherwise.
    """
    try:
        # --- 1. Load Data and Perform Checks ---
        print(f"Loading data from '{file_path}'...")
        header_option = "infer" if has_header else None
        df = pd.read_csv(file_path, header=header_option)

        expected_rows = n_classes * n_images_per_class
        actual_rows = len(df)
        if actual_rows != expected_rows:
            print(
                f"Error: The number of rows in the CSV ({actual_rows}) does not match "
                f"the expected number based on classes and images per class ({expected_rows})."
            )
            return

        print(f"Data loaded successfully. Found {actual_rows} rows.")
        all_prototypes_list = []

        # --- 2. Iterate Through Classes and Select Prototypes ---
        for i in range(n_classes):
            print(f"Processing class {i+1}/{n_classes}...")

            start_index = i * n_images_per_class
            end_index = start_index + n_images_per_class
            features = df.iloc[start_index:end_index]

            # Use KMedoids to find the most central ACTUAL data points
            kmedoids = KMedoids(
                n_clusters=n_prototypes,
                random_state=42,
                metric="canberra",
                init="k-medoids++",
            )
            kmedoids.fit(features)

            # The .cluster_centers_ attribute of KMedoids returns the medoids,
            # which are actual samples from the 'features' dataframe.
            class_prototypes = kmedoids.cluster_centers_
            all_prototypes_list.append(class_prototypes)

        # --- 3. Combine and Save Prototypes ---
        final_prototypes_array = np.vstack(all_prototypes_list)
        final_prototypes_df = pd.DataFrame(final_prototypes_array)

        if has_header:
            final_prototypes_df.columns = df.columns

        final_prototypes_df.to_csv(output_path, index=False, header=has_header)

        print(
            f"\nSuccessfully selected {len(final_prototypes_df)} prototypes (medoids)."
        )
        print(
            f"Prototypes saved to '{output_path}'. Each prototype is an actual image from the dataset."
        )

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ImportError:
        print("\nError: The 'sklearn_extra' library is required for K-Medoids.")
        print("Please install it by running: pip install scikit-learn-extra")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- PLEASE CUSTOMIZE THESE PARAMETERS ---

    # 1. The path to your input CSV file (which is ordered by class)
    input_csv_path = "data\out\histograms\corel10k_combined_histograms.csv"

    # 2. The path where you want to save the output CSV of prototypes
    output_csv_path = "data\out\histograms\corel10k_combined_histograms_prototypes.csv"

    # 3. The total number of classes in the dataset
    num_classes = 100

    # 4. The number of images (rows) in each class
    num_images_per_class = 100

    # 5. The number of prototypes you want to generate for each class
    num_prototypes_per_class = 20

    # 6. Does your CSV file have a header row? (True or False)
    csv_has_header = False

    # --- END OF CUSTOMIZATION ---

    select_prototypes_by_row_order(
        file_path=input_csv_path,
        n_classes=num_classes,
        n_images_per_class=num_images_per_class,
        n_prototypes=num_prototypes_per_class,
        output_path=output_csv_path,
        has_header=csv_has_header,
    )
