import os
import cv2
import re
from utils.image_utils import (
    read_and_resize_image,
    convert_image_to_color_spaces,
    natural_sort_key,
)
from utils.histogram_utils import (
    compute_histogram,
    compute_2d_histogram,
    compute_lbp_histogram,
    plot_and_save_histograms,
    plot_and_save_2d_histograms,
    append_histogram_to_csv,
)
from src.constants import (
    IMAGE_DATASET_PATH,
    HISTOGRAM_OUTPUT_PATH,
    CSV_FILENAME,
    IMAGE_FILE_EXTENSION,
    COLOR_SPACES,
    ONE_D_HISTOGRAMS_DIR,
    TWO_D_HISTOGRAMS_DIR,
    INTRA_COLORSPACE_DIR,
    INTER_COLORSPACE_DIR,
)


def process_image(image_filename, csv_filename):
    """
    Processes a single image: reads, converts to color spaces, computes histograms,
    and appends the histogram to a CSV file.

    Args:
        image_filename (str): Filename of the image to process.
        csv_filename (str): Path to the CSV file to append histogram data.
    """
    try:
        # Read the image and convert to different color spaces
        image_path = os.path.join(IMAGE_DATASET_PATH, image_filename)
        image = read_and_resize_image(image_path)
        color_space_images = convert_image_to_color_spaces(image)

        # Create directory for saving histograms based on the image filename
        image_folder = os.path.join(
            HISTOGRAM_OUTPUT_PATH, os.path.splitext(image_filename)[0]
        )
        os.makedirs(image_folder, exist_ok=True)

        # Create directories for 1D and 2D histograms
        one_d_histograms_folder = os.path.join(image_folder, ONE_D_HISTOGRAMS_DIR)
        two_d_histograms_folder = os.path.join(image_folder, TWO_D_HISTOGRAMS_DIR)

        os.makedirs(one_d_histograms_folder, exist_ok=True)
        os.makedirs(two_d_histograms_folder, exist_ok=True)

        # Create folders for color spaces within 1D histogram folder
        for color_space in COLOR_SPACES.values():
            os.makedirs(
                os.path.join(one_d_histograms_folder, color_space), exist_ok=True
            )

        # Create folders for intra and inter color space 2D histograms
        intra_colorspace_folder = os.path.join(
            two_d_histograms_folder, INTRA_COLORSPACE_DIR
        )
        inter_colorspace_folder = os.path.join(
            two_d_histograms_folder, INTER_COLORSPACE_DIR
        )

        os.makedirs(intra_colorspace_folder, exist_ok=True)
        os.makedirs(inter_colorspace_folder, exist_ok=True)

        for color_space in COLOR_SPACES.values():
            os.makedirs(
                os.path.join(intra_colorspace_folder, color_space), exist_ok=True
            )

        # Compute and save 1D histograms
        all_histograms = {}
        for color_space, color_space_image in color_space_images.items():
            histograms = compute_histogram(color_space_image, color_space)
            all_histograms[color_space] = histograms

            # Save each histogram as a JPG file
            plot_and_save_histograms(
                histograms, os.path.join(one_d_histograms_folder, color_space)
            )

        # Compute LBP histogram and save to the 1D histograms folder
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp_histogram = compute_lbp_histogram(grayscale_image)

        # Save LBP histogram as an image
        lbp_histograms = {"LBP_Histogram": lbp_histogram}
        lbp_folder = os.path.join(one_d_histograms_folder, "LBP")
        os.makedirs(lbp_folder, exist_ok=True)
        plot_and_save_histograms(lbp_histograms, lbp_folder)

        # Include LBP histogram in the combined_histograms dictionary
        all_histograms["LBP_Histogram"] = lbp_histogram

        # Compute and save 2D histograms (intra-color space)
        all_2d_histograms = {}
        for color_space in COLOR_SPACES.values():
            for channel1 in range(3):
                for channel2 in range(channel1 + 1, 3):  # Avoid redundant combinations
                    hist_key = f"{color_space}_Channel_{channel1}_vs_{color_space}_Channel_{channel2}"
                    channel1_image = color_space_images[color_space][:, :, channel1]
                    channel2_image = color_space_images[color_space][:, :, channel2]
                    hist_2d = compute_2d_histogram(
                        channel1_image, channel2_image, color_space, color_space
                    )
                    all_2d_histograms[hist_key] = hist_2d

        # Save intra-color space 2D histograms
        for color_space in COLOR_SPACES.values():
            intra_color_space_histograms = {
                k: v for k, v in all_2d_histograms.items() if f"{color_space}" in k
            }
            plot_and_save_2d_histograms(
                intra_color_space_histograms,
                os.path.join(intra_colorspace_folder, color_space),
            )

        # Compute and save inter-color space 2D histograms
        all_inter_2d_histograms = {}
        for cs1_idx, cs1 in enumerate(COLOR_SPACES.values()):
            for cs2_idx, cs2 in enumerate(COLOR_SPACES.values()):
                if cs1_idx < cs2_idx:  # Ensure only one combination is calculated
                    for channel1 in range(3):
                        for channel2 in range(3):
                            hist_key = (
                                f"{cs1}_Channel_{channel1}_vs_{cs2}_Channel_{channel2}"
                            )
                            channel1_image = color_space_images[cs1][:, :, channel1]
                            channel2_image = color_space_images[cs2][:, :, channel2]
                            hist_2d = compute_2d_histogram(
                                channel1_image, channel2_image, cs1, cs2
                            )
                            all_inter_2d_histograms[hist_key] = hist_2d

        # Save inter-color space 2D histograms
        for cs1_idx, cs1 in enumerate(COLOR_SPACES.values()):
            for cs2_idx, cs2 in enumerate(COLOR_SPACES.values()):
                if cs1_idx < cs2_idx:  # Avoid redundancy in inter-color space as well
                    folder_name = f"{cs1}_vs_{cs2}"
                    inter_color_space_folder_path = os.path.join(
                        inter_colorspace_folder, folder_name
                    )
                    os.makedirs(inter_color_space_folder_path, exist_ok=True)
                    inter_color_space_histograms = {
                        k: v
                        for k, v in all_inter_2d_histograms.items()
                        if f"{cs1}_Channel_" in k and f"vs_{cs2}_Channel_" in k
                    }
                    plot_and_save_2d_histograms(
                        inter_color_space_histograms,
                        inter_color_space_folder_path,
                    )

        # Combine histograms
        combined_histograms = {
            **all_histograms,
            **all_2d_histograms,
            **all_inter_2d_histograms,
        }

        # Append histogram data to CSV
        append_histogram_to_csv(image_filename, combined_histograms, csv_filename)

        print(f"Processed and saved histograms for {image_filename}")

    except Exception as e:
        print(f"Error processing {image_filename}: {e}")


def process_all_images():
    """
    Processes all images in the dataset directory and saves the results to a CSV file incrementally.
    """
    image_filenames = [
        f for f in os.listdir(IMAGE_DATASET_PATH) if f.endswith(IMAGE_FILE_EXTENSION)
    ]

    # Sort the filenames in natural order
    image_filenames.sort(key=natural_sort_key)

    # CSV filename where all image histograms will be saved
    csv_filename = os.path.join(HISTOGRAM_OUTPUT_PATH, CSV_FILENAME)

    # Ensure the CSV file starts fresh (overwrite if already exists)
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    for image_filename in image_filenames:
        process_image(image_filename, csv_filename)
