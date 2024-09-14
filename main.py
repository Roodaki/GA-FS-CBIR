from src.process_image import process_all_images
from src.retrieve_image import retrieve_similar_images_for_input


if __name__ == "__main__":
    # process_all_images()

    # Path to the input image
    input_image_path = "data/corel-1k/999.jpg"  # Update with your image path

    # Perform image retrieval
    retrieve_similar_images_for_input(input_image_path)
