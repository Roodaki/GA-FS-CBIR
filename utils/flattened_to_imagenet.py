import os
import shutil

# --- USER INPUT ---

# 1. Path to the folder with your original flattened images
flattened_dataset_path = "data\Datasets\Corel-10K"

# 2. Path where the new ImageNet-style dataset will be created
new_dataset_path = "data\Datasets\Corel-10K-ImageNet"

# 3. The total number of classes
num_classes = 100  # Example: 10 classes

# 4. The number of images in each class (since it's balanced)
images_per_class_count = 100  # Example: 50 images per class

# 5. The file extension of your images (e.g., '.jpg', '.png')
image_extension = ".jpeg"

# --- END OF USER INPUT ---


def create_imagenet_style_dataset():
    """
    Organizes a flattened, balanced image dataset into an ImageNet-style directory structure.
    """
    # --- Validation ---
    if not os.path.isdir(flattened_dataset_path):
        print(
            f"❌ Error: The source directory '{flattened_dataset_path}' does not exist."
        )
        return

    # Automatically create the list of image counts for each class
    images_per_class_list = [images_per_class_count] * num_classes
    total_images_to_process = sum(images_per_class_list)

    print(f"ℹ️  Dataset is balanced.")
    print(f"ℹ️  Total number of classes: {num_classes}")
    print(f"ℹ️  Images per class: {images_per_class_count}")
    print(f"ℹ️  Total number of images to process: {total_images_to_process}")

    # --- Directory Creation ---
    print(f"\nCreating new dataset directory at: '{new_dataset_path}'")
    os.makedirs(new_dataset_path, exist_ok=True)

    for i in range(num_classes):
        class_dir = os.path.join(new_dataset_path, f"{i}")
        os.makedirs(class_dir, exist_ok=True)
    print("✅ Class directories created successfully.")

    # --- Image Copying Process ---
    print("\nStarting to copy images...")
    image_counter = 0
    # Iterate through each class index
    for class_index in range(num_classes):
        print(f"  -> Processing class_{class_index}...")

        # For each class, copy the correct number of images
        for _ in range(images_per_class_count):
            source_image_name = str(image_counter) + image_extension
            source_path = os.path.join(flattened_dataset_path, source_image_name)

            destination_dir = os.path.join(new_dataset_path, f"{class_index}")
            destination_path = os.path.join(destination_dir, source_image_name)

            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
            else:
                print(f"  ⚠️ Warning: Source image not found, skipping: {source_path}")

            image_counter += 1

    print("\n🎉 Dataset reorganization complete!")
    print(f"Total images copied: {image_counter}")


if __name__ == "__main__":
    create_imagenet_style_dataset()
