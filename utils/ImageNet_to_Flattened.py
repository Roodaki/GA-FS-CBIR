import os
import shutil
from PIL import Image


def main():
    # Get paths from user input
    input_dir = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Content-Based Image Retrieval (CBIR)\Codebase (Python)\GA-Feature-Reduction-CBIR\data\Datasets\Olivia\Olivia_ImageNet"
    output_dir = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Content-Based Image Retrieval (CBIR)\Codebase (Python)\GA-Feature-Reduction-CBIR\data\Datasets\Olivia\Olivia_Flattened"

    # Validate paths
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConverting dataset from {input_dir} to flattened format in {output_dir}")
    print("This may take several minutes depending on dataset size...")

    # Supported image formats
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    counter = 0

    # Traverse through all class directories
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Process each image in class directory
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(img_extensions):
                img_path = os.path.join(class_path, img_name)

                try:
                    # Open image and convert to RGB
                    with Image.open(img_path) as img:
                        # Convert non-RGB images (like RGBA, CMYK, etc.)
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # Save as JPG with sequential numbering
                        output_path = os.path.join(output_dir, f"{counter}.jpg")
                        img.save(output_path, "JPEG", quality=95)

                    counter += 1
                    if counter % 1000 == 0:
                        print(f"Processed {counter} images...")

                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

    print(f"\nConversion complete! {counter} images copied to {output_dir}")


if __name__ == "__main__":
    main()
