# extract_features_all_variants_torch.py

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Generator
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from constants import IMAGE_DATASET_PATH, HISTOGRAM_OUTPUT_PATH

# ###########################################################################
# ### MODEL CONFIGURATION ###
# ###########################################################################
MODEL_NAMES = [
    # VGG Family
    #     "vgg11",
    #     "vgg11_bn",
    #     "vgg13",
    #     "vgg13_bn",
    "vgg16",
    #     "vgg16_bn",
    "vgg19",
    #     "vgg19_bn",
    # ResNet Family and its derivatives
    #     "resnet18",
    #     "resnet34",
    "resnet50",
    #     "resnet101",
    #     "resnet152",
    #     "resnext50_32x4d",
    #     "resnext101_32x8d",
    #     "wide_resnet50_2",
    #     "wide_resnet101_2",
    # DenseNet Family
    "densenet121",
    #     "densenet161",
    #     "densenet169",
    #     "densenet201",
    # EfficientNet Family
    "efficientnet_b0",
    #     "efficientnet_b1",
    #     "efficientnet_b2",
    #     "efficientnet_b3",
    #     "efficientnet_b4",
    #     "efficientnet_b5",
    #     "efficientnet_b6",
    #     "efficientnet_b7",
    #     "efficientnet_v2_s",
    #     "efficientnet_v2_m",
    #     "efficientnet_v2_l",
    # MobileNet Family
    #     "mobilenet_v2",
    #     "mobilenet_v3_small",
    "mobilenet_v3_large",
    # Modern CNN Family
    # "convnext_tiny",
    # "convnext_small",
    "convnext_base",
    # "convnext_large",
]

# Adjust batch size if you encounter GPU memory errors, especially for larger models.
BATCH_SIZE = 32
# ###########################################################################


def get_device() -> torch.device:
    """Checks for a CUDA-enabled GPU and returns the appropriate torch device."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✅ Found GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(
            "⚠️ No GPU found. The script will run on the CPU, which will be much slower."
        )
    return device


def load_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Loads a pre-trained model, modifies it for feature extraction, and moves it to the device.
    This function dynamically handles different model families.
    """
    print(f"\nLoading {model_name.upper()} model...")

    # Dynamically get the model constructor from torchvision.models
    # E.g., model_constructor becomes models.resnet50
    model_constructor = getattr(models, model_name)
    base_model = model_constructor(weights="IMAGENET1K_V1")

    # Define feature extractor based on model family's architecture
    if model_name.startswith("vgg"):
        feature_extractor = nn.Sequential(
            base_model.features,
            nn.Flatten(),
            *list(base_model.classifier.children())[:-1],
        )
    elif (
        model_name.startswith("resnet")
        or model_name.startswith("resnext")
        or model_name.startswith("wide_resnet")
    ):
        feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1], nn.Flatten()
        )

    elif model_name.startswith("densenet"):
        feature_extractor = nn.Sequential(
            base_model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )

    elif (
        model_name.startswith("efficientnet")
        or model_name.startswith("mobilenet")
        or model_name.startswith("convnext")
    ):
        # These models share a similar structure: features -> avgpool -> classifier
        feature_extractor = nn.Sequential(
            base_model.features, base_model.avgpool, nn.Flatten()
        )
    else:
        raise ValueError(
            f"Model family for '{model_name}' is not recognized or handled."
        )

    feature_extractor.to(device)
    feature_extractor.eval()

    print(f"Model {model_name.upper()} loaded successfully and set to evaluation mode.")
    return feature_extractor


def get_preprocess_transform() -> transforms.Compose:
    """Returns the standard preprocessing pipeline for 224x224 ImageNet models."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def batch_generator(
    image_paths: List[str], batch_size: int, transform: transforms.Compose
) -> Generator[torch.Tensor, None, None]:
    """A generator that yields batches of preprocessed image tensors."""
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception as e:
                print(f"\nSkipping malformed image {img_path}. Error: {e}")

        if batch_tensors:
            yield torch.stack(batch_tensors)


def process_folder_batched(
    model: nn.Module, folder_path: str, batch_size: int, device: torch.device
) -> List[np.ndarray]:
    """Processes all images in a folder to extract their raw (un-normalized) features."""
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    image_paths = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(supported_formats)
        ]
    )

    if not image_paths:
        print(f"No images found in '{folder_path}'.")
        return []

    print(
        f"\nFound {len(image_paths)} images. Processing in batches of {batch_size}..."
    )

    all_features = []
    preprocess = get_preprocess_transform()
    batches = batch_generator(image_paths, batch_size, preprocess)
    num_batches = math.ceil(len(image_paths) / batch_size)

    with torch.no_grad():
        for batch in tqdm(batches, total=num_batches, desc="Extracting Features"):
            batch = batch.to(device)
            # Get the raw feature vectors
            batch_features = model(batch)
            # Add features to the list
            all_features.extend(batch_features.cpu().numpy())

    return all_features


def save_features_to_csv(features_list: List[np.ndarray], output_path: str):
    """Saves the extracted features to a CSV file without a header or index."""
    if not features_list:
        print("No features were extracted. The CSV file will not be created.")
        return

    features_df = pd.DataFrame(features_list)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)

    features_df.to_csv(output_path, header=False, index=False)
    print(f"\n✅ Features successfully saved to {output_path}")
    print(
        f"   File contains {len(features_list)} rows and {features_list[0].shape[0]} columns."
    )


def main():
    """Main function to run the feature extraction process for all specified models."""
    if not os.path.isdir(IMAGE_DATASET_PATH):
        print(f"\nError: Input folder not found at '{IMAGE_DATASET_PATH}'")
        return

    device = get_device()
    dataset_name = os.path.basename(os.path.normpath(IMAGE_DATASET_PATH))

    for model_name in MODEL_NAMES:
        print("\n" + "=" * 80)
        print(f"Processing model: {model_name.upper()}")
        print("=" * 80)

        output_filename = f"{dataset_name}_{model_name}.csv"
        full_output_path = os.path.join(HISTOGRAM_OUTPUT_PATH, output_filename)

        if os.path.exists(full_output_path):
            print(
                f"Output file already exists at {full_output_path}. Skipping this model."
            )
            continue

        try:
            model = load_model(model_name, device)
            features_data = process_folder_batched(
                model, IMAGE_DATASET_PATH, BATCH_SIZE, device
            )
            save_features_to_csv(features_data, full_output_path)

        except Exception as e:
            print(f"\nAN ERROR OCCURRED while processing {model_name.upper()}: {e}")
            print("Skipping to the next model.")

        finally:
            # Free up GPU memory before loading the next model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll specified models have been processed.")


if __name__ == "__main__":
    main()
