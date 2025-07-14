import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys


def main():
    """
    Main function to run the PCA process.
    """
    # --- Configuration ---
    # ⬇️  1. Set the path to your original CSV file here
    INPUT_CSV_PATH = "data\out\histograms\Olivia2688_combined_histograms_prototype.csv"

    # ⬇️  2. Set the desired path for the output file here
    OUTPUT_CSV_PATH = (
        "data\out\histograms\Olivia2688_combined_histograms_prototype_PCA.csv"
    )
    # -------------------

    try:
        # Load the dataset, assuming no header row
        print(f"🔄 Loading data from '{INPUT_CSV_PATH}'...")
        features = pd.read_csv(INPUT_CSV_PATH, header=None)
        original_feature_count = features.shape[1]
        print(f"✅ Data loaded successfully with {original_feature_count} features.")

        # Standardize the features before applying PCA
        print("⚙️  Standardizing features...")
        scaled_features = StandardScaler().fit_transform(features)

        # Apply PCA - retaining 95% of the variance by default
        print("🤖 Applying Principal Component Analysis (PCA)...")
        pca = PCA(n_components=0.95)
        principal_components = pca.fit_transform(scaled_features)
        new_feature_count = principal_components.shape[1]

        # Create a new DataFrame with the principal components
        pca_df = pd.DataFrame(principal_components)

        # Save the result to a new CSV file without header or index
        print(f"💾 Saving transformed data to '{OUTPUT_CSV_PATH}'...")
        pca_df.to_csv(OUTPUT_CSV_PATH, header=False, index=False)

        print("\n🎉 PCA applied successfully!")
        print(f"Original number of features: {original_feature_count}")
        print(f"Reduced number of features: {new_feature_count}")

    except FileNotFoundError:
        print(f"❌ Error: The file '{INPUT_CSV_PATH}' was not found.", file=sys.stderr)
        print(
            "Please make sure the INPUT_CSV_PATH variable is set correctly.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


# This calls the main function when the script is executed
if __name__ == "__main__":
    main()
