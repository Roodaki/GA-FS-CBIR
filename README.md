# Genetic Feature Selection for Enhanced Image Retrieval

A state-of-the-art framework for optimizing feature selection in Content-Based Image Retrieval (CBIR) systems. By leveraging a Genetic Algorithm (GA) and a novel weighted fitness function, the framework effectively balances retrieval precision and computational efficiency through significant dimensionality reduction and enhanced retrieval performance.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Datasets](#datasets)
- [Experimental Results](#experimental-results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The GA-FS-CBIR framework tackles the challenges associated with high-dimensional feature spaces in CBIR systems. The method extracts comprehensive color and texture descriptors—including 1D and 2D color histograms (using RGB, HSV, and LAB color spaces) and Complete Local Binary Patterns (CLBP)—and applies a Genetic Algorithm to select an optimal subset of features. This approach reduces the feature dimensionality by over 91% while simultaneously improving retrieval precision.

![Proposed Image Retrieval Overview Shematic](https://github.com/user-attachments/assets/e0dc3eba-d3c7-47a1-aa7a-76ead5367f12)

---

## Key Features

- **Efficient Feature Extraction:** Combines 1D and 2D color histograms with CLBP for a robust representation of image content.
- **Genetic Algorithm Optimization:** Employs multi-objective optimization to balance retrieval accuracy with feature dimensionality.
- **Significant Dimensionality Reduction:** Reduces the number of features from 2,972 to 255 (a 91.42% reduction).
- **Enhanced Retrieval Performance:** Improves Precision@10 by 12.23%, achieving an overall precision of 93.07%.
- **Flexible and Scalable:** Adaptable to various feature types and applicable to large-scale image databases.

---

## Architecture

The system architecture consists of the following stages:

1. **Feature Extraction:**  
   - **Color Features:** Extraction using 1D and 2D histograms across RGB, HSV, and LAB color spaces.
   - **Texture Features:** Extraction using Complete Local Binary Patterns (CLBP) to capture fine-grained textures.

2. **Feature Selection:**  
   - Utilizes a Genetic Algorithm with a weighted fitness function that prioritizes high retrieval precision and minimal feature count.

3. **Image Retrieval:**  
   - Uses a K-Nearest Neighbors (KNN) algorithm with Canberra distance to retrieve and rank images based on similarity.

---

## Installation

### Prerequisites

- **Python 3.7+**
- **Required Libraries:**  
  - NumPy
  - Pandas
  - OpenCV  
  - Scikit-learn  
  - Matplotlib
  - DEAP

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Roodaki/GA-FS-CBIR.git
   cd GA-FS-CBIR
   ```
2. Install Dependencies:
   ```bash
    pip install -r requirements.txt
   ```
3. Download the Dataset:
   
    The framework is tested on the Corel-1K dataset. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/elkamel/corel-images) and place it in the designated directory.

---

## Datasets

- Corel-1K: A diverse dataset of 1,000 images categorized into 10 classes (e.g., Africans, Beaches, Buses). The dataset is available on [Kaggle](https://www.kaggle.com/datasets/elkamel/corel-images).

---

# Experimental Results

![ga_precision_plot](https://github.com/user-attachments/assets/a0226345-cc63-4b44-a0b3-10591ad4bd01)


The proposed GA-FS-CBIR framework has achieved the following results on the Corel-1K dataset:

- **Feature Dimensionality Reduction:** Reduced from 2,972 to 255 features (91.42% reduction).
- **Retrieval Precision:** Improved Precision@10 from 82.91% to 93.07% (a 12.23% increase).
- **Comparative Advantage:** Outperforms both traditional handcrafted methods and deep learning-based models in terms of accuracy and computational efficiency.

Detailed experimental results, including precision-recall curves and class-wise performance metrics, are provided in the research paper.

---

# Contributing

We welcome contributions to enhance the GA-FS-CBIR framework. Please follow these steps:

1. **Fork the Repository.**
2. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit Your Changes:
   ```bash
    git commit -am "Add new feature"
   ```
4. Push to the Branch: 
    ```bash
    git push origin feature/my-new-feature
    ```
5. Create a Pull Request.

For major changes, please open an issue first to discuss what you would like to change.

---

## License 

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or feedback, please contact: amirhossein.rdk@gmail.com
