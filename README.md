<div align="center">

# Genetic Algorithm-Based Feature Selection for Improved CBIR

[![Paper](https://img.shields.io/badge/IEEE_Access-10.1109/ACCESS.2025.3633906-blue.svg)](https://doi.org/10.1109/ACCESS.2025.3633906)
[![Python](https://img.shields.io/badge/Python-3.7%2B-yellow.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A state-of-the-art framework for optimizing feature selection in Content-Based Image Retrieval (CBIR) systems, achieving superior precision with over 90% reduction in feature dimensionality.**

[Overview](#overview) • [Key Features](#key-features) • [Architecture](#architecture) • [Datasets](#datasets) • [Performance](#performance) • [Installation](#installation) • [Citation](#citation)

</div>

---

## Overview

This repository contains the implementation of the **GA-FS-CBIR** framework as presented in our paper published in **IEEE Access**. 

The exponential growth of digital image repositories demands fast and accurate retrieval systems. However, high-dimensional feature spaces often lead to the "curse of dimensionality," causing computational inefficiencies and lower accuracy. This framework addresses these challenges by combining a novel **Genetic Algorithm (GA)** strategy with high-dimensional handcrafted descriptors.

By extracting comprehensive color and texture descriptors and applying a multi-objective GA, this method reduces feature dimensionality by **91.42%** on average while simultaneously improving retrieval precision across diverse datasets.

---

## Key Features

### 1. Novel Feature Extraction
We introduce a robust feature set combining Color and Texture:
* **1D Color Histograms:** Extracted from **RGB**, **HSV**, and **LAB** color spaces.
* **Novel 2D Color Histograms:**
    * **Intra-Color:** Captures relationships between channels within the same space (e.g., R vs. G).
    * **Inter-Color:** Captures relationships between channels across different spaces (e.g., RGB-Red vs. HSV-Saturation).
* **Texture:** Uses **Complete Local Binary Patterns (CLBP)** to capture sign, magnitude, and center pixel patterns.

### 2. Evolutionary Feature Selection
A specialized Genetic Algorithm optimizes the feature subset using a weighted fitness function that balances two conflicting objectives:
1.  **Maximize Retrieval Accuracy** (Precision@K).
2.  **Minimize Feature Dimensionality** ($F_{selected}$).

$$Fitness = \alpha \times Precision + (\alpha-1) \times \frac{F_{selected}}{F_{total}}$$

### 3. Scalability & Efficiency
* **Drastic Reduction:** Reduces feature counts from ~3,000 to ~250 without information loss.
* **Speed:** validated on large-scale datasets (Corel-10K, GHIM-10K), reducing query time by up to **38%**.
* **Versatility:** The selection method is effective on both **handcrafted features** and **deep learning embeddings** (e.g., ConvNeXt, VGG19).

---

## Architecture

The system follows a three-stage pipeline:

1.  **Feature Extraction:** Generates high-dimensional feature vectors ($F_{total}$).
2.  **Evolutionary Optimization:** The GA evolves a population of feature masks using Tournament Selection, Uniform Crossover, and Bit Flip Mutation.
3.  **Retrieval:** Uses **K-Nearest Neighbors (KNN)** with **Canberra Distance** for similarity ranking.

![System Architecture](https://github.com/user-attachments/assets/e0dc3eba-d3c7-47a1-aa7a-76ead5367f12)
*Figure 1: The proposed GA-FS-CBIR architecture flow.*

---

## Datasets

The framework is evaluated on five diverse datasets, covering object-centric, heritage, and natural scene images.

| Dataset | Images | Classes | Description | Download |
| :--- | :--- | :--- | :--- | :--- |
| **Corel-1K** | 1,000 | 10 | Classical benchmark (Horses, Beaches, etc.) | [Kaggle Link](https://www.kaggle.com/datasets/amirhosseinroodaki/corel-1k-corel-5k-and-corel-10k-datasets) |
| **Corel-10K** | 10,000 | 100 | Large-scale version with diverse categories | [Kaggle Link](https://www.kaggle.com/datasets/amirhosseinroodaki/corel-1k-corel-5k-and-corel-10k-datasets) |
| **GHIM-10K** | 10,000 | 20 | Geological and Heritage images | [Kaggle Link](https://www.kaggle.com/datasets/guohey/ghim10k) |
| **Produce-1400** | 1,400 | 14 | Fruits and vegetables (Object recognition) | [Kaggle Link](https://www.kaggle.com/datasets/amirhosseinroodaki/produce-1400) |
| **Olivia-2688** | 2,688 | 8 | Complex natural scenes (Nature, Street, etc.) | [Kaggle Link](https://www.kaggle.com/datasets/amirhosseinroodaki/olivia-2688) |

---

## Performance

### 1. Optimization Results
The proposed method achieves significant improvements in **Precision@10** while drastically reducing the number of features (Table IV of the paper).

| Dataset | Stage | Features | Precision@10 (%) | Recall@10 (%) | Query Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Corel-1K** | Initial | 2,972 | 82.91 | 8.29 | 29.24 |
| | **Optimized** | **255** | **93.07** | **9.31** | **24.59** |
| **Corel-10K** | Initial | 1,364 | 50.94 | 5.09 | 218.78 |
| | **Optimized** | **241** | **72.19** | **7.22** | **198.73** |
| **GHIM-10K** | Initial | 2,972 | 68.36 | 1.37 | 375.06 |
| | **Optimized** | **218** | **89.13** | **1.83** | **230.94** |
| **Produce-1400** | Initial | 5,148 | 98.55 | 9.85 | 49.50 |
| | **Optimized** | **50** | **99.36** | **9.94** | **39.65** |

### 2. Convergence Analysis
The Genetic Algorithm consistently improves the population's precision over generations.

![Precision Plot](https://github.com/user-attachments/assets/a0226345-cc63-4b44-a0b3-10591ad4bd01)
*Figure 2: Evolution of Precision@10 over generations on the Corel-1K dataset.*

### 3. Comparison with Deep Learning & PCA
* **Vs. PCA:** Unlike PCA, which degraded precision (e.g., -36% on Corel-1K) to achieve reduction, our method **improved** precision while achieving higher reduction rates.
* **Vs. Deep Learning:** Our method outperformed specific Deep Learning benchmarks in efficiency. For example, on **GHIM-10K**, it surpassed SqueezeNet-based retrieval (89.13% vs 85.3%) while using significantly fewer features.

---

## Installation

### Prerequisites
* Python 3.7+
* Libraries: `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `matplotlib`, `deap`

### Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Roodaki/GA-FS-CBIR.git](https://github.com/Roodaki/GA-FS-CBIR.git)
    cd GA-FS-CBIR
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup:**
    Download the datasets from the links provided in the [Datasets](#datasets) section.

---

## Citation

If you use this code or the datasets in your research, please cite [our IEEE Access paper](https://doi.org/10.1109/ACCESS.2025.3633906):

```bibtex
@article{roodaki2025genetic,
author={Roodaki, AmirHossein and Sotoodeh, Mahmood and Moosavi, Mohammad Reza},
  journal={IEEE Access}, 
  title={Genetic Algorithm-Based Feature Selection from High-Dimensional Descriptors for Improved Content-Based Image Retrieval},
  year={2025},
  doi={10.1109/ACCESS.2025.3633906}
}}
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

AmirHossein Roodaki - amirhossein.rdk@gmail.com | amirhossein.roodaki@hafez.shirazu.ac.ir

Dr. Mahmood Sotoodeh (Corresponding Author) - m-sotoodeh@fasau.ac.ir

