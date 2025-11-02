# GTP-PANDA

Graph Transformer for Prostate cANcer Detection and grAding (GTP-PANDA) - A deep learning framework for analyzing prostate histopathology images using graph-based representations and transformer architectures.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview

GTP-PANDA implements a graph-based deep learning approach for prostate cancer detection and Gleason grading from whole slide images (WSI). The framework extracts tissue graphs from histopathology slides and processes them using graph transformer architectures.

## Project Structure

```
gtp-panda/
├── feature_extractor/          # Graph construction and feature extraction
│   ├── run.py                  # Main script for feature extraction pipeline
│   ├── build_graphs.py         # Graph construction from WSI patches
│   └── config_panda.yaml       # Configuration for feature extraction
├── models/                     # Neural network architectures
│   ├── GraphTransformer.py     # Graph Transformer implementation
│   └── ...                     # Additional model files
├── data/                       # Data loading and preprocessing
│   ├── panda_dataset.py        # PyTorch Dataset for PANDA challenge
│   ├── __init__.py             
│   └── panda_labels.csv        # Ground truth labels
├── utils/                      # Utility functions
│   └── metrics.py              # Evaluation metrics (accuracy, kappa, etc.)
├── train_panda.py              # Training script
├── README.md                   # Project documentation
└── environment.yml             # Conda environment specification (optional)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda or pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gtp-panda.git
cd gtp-panda
```

2. Create and activate the environment:

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate gtp-panda
```

**Using pip:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Feature Extraction and Graph Construction

Extract features and build tissue graphs from whole slide images:

```bash
cd feature_extractor
python run.py --config config_panda.yaml
```

Edit `config_panda.yaml` to customize:
- Input WSI directories
- Patch size and magnification
- Graph construction parameters (k-NN, radius, etc.)
- Feature extractor backbone

### 2. Training

Train the Graph Transformer model:

```bash
python train_panda.py \
    --data_dir ./data \
    --graph_dir ./graphs \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0001
```

### 3. Evaluation

Evaluate model performance on test set:

```bash
python train_panda.py \
    --mode eval \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data
```

## Configuration

### Feature Extraction (config_panda.yaml)

Key parameters:
- `wsi_dir`: Path to whole slide images
- `output_dir`: Directory for saving graphs
- `patch_size`: Size of extracted patches (e.g., 256)
- `magnification`: Magnification level (e.g., 20x)
- `feature_extractor`: Backbone architecture (ResNet, ViT, etc.)
- `graph_type`: Graph construction method (knn, radius, etc.)

### Training Parameters

Modify in `train_panda.py` or pass as arguments:
- Learning rate
- Batch size
- Number of epochs
- Model architecture hyperparameters
- Data augmentation settings

## Dataset

This project uses the PANDA (Prostate cANcer graDe Assessment) challenge dataset. 

**Data Structure:**
- `data/panda_labels.csv`: Contains slide IDs, ISUP grades, and Gleason scores
- Expected columns: `image_id`, `isup_grade`, `gleason_score`

Download the PANDA dataset from [Kaggle](https://www.kaggle.com/c/prostate-cancer-grade-assessment).

## Metrics

Evaluation metrics implemented in `utils/metrics.py`:
- Quadratic Weighted Kappa (primary metric)
- Accuracy
- Confusion Matrix
- Per-class precision/recall

## Model Architecture

The Graph Transformer model (`models/GraphTransformer.py`) implements:
- Multi-head graph attention layers
- Positional encodings for spatial information
- Global pooling for slide-level predictions
- Classification head for ISUP grade prediction (0-5)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

## Acknowledgments

- PANDA Challenge organizers
- Graph neural network and transformer architecture references
- Open-source libraries used in this project

---

**Note:** Update paths, URLs, and contact information according to your specific implementation.