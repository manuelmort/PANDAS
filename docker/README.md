# GTP/GAT - Graph-based Prostate Cancer Grading

Docker container for prostate cancer grading using Graph Transformer (GTP) and Graph Attention Networks (GAT) on the PANDA Challenge dataset.

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t gtp-panda:latest .
```

### 2. Run Inference

```bash
# Using GAT (recommended - best performance)
docker run --gpus all \
    -v $(pwd)/data/graphs:/data/graphs \
    -v $(pwd)/data/output:/data/output \
    -v $(pwd)/weights:/app/weights \
    gtp-panda:latest \
    python inference.py --input /data/graphs --output /data/output/predictions.csv --model gat

# Using Graph Transformer
docker run --gpus all \
    -v $(pwd)/data/graphs:/data/graphs \
    -v $(pwd)/data/output:/data/output \
    -v $(pwd)/weights:/app/weights \
    gtp-panda:latest \
    python inference.py --input /data/graphs --output /data/output/predictions.csv --model transformer
```

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── inference.py              # Main inference script
├── main.py                   # Graph Transformer training
├── main_gat.py              # GAT training
├── evaluate.py              # Transformer evaluation
├── evaluate_gat.py          # GAT evaluation
├── helper.py                # Utility functions
├── models/
│   ├── GAT.py               # Graph Attention Network
│   ├── GraphTransformer.py  # Graph Transformer
│   └── ...
├── utils/
│   └── dataset.py           # Data loading
├── scripts/
│   ├── train_set.txt        # Training split
│   ├── val_set.txt          # Validation split
│   └── test_set.txt         # Test split
├── weights/                  # Pre-trained models
│   ├── gat_phikon_model.pth
│   ├── phikon_transformer_model.pth
│   ├── imagenet_transformer_model.pth
│   └── simclr_transformer_model.pth
└── data/
    ├── graphs/              # Input graph data
    └── output/              # Predictions
```

## Docker Compose

### Run Inference
```bash
docker-compose up gtp-inference
```

### Train GAT
```bash
docker-compose up gat-train
```

### Train Graph Transformer
```bash
docker-compose up gtp-train
```

### Evaluate
```bash
docker-compose up evaluate
```

## Available Models

| Model | Backbone | QWK | Parameters | Weights File |
|-------|----------|-----|------------|--------------|
| GAT | Phikon | **0.8744** | 280K | `gat_phikon_model.pth` |
| Transformer | Phikon | 0.8576 | 1.2M | `phikon_transformer_model.pth` |
| Transformer | ImageNet | 0.7700 | 1.2M | `imagenet_transformer_model.pth` |
| Transformer | SimCLR | 0.5625 | 1.2M | `simclr_transformer_model.pth` |

## Inference Options

```bash
python inference.py --help

Options:
  --input         Path to input graph directory (required)
  --output        Path to output CSV file (required)
  --model         Model type: gat or transformer (default: gat)
  --backbone      Feature backbone: phikon, imagenet, simclr (default: phikon)
  --weights       Path to model weights (auto-detected if not specified)
  --eval          Enable evaluation mode (requires labels)
  --test_file     Path to test set file with labels
  --device        Device: auto, cuda, cpu (default: auto)
```

### Examples

```bash
# Basic inference with GAT
python inference.py --input /data/graphs --output predictions.csv --model gat

# Inference with evaluation
python inference.py --input /data/graphs --output results.csv --model gat --eval --test_file test_set.txt

# Use specific backbone
python inference.py --input /data/graphs --output predictions.csv --model transformer --backbone imagenet

# CPU-only mode
python inference.py --input /data/graphs --output predictions.csv --model gat --device cpu
```

## Output Format

The output CSV contains:

| Column | Description |
|--------|-------------|
| `slide_id` | Slide identifier |
| `prediction` | Predicted class (0, 1, 2) |
| `pred_class` | Class name (Background, Benign, Cancerous) |
| `prob_background` | Probability of class 0 |
| `prob_benign` | Probability of class 1 |
| `prob_cancerous` | Probability of class 2 |
| `true_label` | Ground truth (if provided) |
| `correct` | Whether prediction was correct |

## Class Definitions

| Class | ISUP Grade | Description |
|-------|------------|-------------|
| 0 - Background | ISUP 0 | No cancer detected |
| 1 - Benign | ISUP 1 | Low-grade (Gleason 3+3) |
| 2 - Cancerous | ISUP 2-5 | Intermediate to high-grade |

## GPU Requirements

- **GAT**: 8-16 GB GPU memory (recommended: 16GB)
- **Transformer**: 16-40 GB GPU memory (recommended: 40GB)

### Check GPU Availability
```bash
docker run --gpus all gtp-panda:latest nvidia-smi
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU mode
docker run -v ... gtp-panda:latest python inference.py --input /data/graphs --output pred.csv --device cpu
```

### torch_geometric Import Error
Ensure PyTorch Geometric is properly installed:
```bash
docker run gtp-panda:latest pip list | grep torch
```

### Missing Model Weights
Download pre-trained weights and place in `./weights/` directory.

## Citation

```bibtex
@article{gtp2024panda,
  title={Graph-Transformer for Prostate Cancer Grade Assessment},
  author={Morteo, Manuel},
  institution={Boston University},
  year={2025}
}
```

## License

MIT License

## Authors

- Manuel Morteo (mmorteo@bu.edu)

Boston University - Medical Imaging with AI, Fall 2025