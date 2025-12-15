#!/usr/bin/env python3
"""
GTP/GAT Inference Script
Accepts user-provided test data (folder or file list)

Usage:
    # From a folder of graphs
    python inference.py --input /data/graphs --output predictions.csv --model gat
    
    # From a file list
    python inference.py --input /data/graphs --test_file /data/test_ids.txt --output predictions.csv
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import GraphDataset
from helper import collate

# ============================================================
# CONFIGURATION
# ============================================================
BACKBONE_CONFIGS = {
    'phikon': {'n_features': 768, 'name': 'Phikon'},
    'imagenet': {'n_features': 2048, 'name': 'ImageNet ResNet50'},
    'simclr': {'n_features': 512, 'name': 'SimCLR'},
}

CLASS_NAMES = ['Background', 'Benign', 'Cancerous']


def parse_args():
    parser = argparse.ArgumentParser(
        description='GTP/GAT Inference - Prostate Cancer Grading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on a folder of graphs
  python inference.py --input /data/graphs --output predictions.csv --model gat

  # Run inference with a test file (list of slide IDs)
  python inference.py --input /data/graphs --test_file test_ids.txt --output predictions.csv

  # Use Graph Transformer instead of GAT
  python inference.py --input /data/graphs --output predictions.csv --model transformer

  # Use different backbone
  python inference.py --input /data/graphs --output predictions.csv --backbone imagenet
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input graph directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output predictions CSV')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Optional: text file with slide IDs (one per line)')
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'transformer'],
                        help='Model type: gat or transformer (default: gat)')
    parser.add_argument('--backbone', type=str, default='phikon',
                        choices=['phikon', 'imagenet', 'simclr'],
                        help='Feature extraction backbone (default: phikon)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu (default: auto)')
    
    return parser.parse_args()


def get_slide_ids(input_path, test_file=None):
    """
    Get slide IDs from either:
    1. A test file (text file with one slide ID per line)
    2. Subdirectories in the input folder
    
    Returns list in format required by GraphDataset: "panda/slide_id\t0"
    (label is dummy since we're doing inference)
    """
    if test_file and os.path.exists(test_file):
        print(f"Loading slide IDs from: {test_file}")
        with open(test_file, 'r') as f:
            slide_ids = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # If already in correct format (panda/slide_id\tlabel), keep it
                if '\t' in line and '/' in line:
                    slide_ids.append(line)
                # If just has tab (slide_id\tlabel), add panda/ prefix
                elif '\t' in line:
                    parts = line.split('\t')
                    slide_ids.append(f"panda/{parts[0]}\t{parts[1]}")
                # If has path but no label (panda/slide_id), add dummy label
                elif '/' in line:
                    slide_ids.append(f"{line}\t0")
                # If just slide_id, add both prefix and dummy label
                else:
                    slide_ids.append(f"panda/{line}\t0")
        return slide_ids
    else:
        print(f"Scanning directory: {input_path}")
        # Get all subdirectories as slide IDs with dummy labels
        slide_ids = [f"panda/{d}\t0" for d in os.listdir(input_path) 
                     if os.path.isdir(os.path.join(input_path, d))]
        return sorted(slide_ids)


def load_model(model_type, backbone, weights_path, device):
    """Load the appropriate model"""
    n_features = BACKBONE_CONFIGS[backbone]['n_features']
    
    if model_type == 'gat':
        from models.GAT import GATClassifier
        model = GATClassifier(
            n_class=3,
            n_features=n_features,
            hidden_dim=64,
            heads=4
        )
        default_weights = f'./weights/gat_{backbone}_model.pth'
    else:
        from models.GraphTransformer import Classifier
        model = Classifier(n_class=3, n_features=n_features)
        default_weights = f'./weights/{backbone}_transformer_model.pth'
    
    # Use provided weights or default
    weights_path = weights_path or default_weights
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}\n"
                                f"Please mount weights to /app/weights/")
    
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    
    # Handle checkpoint format
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device).eval()
    return model


def run_inference(model, model_type, dataloader, device, slide_ids_list):
    """Run inference and return predictions"""
    all_preds = []
    all_probs = []
    processed_ids = []
    
    # For transformer, setup hook to capture logits
    logits_list = []
    if model_type == 'transformer':
        def hook_fn(module, input, output):
            if hasattr(output, 'shape') and len(output.shape) == 2 and output.shape[1] == 3:
                logits_list.append(output.detach().cpu())
        
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(hook_fn)
    
    print("\nRunning inference...")
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataloader, desc="Processing")):
            if sample is None:
                continue
            
            try:
                # Get data
                if isinstance(sample["image"], list):
                    img = sample["image"][0].unsqueeze(0).float().to(device)
                    adj = sample["adj_s"][0].unsqueeze(0).float().to(device)
                else:
                    img = sample["image"].unsqueeze(0).float().to(device)
                    adj = sample["adj_s"].unsqueeze(0).float().to(device)
                
                mask = torch.ones(1, img.size(1)).to(device)
                dummy_label = torch.tensor([0], dtype=torch.long).to(device)
                
                if model_type == 'gat':
                    pred, probs, _ = model(img, dummy_label, adj, mask)
                    all_preds.append(pred.cpu().item())
                    all_probs.append(probs.cpu().numpy())
                else:
                    logits_list.clear()
                    model(img, dummy_label, adj, mask)
                    if logits_list:
                        logits = logits_list[-1]
                        probs = torch.softmax(logits, dim=1).numpy()[0]
                        pred = logits.argmax(1).item()
                        all_preds.append(pred)
                        all_probs.append(probs)
                
                # Extract just slide ID for output (remove panda/ prefix and label)
                raw_id = slide_ids_list[i] if i < len(slide_ids_list) else f"slide_{i}"
                if '\t' in raw_id:
                    raw_id = raw_id.split('\t')[0]
                if '/' in raw_id:
                    raw_id = raw_id.split('/')[-1]
                processed_ids.append(raw_id)
                
            except Exception as e:
                print(f"\nWarning: Error processing sample {i}: {e}")
                continue
    
    return processed_ids, all_preds, np.array(all_probs)


def save_predictions(output_path, slide_ids, preds, probs):
    """Save predictions to CSV"""
    df = pd.DataFrame({
        'slide_id': slide_ids,
        'prediction': preds,
        'predicted_class': [CLASS_NAMES[p] for p in preds],
        'prob_background': probs[:, 0],
        'prob_benign': probs[:, 1],
        'prob_cancerous': probs[:, 2],
    })
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def main():
    args = parse_args()
    
    print("=" * 60)
    print("GTP/GAT - Prostate Cancer Grading Inference")
    print("=" * 60)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nConfiguration:")
    print(f"  Model:    {args.model.upper()}")
    print(f"  Backbone: {BACKBONE_CONFIGS[args.backbone]['name']}")
    print(f"  Device:   {device}")
    print(f"  Input:    {args.input}")
    print(f"  Output:   {args.output}")
    
    # Get slide IDs
    slide_ids = get_slide_ids(args.input, args.test_file)
    print(f"  Slides:   {len(slide_ids)}")
    
    if len(slide_ids) == 0:
        print("\nError: No slides found!")
        print("Make sure your input directory contains graph subdirectories.")
        sys.exit(1)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, args.backbone, args.weights, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    dataset = GraphDataset(args.input, slide_ids, site='panda')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate, 
                           num_workers=0, shuffle=False)
    
    # Run inference
    processed_ids, preds, probs = run_inference(
        model, args.model, dataloader, device, slide_ids
    )
    
    print(f"\nProcessed {len(preds)} slides successfully")
    
    # Save predictions
    df = save_predictions(args.output, processed_ids, preds, probs)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nPrediction distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = (df['prediction'] == i).sum()
        pct = 100 * count / len(df)
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    print(f"\nPredictions saved to: {args.output}")
    print(f"\nOutput format:")
    print(f"  - slide_id: Slide identifier")
    print(f"  - prediction: Predicted class (0, 1, 2)")
    print(f"  - predicted_class: Class name")
    print(f"  - prob_*: Class probabilities")
    
    print(f"\n{'=' * 60}")
    print("Inference complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()