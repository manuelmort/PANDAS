#!/usr/bin/env python3
"""
GAT Training Script for PANDA Dataset
Handles variable-sized graphs (different number of nodes per WSI)
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.GAT import GATClassifier
from utils.dataset import GraphDataset
from helper import collate

def parse_args():
    parser = argparse.ArgumentParser(description='GAT Training for PANDA')
    
    # Model parameters
    parser.add_argument('--n_class', type=int, default=3, help='Number of classes')
    parser.add_argument('--n_features', type=int, default=768, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to graph data')
    parser.add_argument('--train_set', type=str, required=True, help='Path to train set file')
    parser.add_argument('--val_set', type=str, required=True, help='Path to validation set file')
    parser.add_argument('--site', type=str, default='panda', help='Dataset site')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (accumulation steps)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Output parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to save models')
    parser.add_argument('--log_path', type=str, required=True, help='Path for tensorboard logs')
    parser.add_argument('--task_name', type=str, required=True, help='Task name for saving')
    
    # Flags
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights')
    
    return parser.parse_args()

def load_data(args):
    """Load training and validation datasets"""
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # Load train IDs
    with open(args.train_set, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    
    # Load val IDs
    with open(args.val_set, 'r') as f:
        val_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples: {len(val_ids)}")
    
    # Create datasets
    train_dataset = GraphDataset(args.data_path, train_ids, site=args.site)
    val_dataset = GraphDataset(args.data_path, val_ids, site=args.site)
    
    # Use batch_size=1 since graphs have different sizes
    # We'll accumulate gradients to simulate larger batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Process one graph at a time
        collate_fn=collate, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=collate, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"Train samples: {len(train_loader)}")
    print(f"Val samples: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset

def compute_class_weights(dataset, n_class, device):
    """Compute class weights for imbalanced data"""
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is not None:
            labels.append(sample['label'])
    
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=n_class)
    total = len(labels)
    
    weights = total / (n_class * class_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights.cpu().numpy()}")
    
    return weights

def train_one_epoch(model, train_loader, optimizer, device, epoch, accum_steps=8):
    """Train for one epoch with gradient accumulation"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    accum_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, sample in enumerate(pbar):
        if sample is None:
            continue
        
        try:
            # Get single graph data
            if isinstance(sample["image"], list):
                img = sample["image"][0].unsqueeze(0).float().to(device)
                adj = sample["adj_s"][0].unsqueeze(0).float().to(device)
                label = torch.tensor([sample["label"][0]]).to(device)
            else:
                img = sample["image"].unsqueeze(0).float().to(device)
                adj = sample["adj_s"].unsqueeze(0).float().to(device)
                label = torch.tensor([sample["label"]]).to(device)
            
            mask = torch.ones(1, img.size(1)).to(device)
            
            # Forward pass
            pred, _, loss = model(img, label, adj, mask)
            
            # Scale loss for gradient accumulation
            loss = loss / accum_steps
            loss.backward()
            
            accum_loss += loss.item()
            
            # Track metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            # Update weights every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += accum_loss * accum_steps
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{accum_loss * accum_steps:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
                
                accum_loss = 0
                
        except Exception as e:
            # Print error once every 100 batches to avoid spam
            if batch_idx % 100 == 0:
                print(f"\nError in batch {batch_idx}: {e}")
            continue
    
    # Final optimizer step for remaining gradients
    if accum_loss > 0:
        optimizer.step()
        optimizer.zero_grad()
        total_loss += accum_loss * accum_steps
    
    # Compute epoch metrics
    if len(all_labels) == 0:
        return 0, 0, 0
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, qwk

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for batch_idx, sample in enumerate(pbar):
            if sample is None:
                continue
            
            try:
                # Get single graph data
                if isinstance(sample["image"], list):
                    img = sample["image"][0].unsqueeze(0).float().to(device)
                    adj = sample["adj_s"][0].unsqueeze(0).float().to(device)
                    label = torch.tensor([sample["label"][0]]).to(device)
                else:
                    img = sample["image"].unsqueeze(0).float().to(device)
                    adj = sample["adj_s"].unsqueeze(0).float().to(device)
                    label = torch.tensor([sample["label"]]).to(device)
                
                mask = torch.ones(1, img.size(1)).to(device)
                
                pred, _, loss = model(img, label, adj, mask)
                
                total_loss += loss.item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                
            except Exception as e:
                if batch_idx % 100 == 0:
                    print(f"\nError in val batch {batch_idx}: {e}")
                continue
    
    # Compute metrics
    if len(all_labels) == 0:
        return 0, 0, 0, 0
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, qwk, f1

def main():
    args = parse_args()
    
    print("=" * 60)
    print("GAT Training for PANDA")
    print("=" * 60)
    print(f"Task: {args.task_name}")
    print(f"Features: {args.n_features}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Attention heads: {args.heads}")
    print(f"Dropout: {args.dropout}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size (grad accum): {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_loader, val_loader, train_dataset = load_data(args)
    
    # Compute class weights if needed
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, args.n_class, device)
    
    # Create model
    model = GATClassifier(
        n_class=args.n_class,
        n_features=args.n_features,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        class_weights=class_weights
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Tensorboard
    os.makedirs(args.log_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_path, args.task_name))
    
    # Create model save directory
    os.makedirs(args.model_path, exist_ok=True)
    
    # Training loop
    best_qwk = 0
    best_epoch = 0
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        
        # Train with gradient accumulation
        train_loss, train_acc, train_qwk = train_one_epoch(
            model, train_loader, optimizer, device, epoch, accum_steps=args.batch_size
        )
        
        # Validate
        val_loss, val_acc, val_qwk, val_f1 = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_qwk)
        
        epoch_time = time.time() - start_time
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('QWK/train', train_qwk, epoch)
        writer.add_scalar('QWK/val', val_qwk, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, QWK: {train_qwk:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, QWK: {val_qwk:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            
            save_path = os.path.join(args.model_path, f"{args.task_name}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  *** New best model saved! QWK: {val_qwk:.4f} ***")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.model_path, f"{args.task_name}_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_qwk': val_qwk,
                'best_qwk': best_qwk,
            }, checkpoint_path)
    
    writer.close()
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best QWK: {best_qwk:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {os.path.join(args.model_path, f'{args.task_name}.pth')}")
    print("=" * 60)

if __name__ == "__main__":
    main()