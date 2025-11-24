import pandas as pd
import numpy as np
import os

# Define paths

SPLITS_DIR = '../data/splits'
TRAIN_SPLIT_PATH = os.path.join(SPLITS_DIR, 'train_split.csv')
PATCHES_DIR = '../data/patches'

# Read the training split
print("Loading training data...")
train_data = pd.read_csv(TRAIN_SPLIT_PATH)
print(f"Total training entries: {len(train_data):,}")

# Check file size
train_split_size_gb = os.path.getsize(TRAIN_SPLIT_PATH) / (1024**3)
train_split_size_mb = os.path.getsize(TRAIN_SPLIT_PATH) / (1024**2)
print(f"Train split size: {train_split_size_gb:.2f} GB ({train_split_size_mb:.1f} MB)")

# Set number of patches
NUM_PATCHES = 10
entries_per_patch = int(np.ceil(len(train_data) / NUM_PATCHES))

print(f"\nSplitting into {NUM_PATCHES} patches:")
print(f"  Entries per patch: ~{entries_per_patch:,}")
print(f"  Size per patch: ~{train_split_size_mb/NUM_PATCHES:.1f} MB (~{train_split_size_gb/NUM_PATCHES:.3f} GB)")
print(f"  Total storage needed: ~{train_split_size_gb:.2f} GB")

# Ask for confirmation
print(f"\nProceed with creating {NUM_PATCHES} patches? (y/n): ", end="")
response = input().strip().lower()

if response == 'y':
    # Create patches directory
    os.makedirs(PATCHES_DIR, exist_ok=True)
    
    print(f"\nCreating {NUM_PATCHES} patches in {PATCHES_DIR}...")
    
    for i in range(NUM_PATCHES):
        start_idx = i * entries_per_patch
        end_idx = min((i + 1) * entries_per_patch, len(train_data))
        
        patch = train_data.iloc[start_idx:end_idx]
        patch_file = os.path.join(PATCHES_DIR, f'train_patch_{i+1:02d}.csv')
        
        print(f"  Creating patch {i+1}/{NUM_PATCHES}...", end=" ")
        patch.to_csv(patch_file, index=False)
        
        patch_size_mb = os.path.getsize(patch_file) / (1024**2)
        patch_size_gb = os.path.getsize(patch_file) / (1024**3)
        print(f"✓ {len(patch):,} entries ({patch_size_mb:.1f} MB)")
    
    print(f"\n{'='*60}")
    print(f"✓ All {NUM_PATCHES} patches created successfully!")
    print(f"{'='*60}")
    print(f"Location: {PATCHES_DIR}/")
    print(f"Files: train_patch_01.csv through train_patch_{NUM_PATCHES:02d}.csv")
    
    # Calculate total patch size
    total_patch_size = sum(
        os.path.getsize(os.path.join(PATCHES_DIR, f'train_patch_{i+1:02d}.csv'))
        for i in range(NUM_PATCHES)
    ) / (1024**3)
    print(f"Total patch storage: {total_patch_size:.2f} GB")
    
else:
    print("Patching cancelled.")
