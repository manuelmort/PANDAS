import os
import shutil
import pandas as pd
import argparse

def create_subset(patches_dir, subset_csv, output_dir):
    """Find WSI files based on CSV (no copying)"""
    
    # Read CSV
    df = pd.read_csv(subset_csv)
    image_ids = df['image_id'] if 'image_id' in df.columns else df.iloc[:, 0]
    
    # Query each slide
    found = []
    missing = []
    
    for img_id in image_ids:
        # Try different extensions
        found_file = None
        for ext in ['.tiff', '.tif', '.svs', '.ndpi']:
            src = os.path.join(patches_dir, str(img_id) + ext)
            if os.path.exists(src):
                found_file = src
                break
        
        if found_file:
            found.append(img_id)
            print(f"Found: {img_id} -> {found_file}")
        else:
            missing.append(img_id)
            print(f"Missing: {img_id}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Found:   {len(found)}/{len(image_ids)}")
    print(f"  Missing: {len(missing)}/{len(image_ids)}")
    print(f"{'='*60}")
    
    if missing:
        print(f"\nMissing IDs:")
        for img_id in missing:
            print(f"  - {img_id}")

if __name__ == "__main__":
    # Hard-coded paths
    patches_dir = "/projectnb/ec500kb/projects/Project_1_Team_1/PANDA_DATA_MANNY/DATA/train_images"
    subset_csv = "/projectnb/ec500kb/projects/Project_1_Team_1/Official_GTP_PANDAS/PANDAS/data/patches/train_patch_10.csv"
    output_dir = "/projectnb/ec500kb/projects/Project_1_Team_1/PANDA_DATA_MANNY/DATA_SUBSETS/train_subset_01"
    
    create_subset(patches_dir, subset_csv, output_dir)
