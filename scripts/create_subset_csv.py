"""
create_subset_csv.py
---------------------------------
Creates a smaller train_subset.csv file from the full PANDA train.csv.
Useful for downloading or testing on a small portion of the dataset.

Usage:
    python scripts/create_subset_csv.py --input data/train.csv \
        --output data/train_subset.csv --samples 10     
"""

import pandas as pd
from pathlib import Path

df = pd.read_csv("PANDAS/data/train.csv")
# Load master log (if exists)
master_path = Path("data/subsets/master_download_log.csv")
if master_path.exists():
    downloaded = pd.read_csv(master_path)
    used_ids = set(downloaded["image_id"])
else:
    used_ids = set()

# Filter out previously used IDs
available = df[~df["image_id"].isin(used_ids)]

# Sample new ones
subset = pd.concat([
    available[available["isup_grade"] == g].sample(10, random_state=42)
    for g in range(6)
    if not available[available["isup_grade"] == g].empty
])

# Save new subset
version = len(list(Path("PANDAS/data/subsets").glob("train_subset_v*.csv"))) + 1
subset_path = Path(f"PANDAS/data/subsets/train_subset_v{version}.csv")
subset_path.parent.mkdir(parents=True, exist_ok=True)
subset.to_csv(subset_path, index=False)

# Append to master log
if master_path.exists():
    master = pd.read_csv(master_path)
    master = pd.concat([master, subset])
else:
    master = subset
master.to_csv(master_path, index=False)

print(f"Created {subset_path} and updated master log ({len(master)} total samples).")


