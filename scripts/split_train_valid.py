import pandas as pd
from sklearn.model_selection import train_test_split

# Read the training data
train_df = pd.read_csv('../data/train.csv')

print(f"Total entries in train.csv: {len(train_df)}")

# Split into 80% training, 20% validation
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42,  # For reproducibility
    shuffle=True      # Shuffle before splitting
)

print(f"\nAfter 80/20 split:")
print(f"  Training set:   {len(train_data):,} entries (80%)")
print(f"  Validation set: {len(val_data):,} entries (20%)")

# Save the splits to separate files
train_data.to_csv('../data/splits/train_split.csv', index=False)
val_data.to_csv('../data/splits/val_split.csv', index=False)

print(f"\n✓ Saved train_split.csv")
print(f"✓ Saved val_split.csv")

# Optional: Verify the splits
print(f"\nVerification:")
print(f"  train_split.csv: {len(pd.read_csv('../data/splits/train_split.csv'))} entries")
print(f"  val_split.csv: {len(pd.read_csv('../data/splits/val_split.csv'))} entries")
print(f"  Total: {len(pd.read_csv('../data/splits/train_split.csv')) + len(pd.read_csv('../data/splits/val_split.csv'))} entries")
