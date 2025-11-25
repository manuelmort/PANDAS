#!/usr/bin/env python3
import os
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------
# Background filter
# ---------------------------------------------------------
def is_not_background(tile: Image.Image, bg_thresh=220, max_bg_pct=0.50) -> bool:
    arr = np.array(tile)
    if arr.ndim != 3:
        return False
    gray = arr.mean(axis=2)
    pct_background = np.mean(gray > bg_thresh)
    return pct_background <= max_bg_pct


# ---------------------------------------------------------
# DeepZoom coordinates generator
# ---------------------------------------------------------
def dz_coords(level_w: int, level_h: int, s: int, e: int):
    """
    Produce DeepZoom tile windows.
    Final tile size = s + 2e.
    """
    stride = s
    tile_full = s + 2*e

    cols = max(1, (level_w  + stride - 1) // stride)
    rows = max(1, (level_h + stride - 1) // stride)

    for row in range(rows):
        for col in range(cols):
            # interior position
            x_in = col * stride
            y_in = row * stride

            # clamp interior
            x_in = min(x_in, level_w  - s)
            y_in = min(y_in, level_h - s)

            # apply overlap
            x0 = max(0, x_in - e)
            y0 = max(0, y_in - e)
            x1 = min(level_w,  x_in + s + e)
            y1 = min(level_h, y_in + s + e)

            yield col, row, x0, y0, x1, y1


# ---------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------
def tile_worker(args):
    (slide_path, out_dir, level, s, e, bg_thresh, max_bg_pct) = args

    slide = openslide.OpenSlide(slide_path)
    level_w, level_h = slide.level_dimensions[level]
    ds = slide.level_downsamples[level]

    width_l = int(level_w)
    height_l = int(level_h)

    count = 0
    for col, row, x0_l, y0_l, x1_l, y1_l in dz_coords(width_l, height_l, s, e):
        # convert LV coordinates to Level-0 coordinates
        x0 = int(round(x0_l * ds))
        y0 = int(round(y0_l * ds))
        w = int(round((x1_l - x0_l)))
        h = int(round((y1_l - y0_l)))

        tile = slide.read_region((x0, y0), level, (w, h)).convert("RGB")

        if not is_not_background(tile, bg_thresh, max_bg_pct):
            continue

        tile.save(out_dir / f"{col}_{row}.png")
        count += 1

    slide.close()
    return count


# ---------------------------------------------------------
# Main tiling pipeline
# ---------------------------------------------------------
def process_slide(slide_path: str, output_root: Path,
                  level: int, s: int, e: int,
                  bg_thresh: int, max_bg_pct: float):

    slide_path = Path(slide_path)
    out_dir = output_root / slide_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f" → Tiling {slide_path.name}")

    args = (
        str(slide_path),
        out_dir,
        level,
        s,
        e,
        bg_thresh,
        max_bg_pct,
    )

    count = tile_worker(args)
    print(f"    Saved {count} tiles.")


# ---------------------------------------------------------
# Run whole CSV list
# ---------------------------------------------------------
def run_csv(csv_path: Path, data_dir: Path, output_dir: Path,
            level: int, s: int, e: int, bg_thresh: int, max_bg_pct: float):

    import pandas as pd
    df = pd.read_csv(csv_path)

    # resolve IDs
    if "image_id" in df.columns:
        ids = df["image_id"].astype(str).tolist()
    else:
        ids = df.iloc[:, 0].astype(str).tolist()

    print(f"Loaded {len(ids)} IDs.")

    extensions = [".tiff", ".tif", ".svs"]

    # find files
    found = []
    missing = []
    for img_id in ids:
        match = None
        for ext in extensions:
            p = data_dir / f"{img_id}{ext}"
            if p.exists():
                match = p
                break
        if match:
            found.append(match)
        else:
            missing.append(img_id)

    print(f"Found {len(found)} slides.")
    print(f"Missing {len(missing)}.\n")

    for i, slide_path in enumerate(found, 1):
        print(f"[{i}/{len(found)}] {slide_path.name}")
        process_slide(
            slide_path,
            output_dir,
            level,
            s,
            e,
            bg_thresh,
            max_bg_pct
        )

    print("✓ All done.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
# ---------------------------------------------------------
# MAIN ENTRY (HARD-CODED PATHS)
# ---------------------------------------------------------
def main():
    # === HARD-CODED SCC PATHS ===
    csv_path     = Path("Official_GTP_PANDAS/PANDAS/data/subsets/train_patch_01.csv")
    data_dir     = Path("PANDA_DATA_MANNY/DATA/train_images")
    output_dir   = Path("PANDA_DATA_MANNY/PATCHES_SUBSET_01")

    # === TILEING PARAMETERS ===
    tile_size    = 512     # -s
    tile_overlap = 0       # -e
    level        = 1       # -L
    bg_thresh    = 220     # --bg_thresh
    max_bg_pct   = 0.50    # -B

    print("Running WSI tiling with hard-coded parameters...")
    print(f"CSV:       {csv_path}")
    print(f"Data dir:  {data_dir}")
    print(f"Output:    {output_dir}")
    print()

    run_csv(
        csv_path=csv_path,
        data_dir=data_dir,
        output_dir=output_dir,
        level=level,
        s=tile_size,
        e=tile_overlap,
        bg_thresh=bg_thresh,
        max_bg_pct=max_bg_pct,
    )


if __name__ == "__main__":
    main()

