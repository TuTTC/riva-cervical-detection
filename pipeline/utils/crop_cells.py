# =============================================================================
# TWO-STAGE PAP SMEAR - CELL CROPPING PIPELINE
# =============================================================================
# This script crops individual cells from training images using ground truth
# bounding boxes to create a classification dataset for Stage 2.
#
# Compatible with: Kaggle & Local
# =============================================================================

import os
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# ===================== CONFIGURATION =====================
# Detect environment
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA\two_stage_model'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'

CROP_DIR = os.path.join(BASE_DIR, 'cropped_cells')
CROP_SIZE = 128  # Output size for each crop
PADDING_RATIO = 0.15  # Add 15% padding around bounding box

# Class mapping
CLASS_NAMES = {
    0: 'NILM',
    1: 'ENDO', 
    2: 'INFL',
    3: 'ASCUS',
    4: 'LSIL',
    5: 'HSIL',
    6: 'ASCH',
    7: 'SCC'
}

print(f"🔵 Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"📁 Output directory: {CROP_DIR}")

# ===================== CREATE DIRECTORY STRUCTURE =====================
print("\n📂 Creating directory structure...")

# Clean up existing
if os.path.exists(CROP_DIR):
    shutil.rmtree(CROP_DIR)

# Create class folders for train and val
for split in ['train', 'val']:
    for class_name in CLASS_NAMES.values():
        path = os.path.join(CROP_DIR, split, class_name)
        os.makedirs(path, exist_ok=True)

print("✅ Directory structure created")

# ===================== CROP CELLS =====================
def crop_cell(img, x_center, y_center, width, height, padding_ratio=0.15):
    """
    Crop a cell from the image with padding.
    
    Args:
        img: Full image (numpy array)
        x_center, y_center: Center coordinates of bounding box
        width, height: Bounding box dimensions
        padding_ratio: Extra padding around the crop
    
    Returns:
        Cropped and resized cell image
    """
    h_img, w_img = img.shape[:2]
    
    # Add padding
    pad_w = width * padding_ratio
    pad_h = height * padding_ratio
    
    # Calculate crop coordinates
    x1 = int(max(0, x_center - width/2 - pad_w))
    y1 = int(max(0, y_center - height/2 - pad_h))
    x2 = int(min(w_img, x_center + width/2 + pad_w))
    y2 = int(min(h_img, y_center + height/2 + pad_h))
    
    # Crop
    crop = img[y1:y2, x1:x2]
    
    # Handle edge case of empty crop
    if crop.size == 0:
        return None
    
    # Resize to standard size
    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)
    
    return crop_resized

# Process train and val sets
for split in ['train', 'val']:
    print(f"\n--- Processing {split.upper()} set ---")
    
    csv_path = os.path.join(INPUT_DIR, 'annotations', 'annotations', f'{split}.csv')
    img_dir = os.path.join(INPUT_DIR, 'images', 'images', split)
    
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path)
    
    # Track counts per class
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    
    # Process each image
    for filename, group in tqdm(df.groupby('image_filename')):
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Crop each cell in this image
        for idx, row in enumerate(group.itertuples()):
            class_id = int(row._7)  # 'class' column
            class_name = CLASS_NAMES.get(class_id, 'UNKNOWN')
            
            if class_name == 'UNKNOWN':
                continue
            
            # Crop the cell
            crop = crop_cell(
                img, 
                row.x, row.y, 
                row.width, row.height,
                PADDING_RATIO
            )
            
            if crop is None:
                continue
            
            # Generate unique filename
            base_name = filename.rsplit('.', 1)[0]
            crop_filename = f"{base_name}_cell{idx}.png"
            crop_path = os.path.join(CROP_DIR, split, class_name, crop_filename)
            
            # Save crop
            cv2.imwrite(crop_path, crop)
            class_counts[class_name] += 1
    
    # Print statistics
    print(f"\n📊 {split.upper()} set statistics:")
    total = sum(class_counts.values())
    for class_name, count in sorted(class_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"   {class_name}: {count} ({pct:.1f}%)")
    print(f"   TOTAL: {total}")

# ===================== SUMMARY =====================
print("\n" + "="*50)
print("🎉 CELL CROPPING COMPLETE!")
print("="*50)
print(f"\n📁 Cropped cells saved to: {CROP_DIR}")
print(f"📐 Crop size: {CROP_SIZE}x{CROP_SIZE} pixels")
print("\nDirectory structure:")
print(f"{CROP_DIR}/")
print("├── train/")
for name in CLASS_NAMES.values():
    print(f"│   ├── {name}/")
print("└── val/")
for name in CLASS_NAMES.values():
    print(f"    ├── {name}/")

print("\n✅ Ready for Stage 2: Classification Training!")
