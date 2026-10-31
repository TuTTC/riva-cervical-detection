# =============================================================================
# TWO-STAGE PAP SMEAR DETECTION - STAGE 1: SINGLE-CLASS YOLO DETECTOR
# =============================================================================
# This notebook trains a YOLOv8 model to detect ALL cells as a single class.
# The goal is HIGH RECALL - we want to find every cell, classification comes later.
#
# Compatible with: Kaggle & Local
# =============================================================================

# %% [markdown]
# # Stage 1: Single-Class Cell Detector
# 
# ## Architecture Overview
# - **Model**: YOLOv8m (medium size, good balance of speed & accuracy)
# - **Classes**: 1 (just "cell" - no class distinction)
# - **Goal**: Maximize recall - find ALL cells in the image

# %% 
# ===================== CONFIGURATION =====================
import os
import sys

# Detect environment
IS_KAGGLE = os.path.exists('/kaggle/working')
IS_COLAB = 'google.colab' in sys.modules

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
    print("🔵 Running on Kaggle")
elif IS_COLAB:
    BASE_DIR = '/content'
    INPUT_DIR = '/content/data'  # Adjust path as needed
    print("🟢 Running on Google Colab")
else:
    # Local environment - adjust these paths
    BASE_DIR = r'd:\NĂM 3\RIVA\two_stage_model'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'
    print("🟠 Running Locally")
    os.makedirs(BASE_DIR, exist_ok=True)

DATASET_DIR = os.path.join(BASE_DIR, 'detector_dataset')
print(f"Base directory: {BASE_DIR}")
print(f"Input directory: {INPUT_DIR}")

# %%
# ===================== INSTALL DEPENDENCIES =====================
import subprocess

def install_if_needed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

install_if_needed('ultralytics')
install_if_needed('opencv-python')

# %%
# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
from tqdm import tqdm
from pathlib import Path

# Disable wandb for cleaner output
os.environ['WANDB_MODE'] = 'disabled'

from ultralytics import YOLO

# %%
# ===================== PREPARE SINGLE-CLASS LABELS =====================
print("🔄 Preparing single-class detector dataset...")

# Create directory structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

# Process train and val sets
for split in ['train', 'val']:
    print(f"\n--- Processing {split.upper()} set ---")
    
    # Paths
    csv_path = os.path.join(INPUT_DIR, 'annotations', 'annotations', f'{split}.csv')
    src_img_dir = os.path.join(INPUT_DIR, 'images', 'images', split)
    dst_img_dir = os.path.join(DATASET_DIR, 'images', split)
    dst_lbl_dir = os.path.join(DATASET_DIR, 'labels', split)
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} annotations")
    
    count_images = 0
    count_boxes = 0
    
    # Group by image
    for filename, group in tqdm(df.groupby('image_filename')):
        src_img_path = os.path.join(src_img_dir, filename)
        
        if not os.path.exists(src_img_path):
            continue
        
        # Copy image
        dst_img_path = os.path.join(dst_img_dir, filename)
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)
        
        # Read image to get dimensions
        img = cv2.imread(src_img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        
        # Create YOLO format labels - ALL CLASS 0 (single class)
        yolo_lines = []
        for _, row in group.iterrows():
            # Original has: class, x_center, y_center, width, height
            # We convert ALL classes to 0
            x_center = row['x']
            y_center = row['y']
            w = row['width']
            h = row['height']
            
            # Normalize to 0-1
            x_n = max(0, min(1, x_center / w_img))
            y_n = max(0, min(1, y_center / h_img))
            w_n = max(0, min(1, w / w_img))
            h_n = max(0, min(1, h / h_img))
            
            # CLASS 0 for all cells
            yolo_lines.append(f"0 {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}")
            count_boxes += 1
        
        # Write label file
        txt_name = filename.rsplit('.', 1)[0] + '.txt'
        with open(os.path.join(dst_lbl_dir, txt_name), 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        count_images += 1
    
    print(f"✅ {split}: {count_images} images, {count_boxes} cell boxes (all class 0)")

# %%
# ===================== CREATE YAML CONFIG =====================
print("\n📝 Creating YAML configuration...")

yaml_config = {
    'path': DATASET_DIR,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,  # Single class!
    'names': {
        0: 'cell'
    }
}

yaml_path = os.path.join(BASE_DIR, 'cell_detector.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_config, f, default_flow_style=False)

print(f"✅ YAML saved to: {yaml_path}")
print("Config contents:")
print(yaml.dump(yaml_config, default_flow_style=False))

# %%
# ===================== TRAIN DETECTOR =====================
print("\n🚀 Starting Single-Class Detector Training...")

# Initialize model
model = YOLO('yolov8m.pt')

# Training configuration - optimized for HIGH RECALL
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16 if IS_KAGGLE else 8,  # Smaller batch for local
    patience=15,
    project=os.path.join(BASE_DIR, 'detector_runs'),
    name='single_class_detector',
    exist_ok=True,
    
    # Augmentation for better generalization
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    
    # Optimizer settings
    optimizer='auto',
    cos_lr=True,
    
    # Detection settings - we want HIGH RECALL
    iou=0.5,  # Lower IoU threshold = more detections
    
    # Other
    workers=4 if IS_KAGGLE else 2,
    verbose=True
)

print("\n✅ Training Complete!")

# %%
# ===================== EVALUATE DETECTOR =====================
print("\n📊 Evaluating detector on validation set...")

# Load best model
best_model_path = os.path.join(BASE_DIR, 'detector_runs', 'single_class_detector', 'weights', 'best.pt')
detector = YOLO(best_model_path)

# Validate
metrics = detector.val(data=yaml_path)

print(f"""
=== DETECTOR PERFORMANCE ===
mAP50:     {metrics.box.map50:.4f}
mAP50-95:  {metrics.box.map:.4f}
Precision: {metrics.box.p[0]:.4f}
Recall:    {metrics.box.r[0]:.4f}
============================
""")

# Check if recall is good (should be > 0.7)
if metrics.box.r[0] < 0.7:
    print("⚠️ Warning: Recall is below 0.7. Consider:")
    print("   - Lowering confidence threshold during inference")
    print("   - Training for more epochs")
    print("   - Adjusting augmentation")
else:
    print("✅ Recall is good! Detector should find most cells.")

# %%
# ===================== SAVE MODEL FOR STAGE 2 =====================
final_model_path = os.path.join(BASE_DIR, 'detector_best.pt')
shutil.copy(best_model_path, final_model_path)
print(f"\n📦 Model saved to: {final_model_path}")

# %%
# ===================== TEST INFERENCE =====================
print("\n🔍 Testing inference on sample images...")

# Get some val images for visualization
val_img_dir = os.path.join(DATASET_DIR, 'images', 'val')
sample_images = os.listdir(val_img_dir)[:3]

for img_name in sample_images:
    img_path = os.path.join(val_img_dir, img_name)
    
    # Run detection with lower confidence for more detections
    results = detector.predict(img_path, conf=0.15, verbose=False)
    
    n_detections = len(results[0].boxes)
    print(f"{img_name}: {n_detections} cells detected")

print("\n🎉 Stage 1 Complete! Detector is ready for Stage 2.")
print(f"📍 Model location: {final_model_path}")
