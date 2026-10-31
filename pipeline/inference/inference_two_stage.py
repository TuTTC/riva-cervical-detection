# =============================================================================
# TWO-STAGE PAP SMEAR - END-TO-END INFERENCE & SUBMISSION
# =============================================================================
# This script runs the complete two-stage pipeline on test images:
# 1. Detect all cells using YOLO
# 2. Classify each detected cell using EfficientNet
# 3. Generate submission CSV
#
# Compatible with: Kaggle & Local
# =============================================================================

import os
import sys
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# ===================== CONFIGURATION =====================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA\two_stage_model'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'

# Model paths
DETECTOR_PATH = os.path.join(BASE_DIR, 'detector_best.pt')
CLASSIFIER_PATH = os.path.join(BASE_DIR, 'classifier_best.pt')

# Test images
TEST_IMG_DIR = os.path.join(INPUT_DIR, 'images', 'images', 'test')

# Output
SUBMISSION_PATH = os.path.join(BASE_DIR, 'submission_two_stage.csv')

# Inference settings
DETECTOR_CONF = 0.15  # Low confidence for high recall
CLASSIFIER_IMG_SIZE = 128
CROP_PADDING = 0.15

# Class names (ORIGINAL ORDER - matches submission format)
CLASS_NAMES_ORIGINAL = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']

# ImageFolder uses ALPHABETICAL order, so we need to map:
# IMPORTANT: ImageFolder sorts alphabetically: ASCH, ASCUS, ENDO, HSIL, INFL, LSIL, NILM, SCC
# This means: alphabetical index -> original index
ALPHABETICAL_TO_ORIGINAL = {
    0: 6,  # ASCH -> 6
    1: 3,  # ASCUS -> 3
    2: 1,  # ENDO -> 1
    3: 5,  # HSIL -> 5
    4: 2,  # INFL -> 2
    5: 4,  # LSIL -> 4
    6: 0,  # NILM -> 0
    7: 7,  # SCC -> 7
}

CLASS_NAMES = CLASS_NAMES_ORIGINAL  # For compatibility

print(f"🔵 Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"📁 Test images: {TEST_IMG_DIR}")

# ===================== INSTALL DEPENDENCIES =====================
import subprocess

def install_if_needed(packages):
    for package in packages:
        try:
            __import__(package.split('==')[0].replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

install_if_needed(['ultralytics', 'timm', 'albumentations'])

# ===================== IMPORTS =====================
from ultralytics import YOLO
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# ===================== LOAD MODELS =====================
print("\n📥 Loading models...")

# Load YOLO detector
print(f"  Loading detector from: {DETECTOR_PATH}")
detector = YOLO(DETECTOR_PATH)

# Load EfficientNet classifier
print(f"  Loading classifier from: {CLASSIFIER_PATH}")
checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
classifier = timm.create_model('efficientnet_b3', pretrained=False, num_classes=8)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier = classifier.to(device)
classifier.eval()

print("✅ Models loaded successfully!")

# ===================== PREPROCESSING =====================
# Transform for classifier input
classifier_transform = A.Compose([
    A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def crop_cell(img, x_center, y_center, width, height, padding=0.15):
    """Crop a cell from image with padding"""
    h_img, w_img = img.shape[:2]
    
    pad_w = width * padding
    pad_h = height * padding
    
    x1 = int(max(0, x_center - width/2 - pad_w))
    y1 = int(max(0, y_center - height/2 - pad_h))
    x2 = int(min(w_img, x_center + width/2 + pad_w))
    y2 = int(min(h_img, y_center + height/2 + pad_h))
    
    crop = img[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    return crop

def classify_crop(crop, model, transform):
    """Classify a single cropped cell"""
    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Apply transform
    transformed = transform(image=crop_rgb)
    tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted_class = probs.max(1)
    
    return predicted_class.item(), confidence.item()

# ===================== INFERENCE =====================
print("\n🚀 Running inference on test images...")

# Get test images
test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg'))]
print(f"Found {len(test_images)} test images")

results = []
prediction_id = 0

for img_name in tqdm(test_images):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    # Stage 1: Detect cells
    detections = detector.predict(img_path, conf=DETECTOR_CONF, verbose=False)
    
    if len(detections) == 0 or len(detections[0].boxes) == 0:
        # No detections - still need to include image in submission
        # Add a dummy prediction with low confidence
        results.append({
            'id': prediction_id,
            'image_filename': img_name,
            'class': 0,  # Default to NILM
            'x': img.shape[1] / 2,
            'y': img.shape[0] / 2,
            'width': 100,
            'height': 100,
            'conf': 0.01
        })
        prediction_id += 1
        continue
    
    # Process each detection
    for det in detections[0].boxes:
        # Get bounding box (xywh format - center x, center y, width, height)
        box = det.xywh[0].cpu().numpy()
        x_center, y_center, width, height = box
        detector_conf = det.conf[0].cpu().numpy()
        
        # Crop cell
        crop = crop_cell(img, x_center, y_center, width, height, CROP_PADDING)
        
        if crop is None or crop.size == 0:
            continue
        
        # Stage 2: Classify cell
        predicted_class_alphabetical, classifier_conf = classify_crop(crop, classifier, classifier_transform)
        
        # CRITICAL: Map from alphabetical index to original index
        predicted_class = ALPHABETICAL_TO_ORIGINAL[predicted_class_alphabetical]
        
        # Combine confidences (geometric mean)
        combined_conf = np.sqrt(detector_conf * classifier_conf)
        
        # Add to results
        results.append({
            'id': prediction_id,
            'image_filename': img_name,
            'class': predicted_class,
            'x': float(x_center),
            'y': float(y_center),
            'width': float(width),
            'height': float(height),
            'conf': float(combined_conf)
        })
        prediction_id += 1

print(f"\n✅ Generated {len(results)} predictions")

# ===================== CREATE SUBMISSION =====================
print("\n📝 Creating submission file...")

# Create DataFrame
submission_df = pd.DataFrame(results)

# Ensure correct column order: id, image_filename, class, x, y, width, height, conf
submission_df = submission_df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]

# Save
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"✅ Submission saved to: {SUBMISSION_PATH}")
print(f"\nSubmission preview:")
print(submission_df.head(10))

# ===================== STATISTICS =====================
print("\n📊 Prediction Statistics:")
print(f"Total predictions: {len(submission_df)}")
print(f"Unique images: {submission_df['image_filename'].nunique()}")
print(f"\nClass distribution:")
class_counts = submission_df['class'].value_counts().sort_index()
for class_id, count in class_counts.items():
    print(f"  {CLASS_NAMES[class_id]}: {count} ({count/len(submission_df)*100:.1f}%)")

print(f"\nConfidence statistics:")
print(f"  Mean: {submission_df['conf'].mean():.4f}")
print(f"  Min:  {submission_df['conf'].min():.4f}")
print(f"  Max:  {submission_df['conf'].max():.4f}")

print("\n🎉 Two-Stage Inference Complete!")
