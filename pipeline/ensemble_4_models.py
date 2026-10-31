# =============================================================================
# ENSEMBLE 4 MODELS WITH WEIGHTED BOXES FUSION (WBF)
# =============================================================================
# Combines:
#   - submission_sahi_eda_optimized.csv (YOLO - 0.13)
#   - submission_fixed.csv (Swin-Tiny - 0.12)
#   - submission_frcnn_paper.csv (FRCNN - 0.1075)
#   - submission_swin_large_tta.csv (Swin-Large - NEW)
# =============================================================================

import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Try to import ensemble_boxes
try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except ImportError:
    print("Warning: ensemble_boxes not installed. Run: pip install ensemble-boxes")
    HAS_WBF = False


# =============================================================================
# CONFIGURATION - Based on your proven 0.18855 config
# =============================================================================
SUBMISSIONS_DIR = r"d:\NĂM 3\RIVA\submissions"
SWIN_LARGE_DIR = r"d:\NĂM 3\RIVA\RIVA-TRACK-A\swin_large"

# Input submissions with weights
# Format: (filepath, weight)
# YOLO is most trusted (weight=4), Swin-Large is strong (weight=3)
SUBMISSIONS = [
    (os.path.join(SUBMISSIONS_DIR, "submission_sahi_eda_optimized.csv"), 4),  # YOLO
    (os.path.join(SUBMISSIONS_DIR, "submission_fixed.csv"), 3),               # Swin-Tiny
    (os.path.join(SWIN_LARGE_DIR, "submission_swin_large_tta_fusion.csv"), 3),       # Swin-Large (TTA)
    (os.path.join(SUBMISSIONS_DIR, "submission_frcnn_paper.csv"), 1),         # FRCNN
]

# WBF parameters - YOUR PROVEN CONFIG
IOU_THRESHOLD = 0.55     # Same as your 0.18855 config
SKIP_BOX_THRESHOLD = 0.01  # Same as your 0.18855 config
CONF_TYPE = 'avg'        # How to combine confidence

# Image size for normalization
IMG_SIZE = 1200

# Output
OUTPUT_FILE = os.path.join(SUBMISSIONS_DIR, "submission_ensemble_4models_wbf.csv")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_submission(path):
    """Load submission CSV and group by image"""
    df = pd.read_csv(path)
    
    # Group predictions by image
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        img_name = row['image_filename']
        grouped[img_name].append({
            'class': int(row['class']),
            'conf': float(row['conf']),
            'x': float(row['x']),
            'y': float(row['y']),
            'width': float(row['width']),
            'height': float(row['height'])
        })
    
    return grouped


def convert_to_normalized(boxes, img_size=1200):
    """Convert center-format boxes to normalized corner format [x1, y1, x2, y2] in [0, 1]"""
    normalized = []
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        x1 = (x - w/2) / img_size
        y1 = (y - h/2) / img_size
        x2 = (x + w/2) / img_size
        y2 = (y + h/2) / img_size
        # Clip to [0, 1]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
        normalized.append({
            'class': box['class'],
            'conf': box['conf'],
            'box': [float(x1), float(y1), float(x2), float(y2)]
        })
    return normalized


def convert_from_normalized(boxes, img_size=1200):
    """Convert normalized corner format back to center format"""
    original = []
    for box in boxes:
        x1, y1, x2, y2 = box['box']
        # Convert from normalized [0,1] to pixel coordinates
        x1_px = x1 * img_size
        y1_px = y1 * img_size
        x2_px = x2 * img_size
        y2_px = y2 * img_size
        # Convert to center format
        x = (x1_px + x2_px) / 2
        y = (y1_px + y2_px) / 2
        w = x2_px - x1_px
        h = y2_px - y1_px
        original.append({
            'class': box['class'],
            'conf': box['conf'],
            'x': x,
            'y': y,
            'width': w,
            'height': h
        })
    return original


def wbf_ensemble_image(all_model_boxes, weights, iou_threshold=0.5, skip_threshold=0.0001):
    """
    Apply WBF per class across all models for a single image
    
    Args:
        all_model_boxes: List of lists, each containing normalized box dicts
        weights: Weight for each model
    
    Returns:
        List of box dicts after fusion
    """
    if not HAS_WBF:
        # Fallback: just concatenate all predictions
        all_boxes = []
        for model_boxes in all_model_boxes:
            all_boxes.extend(model_boxes)
        return all_boxes
    
    # Group by class across all models
    class_to_model_boxes = defaultdict(lambda: [[] for _ in range(len(all_model_boxes))])
    class_to_model_scores = defaultdict(lambda: [[] for _ in range(len(all_model_boxes))])
    
    for model_idx, model_boxes in enumerate(all_model_boxes):
        for box in model_boxes:
            cls = box['class']
            class_to_model_boxes[cls][model_idx].append(box['box'])
            class_to_model_scores[cls][model_idx].append(box['conf'])
    
    # Apply WBF per class
    fused_boxes = []
    
    for cls in class_to_model_boxes.keys():
        boxes_list = class_to_model_boxes[cls]
        scores_list = class_to_model_scores[cls]
        labels_list = [[cls] * len(boxes) for boxes in boxes_list]
        
        # Check if any model has predictions for this class
        if all(len(b) == 0 for b in boxes_list):
            continue
        
        try:
            # Ensure all values are float lists (WBF requirement)
            boxes_list = [[list(map(float, box)) for box in boxes] for boxes in boxes_list]
            scores_list = [[float(s) for s in scores] for scores in scores_list]
            labels_list = [[int(l) for l in labels] for labels in labels_list]
            
            boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=[float(w) for w in weights],
                iou_thr=float(iou_threshold),
                skip_box_thr=float(skip_threshold),
                conf_type=CONF_TYPE
            )
            
            for box, score, label in zip(boxes_fused, scores_fused, labels_fused):
                fused_boxes.append({
                    'class': int(label),
                    'conf': float(score),
                    'box': [float(x) for x in box]
                })
                
        except Exception as e:
            print(f"WBF error for class {cls}: {e}")
            # Fallback: keep all boxes
            for model_idx, boxes in enumerate(boxes_list):
                for i, box in enumerate(boxes):
                    fused_boxes.append({
                        'class': cls,
                        'conf': scores_list[model_idx][i],
                        'box': box
                    })
    
    return fused_boxes


# =============================================================================
# MAIN ENSEMBLE
# =============================================================================
def main():
    print("=" * 70)
    print("       4-MODEL WEIGHTED BOXES FUSION (WBF) ENSEMBLE")
    print("=" * 70)
    
    # Load all submissions
    print("\nLoading submissions...")
    all_data = []
    weights = []
    
    for filepath, weight in SUBMISSIONS:
        if not os.path.exists(filepath):
            print(f"   ERROR: {os.path.basename(filepath)} not found at {filepath}!")
            continue
        
        data = load_submission(filepath)
        all_data.append(data)
        weights.append(weight)
        print(f"   Loaded: {os.path.basename(filepath)} (weight: {weight}, {len(data)} images)")
    
    if len(all_data) < 2:
        print("ERROR: Need at least 2 submissions to ensemble!")
        return
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    print(f"\nNormalized weights: {[f'{w:.3f}' for w in weights]}")
    
    # Get all image filenames
    all_images = set()
    for data in all_data:
        all_images.update(data.keys())
    all_images = sorted(list(all_images))
    print(f"\nTotal unique images: {len(all_images)}")
    
    # Ensemble each image
    print("\nRunning WBF ensemble...")
    results = []
    result_id = 0
    
    for img_idx, img_name in enumerate(all_images):
        if (img_idx + 1) % 1000 == 0:
            print(f"   Processed {img_idx + 1}/{len(all_images)} images...")
        
        # Get predictions from each model
        all_model_boxes = []
        for data in all_data:
            boxes = data.get(img_name, [])
            normalized_boxes = convert_to_normalized(boxes, IMG_SIZE)
            all_model_boxes.append(normalized_boxes)
        
        # Apply WBF
        fused_normalized = wbf_ensemble_image(all_model_boxes, weights, IOU_THRESHOLD, SKIP_BOX_THRESHOLD)
        
        # Convert back to center format
        fused_original = convert_from_normalized(fused_normalized, IMG_SIZE)
        
        # Add to results
        for box in fused_original:
            results.append({
                'id': result_id,
                'image_filename': img_name,
                'class': box['class'],
                'x': box['x'],
                'y': box['y'],
                'width': box['width'],
                'height': box['height'],
                'conf': box['conf']
            })
            result_id += 1
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    # Statistics
    print(f"\n{'=' * 70}")
    print(f"   ENSEMBLE COMPLETE!")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Images: {len(all_images)}")
    print(f"   Total predictions: {len(results)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
