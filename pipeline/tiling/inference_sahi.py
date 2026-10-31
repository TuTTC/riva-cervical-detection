# =============================================================================
# INFERENCE VỚI SAHI - TILING LÚC TEST
# =============================================================================
# SAHI (Slicing Aided Hyper Inference) thực hiện tiling tự động lúc inference
# Giúp detect small objects trên ảnh lớn mà không cần resize
# =============================================================================

import os
import sys
import subprocess

# Install SAHI
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sahi', '-q'])

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ===================== CONFIGURATION =====================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
    MODEL_PATH = '/kaggle/working/models/tiled_detection/runs/yolo11x_tiled/weights/best.pt'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'
    MODEL_PATH = r'd:\NĂM 3\RIVA\models\tiled_detection\runs\yolo11x_tiled\weights\best.pt'

# SAHI Tiling parameters (must match training!)
SLICE_SIZE = 1024
OVERLAP_RATIO = 0.2

# Confidence threshold
CONF_THRESH = 0.001  # Low threshold, let NMS handle duplicates
IOU_THRESH = 0.5

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              SAHI INFERENCE FOR HIGH-RES DETECTION           ║
╠══════════════════════════════════════════════════════════════╣
║  Slice Size: {SLICE_SIZE}x{SLICE_SIZE}                                          ║
║  Overlap: {OVERLAP_RATIO*100:.0f}%                                                   ║
║  Model: {os.path.basename(MODEL_PATH):<45}   ║
╚══════════════════════════════════════════════════════════════╝
""")


def load_detection_model():
    """Load YOLO model with SAHI wrapper"""
    print(f"📦 Loading model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return None
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # SAHI uses yolov8 for ultralytics models
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESH,
        device='cuda:0'  # or 'cpu'
    )
    
    print("✅ Model loaded successfully")
    return detection_model


def run_sahi_inference(detection_model, image_path):
    """
    Run SAHI sliced inference on a single image
    
    Returns:
        List of detections: [(class, x_center, y_center, width, height, conf), ...]
    """
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        perform_standard_pred=True,  # Also run on full image
        postprocess_type="NMS",       # Use NMS to merge overlapping detections
        postprocess_match_metric="IOU",
        postprocess_match_threshold=IOU_THRESH,
        verbose=0
    )
    
    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox  # SAHI bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        
        # Convert to center format
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        cls = pred.category.id
        conf = pred.score.value
        
        detections.append((cls, x_center, y_center, width, height, conf))
    
    return detections


def generate_submission(detection_model):
    """Generate submission CSV using SAHI inference"""
    print("\n🔮 Generating submission with SAHI...")
    
    test_dir = os.path.join(INPUT_DIR, 'images', 'images', 'test')
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))])
    
    print(f"📷 Found {len(test_images)} test images")
    
    results_list = []
    pred_id = 0
    
    for img_name in tqdm(test_images, desc="SAHI Inference"):
        img_path = os.path.join(test_dir, img_name)
        
        detections = run_sahi_inference(detection_model, img_path)
        
        if len(detections) == 0:
            # Fallback: add dummy detection
            import cv2
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': 0,
                'x': w / 2,
                'y': h / 2,
                'width': 100,
                'height': 100,
                'conf': 0.001
            })
            pred_id += 1
            continue
        
        for (cls, x_c, y_c, w, h, conf) in detections:
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': int(cls),
                'x': float(x_c),
                'y': float(y_c),
                'width': float(w),
                'height': float(h),
                'conf': float(conf)
            })
            pred_id += 1
    
    # Save submission
    submission_df = pd.DataFrame(results_list)
    submission_df = submission_df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]
    
    submission_path = os.path.join(BASE_DIR, 'submission_sahi.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"   Total predictions: {len(submission_df)}")
    
    return submission_path, submission_df


def save_detections_for_stage2(detection_model, split='test'):
    """
    Run SAHI and save detections for Stage 2 classifier
    Also crops detected cells for classifier training/inference
    """
    print(f"\n📦 Extracting detections for Stage 2 ({split})...")
    
    import cv2
    
    if split == 'test':
        img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'test')
    else:
        img_dir = os.path.join(INPUT_DIR, 'images', 'images', split)
    
    crop_dir = os.path.join(BASE_DIR, 'crops', split)
    os.makedirs(crop_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])
    
    all_detections = []
    crop_id = 0
    
    for img_name in tqdm(image_files, desc=f"Cropping {split}"):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        h_img, w_img = img.shape[:2]
        
        detections = run_sahi_inference(detection_model, img_path)
        
        for (cls, x_c, y_c, w, h, conf) in detections:
            # Calculate crop box with padding
            pad_ratio = 0.1  # 10% padding
            pad_w = int(w * pad_ratio)
            pad_h = int(h * pad_ratio)
            
            x1 = max(0, int(x_c - w/2 - pad_w))
            y1 = max(0, int(y_c - h/2 - pad_h))
            x2 = min(w_img, int(x_c + w/2 + pad_w))
            y2 = min(h_img, int(y_c + h/2 + pad_h))
            
            # Crop and save
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop_filename = f"{crop_id:06d}_{img_name.rsplit('.', 1)[0]}.jpg"
            crop_path = os.path.join(crop_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            all_detections.append({
                'crop_id': crop_id,
                'crop_filename': crop_filename,
                'image_filename': img_name,
                'predicted_class': int(cls),
                'x': float(x_c),
                'y': float(y_c),
                'width': float(w),
                'height': float(h),
                'conf': float(conf)
            })
            crop_id += 1
    
    # Save detection metadata
    det_df = pd.DataFrame(all_detections)
    det_path = os.path.join(BASE_DIR, f'detections_{split}.csv')
    det_df.to_csv(det_path, index=False)
    
    print(f"\n✅ Cropped {len(all_detections)} cells to: {crop_dir}")
    print(f"   Detection metadata: {det_path}")
    
    return det_path


def main():
    # Load model
    detection_model = load_detection_model()
    if detection_model is None:
        return
    
    # Generate submission
    submission_path, submission_df = generate_submission(detection_model)
    
    # Save crops for Stage 2 classifier
    save_detections_for_stage2(detection_model, split='test')
    
    print("\n" + "="*60)
    print("🎉 SAHI INFERENCE COMPLETE!")
    print("="*60)
    print(f"\n📄 Submission (detection only): {submission_path}")
    print("\n📝 Next steps:")
    print("   1. Train Stage 2 classifier on cropped cells")
    print("   2. Run classifier on test crops")
    print("   3. Fuse detection + classification scores")


if __name__ == "__main__":
    main()
