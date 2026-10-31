# =============================================================================
# BƯỚC 3: FUSION - GỘP DETECTION + CLASSIFICATION
# =============================================================================
# Công thức: Final_Score = α × Detection_conf + (1-α) × Classifier_conf
# Với α = 0.3 (Classifier đáng tin hơn Detection cho classification task)
# =============================================================================

import os
import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'timm', '-q'])

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import timm
import torchvision.transforms as T
from PIL import Image

# ===================== CONFIGURATION =====================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
    DETECTOR_MODEL = '/kaggle/working/runs/yolo11l_tiled/weights/best.pt'
    CLASSIFIER_MODEL = '/kaggle/working/models/classifier/best_classifier.pth'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'
    DETECTOR_MODEL = r'd:\NĂM 3\RIVA\models\tiled_detection\runs\yolo11x_tiled\weights\best.pt'
    CLASSIFIER_MODEL = r'd:\NĂM 3\RIVA\models\classifier\best_classifier.pth'

# Fusion parameters
ALPHA = 0.3  # Detection weight (0.3 = Classifier has 70% weight)
CONF_THRESH = 0.001
IOU_THRESH = 0.5

# Classifier settings
IMG_SIZE = 224
CLASS_NAMES = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
NUM_CLASSES = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          TWO-STAGE FUSION INFERENCE                          ║
╠══════════════════════════════════════════════════════════════╣
║  Stage 1: SAHI Detection (boxes + initial class)             ║
║  Stage 2: EfficientNet Classification (refined class)        ║
║  Fusion: α={ALPHA} Detection + {1-ALPHA} Classifier                      ║
╚══════════════════════════════════════════════════════════════╝
""")


# ===================== LOAD MODELS =====================
def load_detector():
    """Load detection model with SAHI"""
    from sahi import AutoDetectionModel
    
    print(f"📦 Loading detector: {DETECTOR_MODEL}")
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=DETECTOR_MODEL,
        confidence_threshold=CONF_THRESH,
        device=DEVICE
    )
    return detection_model


def load_classifier():
    """Load classifier model"""
    print(f"📦 Loading classifier: {CLASSIFIER_MODEL}")
    
    model = timm.create_model('efficientnet_b5', pretrained=False, num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(CLASSIFIER_MODEL, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"   Loaded from epoch {checkpoint.get('epoch', 'N/A')}, acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    return model


def get_classifier_transforms():
    """Transforms for classifier inference"""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ===================== INFERENCE =====================
def run_sahi_detection(detection_model, image_path, slice_size=1024, overlap=0.2):
    """Run SAHI detection on image"""
    from sahi.predict import get_sliced_prediction
    
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=IOU_THRESH,
        verbose=0
    )
    
    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x_center = (bbox.minx + bbox.maxx) / 2
        y_center = (bbox.miny + bbox.maxy) / 2
        width = bbox.maxx - bbox.minx
        height = bbox.maxy - bbox.miny
        
        detections.append({
            'x': x_center,
            'y': y_center,
            'width': width,
            'height': height,
            'det_class': pred.category.id,
            'det_conf': pred.score.value
        })
    
    return detections


def classify_crop(classifier_model, crop_img, transform):
    """Run classifier on a single crop"""
    # Prepare input
    img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = classifier_model(img_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_class = probs.max(1)
    
    return pred_class.item(), conf.item(), probs[0].cpu().numpy()


def extract_crop(img, detection, pad_ratio=0.15):
    """Extract crop from image based on detection"""
    h_img, w_img = img.shape[:2]
    
    x_c, y_c = detection['x'], detection['y']
    w, h = detection['width'], detection['height']
    
    # Add padding
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    
    x1 = max(0, int(x_c - w/2 - pad_w))
    y1 = max(0, int(y_c - h/2 - pad_h))
    x2 = min(w_img, int(x_c + w/2 + pad_w))
    y2 = min(h_img, int(y_c + h/2 + pad_h))
    
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def fuse_predictions(det_class, det_conf, cls_class, cls_conf, cls_probs, alpha=0.3):
    """
    Fuse detection and classification predictions
    
    Strategy:
    - If classifier is confident (>0.8), trust classifier
    - Otherwise, weighted average of both
    """
    # If classifier very confident, trust it
    if cls_conf > 0.8:
        return cls_class, cls_conf
    
    # Otherwise, weighted fusion
    # Create one-hot for detection
    det_probs = np.zeros(NUM_CLASSES)
    det_probs[det_class] = det_conf
    
    # Weighted average
    fused_probs = alpha * det_probs + (1 - alpha) * cls_probs
    
    fused_class = np.argmax(fused_probs)
    fused_conf = fused_probs[fused_class]
    
    return fused_class, fused_conf


# ===================== MAIN PIPELINE =====================
def run_two_stage_inference():
    """Run complete two-stage inference pipeline"""
    print("\n🚀 Starting Two-Stage Inference...")
    
    # Load models
    detector = load_detector()
    classifier = load_classifier()
    transform = get_classifier_transforms()
    
    # Test directory
    test_dir = os.path.join(INPUT_DIR, 'images', 'images', 'test')
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))])
    
    print(f"\n📷 Found {len(test_images)} test images")
    
    results_list = []
    pred_id = 0
    
    for img_name in tqdm(test_images, desc="Two-Stage Inference"):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        h_img, w_img = img.shape[:2]
        
        # Stage 1: Detection
        detections = run_sahi_detection(detector, img_path)
        
        if len(detections) == 0:
            # Fallback
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': 0,
                'x': w_img / 2,
                'y': h_img / 2,
                'width': 100,
                'height': 100,
                'conf': 0.001
            })
            pred_id += 1
            continue
        
        # Stage 2: Classification for each detection
        for det in detections:
            crop = extract_crop(img, det)
            
            if crop is None or crop.size == 0:
                # Use detection class
                final_class = det['det_class']
                final_conf = det['det_conf']
            else:
                # Run classifier
                cls_class, cls_conf, cls_probs = classify_crop(classifier, crop, transform)
                
                # Fuse predictions
                final_class, final_conf = fuse_predictions(
                    det['det_class'], det['det_conf'],
                    cls_class, cls_conf, cls_probs,
                    alpha=ALPHA
                )
            
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': int(final_class),
                'x': float(det['x']),
                'y': float(det['y']),
                'width': float(det['width']),
                'height': float(det['height']),
                'conf': float(final_conf)
            })
            pred_id += 1
    
    # Save submission
    submission_df = pd.DataFrame(results_list)
    submission_df = submission_df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]
    
    submission_path = os.path.join(BASE_DIR, 'submission_two_stage.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"   Total predictions: {len(submission_df)}")
    
    # Class distribution
    print("\n📊 Prediction distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = len(submission_df[submission_df['class'] == i])
        pct = 100 * count / len(submission_df)
        print(f"   [{i}] {name:6s}: {count:5d} ({pct:5.1f}%)")
    
    return submission_path


def compare_with_detection_only(detection_csv, fused_csv):
    """Compare detection-only vs two-stage predictions"""
    if not os.path.exists(detection_csv):
        print(f"⚠️ Detection CSV not found: {detection_csv}")
        return
    
    det_df = pd.read_csv(detection_csv)
    fused_df = pd.read_csv(fused_csv)
    
    print("\n📊 Comparison: Detection vs Two-Stage")
    print("="*50)
    
    # Count class changes
    changed = 0
    for i in range(min(len(det_df), len(fused_df))):
        if det_df.iloc[i]['class'] != fused_df.iloc[i]['class']:
            changed += 1
    
    print(f"   Class changes: {changed} / {len(fused_df)} ({100*changed/len(fused_df):.1f}%)")
    
    print("\n   Detection-only class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = len(det_df[det_df['class'] == i])
        print(f"      [{i}] {name:6s}: {count}")
    
    print("\n   Two-Stage class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = len(fused_df[fused_df['class'] == i])
        print(f"      [{i}] {name:6s}: {count}")


if __name__ == "__main__":
    submission_path = run_two_stage_inference()
    
    # Compare with detection-only
    detection_csv = os.path.join(BASE_DIR, 'submission_sahi.csv')
    compare_with_detection_only(detection_csv, submission_path)
    
    print("\n" + "="*60)
    print("🎉 TWO-STAGE INFERENCE COMPLETE!")
    print("="*60)
    print(f"\n📄 Final submission: {submission_path}")
    print("\n💡 Tips for improvement:")
    print("   1. Try different α values (0.2-0.4)")
    print("   2. Ensemble multiple classifiers (EfficientNet + ViT)")
    print("   3. Use Test-Time Augmentation (TTA) for classifier")
