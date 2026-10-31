# =============================================================================
# RIVA TRACK A - OPTIMIZED COMPETITION PIPELINE
# =============================================================================
# This script includes ALL optimizations to maximize mAP@0.50:0.95:
# 1. Larger model (YOLOv8l/x)
# 2. Higher resolution training
# 3. Stronger augmentation
# 4. Test-Time Augmentation (TTA)
# 5. Ensemble with WBF
# 6. Confidence threshold optimization
# =============================================================================

import os
import sys
import subprocess

# ===================== CONFIGURATION =====================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA\optimized_model'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'
    os.makedirs(BASE_DIR, exist_ok=True)

os.environ['WANDB_MODE'] = 'disabled'

# ===================== INSTALL DEPENDENCIES =====================
def install_packages():
    packages = ['ultralytics', 'ensemble-boxes']
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

install_packages()

# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

print(f"🔵 Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")

# ===================== PREPARE DATASET =====================
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

def prepare_dataset():
    """Prepare YOLO format dataset with proper structure"""
    print("📂 Preparing dataset...")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)
    
    for split in ['train', 'val']:
        csv_path = os.path.join(INPUT_DIR, 'annotations', 'annotations', f'{split}.csv')
        src_img_dir = os.path.join(INPUT_DIR, 'images', 'images', split)
        dst_img_dir = os.path.join(DATASET_DIR, 'images', split)
        dst_lbl_dir = os.path.join(DATASET_DIR, 'labels', split)
        
        df = pd.read_csv(csv_path)
        
        for filename, group in tqdm(df.groupby('image_filename'), desc=f'{split}'):
            src_img_path = os.path.join(src_img_dir, filename)
            if not os.path.exists(src_img_path):
                continue
            
            # Copy image
            dst_img_path = os.path.join(dst_img_dir, filename)
            if not os.path.exists(dst_img_path):
                shutil.copy(src_img_path, dst_img_path)
            
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            
            yolo_lines = []
            for _, row in group.iterrows():
                cls = int(row['class'])
                x_n = max(0, min(1, row['x'] / w_img))
                y_n = max(0, min(1, row['y'] / h_img))
                w_n = max(0, min(1, row['width'] / w_img))
                h_n = max(0, min(1, row['height'] / h_img))
                yolo_lines.append(f"{cls} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}")
            
            txt_name = filename.rsplit('.', 1)[0] + '.txt'
            with open(os.path.join(dst_lbl_dir, txt_name), 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    # Create YAML config
    yaml_config = {
        'path': DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 8,
        'names': ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
    }
    
    yaml_path = os.path.join(BASE_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f)
    
    print(f"✅ Dataset ready at: {DATASET_DIR}")
    return yaml_path

# ===================== TRAIN MULTIPLE MODELS =====================
def train_model(model_name, imgsz, epochs, run_name):
    """Train a single model with optimized settings"""
    print(f"\n🚀 Training {model_name} at {imgsz}px...")
    
    yaml_path = os.path.join(BASE_DIR, 'data.yaml')
    model = YOLO(model_name)
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=8 if imgsz >= 1280 else 16,
        patience=15,
        project=os.path.join(BASE_DIR, 'runs'),
        name=run_name,
        exist_ok=True,
        
        # === STRONG AUGMENTATION ===
        augment=True,
        mosaic=1.0,
        mixup=0.3,           # Increased mixup
        copy_paste=0.1,      # Copy-paste augmentation
        degrees=30,          # Rotation
        translate=0.2,
        scale=0.5,
        shear=10,
        flipud=0.5,          # Vertical flip
        fliplr=0.5,          # Horizontal flip
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # === REGULARIZATION ===
        dropout=0.1,
        weight_decay=0.0005,
        
        # === OPTIMIZER ===
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        
        # === OTHER ===
        workers=4 if IS_KAGGLE else 2,
        verbose=True,
        plots=True
    )
    
    best_path = os.path.join(BASE_DIR, 'runs', run_name, 'weights', 'best.pt')
    return best_path

# ===================== ENSEMBLE INFERENCE =====================
def ensemble_predict(models, img_path, conf_thresh=0.001, iou_thresh=0.5):
    """
    Run ensemble prediction with Weighted Boxes Fusion
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for model in models:
        # Run prediction with TTA
        results = model.predict(
            img_path, 
            conf=conf_thresh, 
            iou=iou_thresh,
            augment=True,  # TTA enabled!
            verbose=False,
            max_det=300
        )
        
        if len(results[0].boxes) == 0:
            continue
        
        boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Normalize boxes to 0-1
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        boxes_norm = np.clip(boxes_norm, 0, 1)
        
        all_boxes.append(boxes_norm.tolist())
        all_scores.append(scores.tolist())
        all_labels.append(labels.tolist())
    
    if len(all_boxes) == 0:
        return [], [], []
    
    # Weighted Boxes Fusion
    weights = [1] * len(all_boxes)  # Equal weights for all models
    
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, 
        all_scores, 
        all_labels,
        weights=weights,
        iou_thr=0.5,
        skip_box_thr=conf_thresh
    )
    
    # Denormalize boxes back to pixel coordinates
    fused_boxes[:, [0, 2]] *= w
    fused_boxes[:, [1, 3]] *= h
    
    return fused_boxes, fused_scores, fused_labels

def xyxy_to_xywh(boxes):
    """Convert xyxy format to xywh (center) format"""
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]         # width
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]         # height
    return xywh

# ===================== GENERATE SUBMISSION =====================
def generate_submission(models, conf_thresh=0.001):
    """Generate submission with ensemble and TTA"""
    print(f"\n🔮 Generating submission with {len(models)} models...")
    
    test_dir = os.path.join(INPUT_DIR, 'images', 'images', 'test')
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    
    results_list = []
    prediction_id = 0
    
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_dir, img_name)
        
        boxes, scores, labels = ensemble_predict(
            models, img_path, conf_thresh=conf_thresh
        )
        
        if len(boxes) == 0:
            # No detections - add dummy
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            results_list.append({
                'id': prediction_id,
                'image_filename': img_name,
                'class': 0,
                'x': w / 2,
                'y': h / 2,
                'width': 100,
                'height': 100,
                'conf': 0.001
            })
            prediction_id += 1
            continue
        
        # Convert xyxy to xywh
        boxes_xywh = xyxy_to_xywh(boxes)
        
        for i in range(len(boxes)):
            results_list.append({
                'id': prediction_id,
                'image_filename': img_name,
                'class': int(labels[i]),
                'x': float(boxes_xywh[i, 0]),
                'y': float(boxes_xywh[i, 1]),
                'width': float(boxes_xywh[i, 2]),
                'height': float(boxes_xywh[i, 3]),
                'conf': float(scores[i])
            })
            prediction_id += 1
    
    # Create DataFrame
    submission_df = pd.DataFrame(results_list)
    submission_df = submission_df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]
    
    # Save
    submission_path = os.path.join(BASE_DIR, 'submission_optimized.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"✅ Submission saved to: {submission_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Stats
    print(f"\nClass distribution:")
    class_names = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
    for cls_id, count in submission_df['class'].value_counts().sort_index().items():
        print(f"  {class_names[cls_id]}: {count}")
    
    return submission_path

# ===================== MAIN PIPELINE =====================
if __name__ == "__main__":
    print("=" * 60)
    print("🏆 RIVA TRACK A - OPTIMIZED COMPETITION PIPELINE")
    print("=" * 60)
    
    # Step 1: Prepare dataset
    yaml_path = prepare_dataset()
    
    # Step 2: Train multiple models for ensemble
    # Option A: Quick (single model) - uncomment this for faster results
    # models_to_train = [('yolov8l.pt', 640, 50, 'yolov8l_640')]
    
    # Option B: Full ensemble (slower but better) - use this for best results
    models_to_train = [
        ('yolov8m.pt', 640, 50, 'yolov8m_640'),
        ('yolov8l.pt', 640, 50, 'yolov8l_640'),
        # ('yolov8l.pt', 1280, 30, 'yolov8l_1280'),  # Uncomment if you have GPU memory
    ]
    
    trained_model_paths = []
    for model_name, imgsz, epochs, run_name in models_to_train:
        model_path = train_model(model_name, imgsz, epochs, run_name)
        trained_model_paths.append(model_path)
        print(f"✅ Trained: {model_path}")
    
    # Step 3: Load all trained models
    print("\n📥 Loading trained models for ensemble...")
    models = [YOLO(path) for path in trained_model_paths]
    
    # Step 4: Generate submission with ensemble + TTA
    # Try different confidence thresholds
    best_conf = 0.001  # Start low, YOLO default is 0.25
    submission_path = generate_submission(models, conf_thresh=best_conf)
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"📄 Submission: {submission_path}")
    print(f"📊 Models used: {len(models)}")
    print(f"⚙️ Confidence threshold: {best_conf}")
    print("\nTips to improve further:")
    print("1. Train for more epochs (100+)")
    print("2. Add YOLOv8x to the ensemble")
    print("3. Try different confidence thresholds")
    print("4. Use higher resolution (1280)")
