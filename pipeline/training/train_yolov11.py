# =============================================================================
# RIVA TRACK A - YOLOv11 TRAINING
# =============================================================================
# YOLOv11 is the latest YOLO version with improved architecture and performance
# Compatible with: Kaggle & Local
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
    BASE_DIR = r'd:\NĂM 3\RIVA\yolov11_model'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'
    os.makedirs(BASE_DIR, exist_ok=True)

os.environ['WANDB_MODE'] = 'disabled'

# ===================== INSTALL ULTRALYTICS (LATEST VERSION) =====================
print("📦 Installing latest ultralytics (includes YOLOv11)...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'ultralytics', '-q'])

# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
from tqdm import tqdm
from ultralytics import YOLO

print(f"🔵 Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")

# ===================== PREPARE DATASET =====================
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

def prepare_dataset():
    """Prepare YOLO format dataset"""
    print("📂 Preparing dataset...")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)
    
    for split in ['train', 'val']:
        csv_path = os.path.join(INPUT_DIR, 'annotations', 'annotations', f'{split}.csv')
        src_img_dir = os.path.join(INPUT_DIR, 'images', 'images', split)
        dst_img_dir = os.path.join(DATASET_DIR, 'images', split)
        dst_lbl_dir = os.path.join(DATASET_DIR, 'labels', split)
        
        df = pd.read_csv(csv_path, engine='python')
        
        for filename, group in tqdm(df.groupby('image_filename'), desc=f'{split}'):
            src_img_path = os.path.join(src_img_dir, filename)
            if not os.path.exists(src_img_path):
                continue
            
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
    
    print(f"✅ Dataset ready!")
    return yaml_path

# ===================== TRAIN YOLOv11 =====================
def train_yolov11(yaml_path, model_size='m', epochs=100):
    """
    Train YOLOv11 model
    
    Args:
        yaml_path: Path to data.yaml
        model_size: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
        epochs: Number of training epochs
    """
    print(f"\n🚀 Training YOLOv11{model_size}...")
    
    # Initialize YOLOv11 model
    model = YOLO(f'yolo11{model_size}.pt')
    
    # MINIMAL augmentation - learned from experiments!
    # User found: YOLOv8m with basic augment=True got 0.12 mAP
    #             YOLOv8l with heavy augmentation got 0.0877 mAP
    # Conclusion: LESS IS MORE for this dataset!
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16 if IS_KAGGLE else 8,
        patience=20,
        project=os.path.join(BASE_DIR, 'runs'),
        name=f'yolo11{model_size}',
        exist_ok=True,
        
        # Just basic augmentation - that's it!
        augment=True,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        
        # Other
        workers=4 if IS_KAGGLE else 2,
        verbose=True,
        plots=True
    )
    
    best_path = os.path.join(BASE_DIR, 'runs', f'yolo11{model_size}', 'weights', 'best.pt')
    return best_path, results

# ===================== INFERENCE WITH TTA =====================
def generate_submission(model_path, conf_thresh=0.001):
    """Generate submission with TTA"""
    print(f"\n🔮 Generating submission with TTA...")
    
    model = YOLO(model_path)
    
    test_dir = os.path.join(INPUT_DIR, 'images', 'images', 'test')
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    
    results_list = []
    pred_id = 0
    
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_dir, img_name)
        
        # Predict with TTA
        results = model.predict(
            img_path, 
            conf=conf_thresh,
            iou=0.5,
            augment=True,  # TTA enabled
            verbose=False,
            max_det=300
        )
        
        if len(results[0].boxes) == 0:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            results_list.append({
                'id': pred_id, 'image_filename': img_name, 'class': 0,
                'x': w/2, 'y': h/2, 'width': 100, 'height': 100, 'conf': 0.001
            })
            pred_id += 1
            continue
        
        for det in results[0].boxes:
            box = det.xywh[0].cpu().numpy()
            x_center, y_center, width, height = box
            conf = det.conf[0].cpu().numpy()
            cls = int(det.cls[0].cpu().numpy())
            
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': cls,
                'x': float(x_center),
                'y': float(y_center),
                'width': float(width),
                'height': float(height),
                'conf': float(conf)
            })
            pred_id += 1
    
    # Save submission
    submission_df = pd.DataFrame(results_list)
    submission_df = submission_df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]
    
    submission_path = os.path.join(BASE_DIR, 'submission_yolov11.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"✅ Submission saved to: {submission_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    return submission_path

# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 60)
    print("🏆 RIVA TRACK A - YOLOv11 TRAINING")
    print("=" * 60)
    
    # Prepare dataset
    yaml_path = prepare_dataset()
    
    # Train YOLOv11m (medium size - good balance)
    # You can also try: 'n', 's', 'l', 'x'
    model_path, results = train_yolov11(yaml_path, model_size='m', epochs=100)
    
    print(f"\n✅ Training complete!")
    print(f"Best model: {model_path}")
    
    # Generate submission
    submission_path = generate_submission(model_path, conf_thresh=0.001)
    
    print("\n" + "=" * 60)
    print("🎉 DONE!")
    print("=" * 60)
    print(f"📄 Submission: {submission_path}")
    print("\nNext steps:")
    print("1. Check validation metrics in the training output")
    print("2. Submit to competition")
    print("3. If needed, try larger model: yolo11l or yolo11x")
