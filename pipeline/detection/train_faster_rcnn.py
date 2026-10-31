# =============================================================================
# FASTER R-CNN BASELINE - KAGGLE VERSION
# =============================================================================
# Phase 1: Train Faster R-CNN với ResNet-50-FPN
# Optimized cho Kaggle T4 16GB
# =============================================================================

import os
import sys
import subprocess

# Install MMDetection
print("📦 Installing MMDetection...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                       'mmdet', 'mmcv', 'mmengine', '-q'])

import torch
import pandas as pd
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import shutil

# ===================== CONFIGURATION =====================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'

OUTPUT_DIR = os.path.join(BASE_DIR, 'faster_rcnn')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset config
CLASS_NAMES = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
NUM_CLASSES = 8

# Training config - Optimized for Kaggle T4 16GB
IMG_SIZE = 800  # Reduced from 1024 for memory
BATCH_SIZE = 2
EPOCHS = 24
BASE_LR = 0.005

print(f"""
╔══════════════════════════════════════════════════════════════╗
║         FASTER R-CNN BASELINE TRAINING                       ║
╠══════════════════════════════════════════════════════════════╣
║  Backbone: ResNet-50-FPN                                     ║
║  Input: {IMG_SIZE}x{IMG_SIZE}                                             ║
║  Batch: {BATCH_SIZE}                                                    ║
║  Epochs: {EPOCHS}                                                   ║
╚══════════════════════════════════════════════════════════════╝
""")


# ===================== CONVERT TO COCO FORMAT =====================
def convert_to_coco(csv_path, img_dir, output_json, split='train'):
    """Convert RIVA CSV format to COCO JSON format"""
    print(f"\n📄 Converting {split} to COCO format...")
    
    df = pd.read_csv(csv_path)
    
    # Build COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
    }
    
    # Group by image
    image_id = 0
    ann_id = 0
    
    for img_name, group in tqdm(df.groupby('image_filename'), desc=f"Converting {split}"):
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        
        # Get image size
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Add image
        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        # Add annotations
        for _, row in group.iterrows():
            # Convert center format to COCO format (x, y, width, height)
            x_center = row['x']
            y_center = row['y']
            box_w = row['width']
            box_h = row['height']
            
            # COCO uses top-left corner
            x = x_center - box_w / 2
            y = y_center - box_h / 2
            
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(row['class']),
                "bbox": [x, y, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })
            ann_id += 1
        
        image_id += 1
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    
    print(f"✅ Saved: {output_json}")
    print(f"   Images: {len(coco['images'])}")
    print(f"   Annotations: {len(coco['annotations'])}")
    
    return output_json


def prepare_dataset():
    """Prepare COCO format dataset"""
    print("\n📂 Preparing dataset...")
    
    ann_dir = os.path.join(OUTPUT_DIR, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    # Convert train
    train_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv')
    train_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'train')
    train_json = os.path.join(ann_dir, 'train.json')
    convert_to_coco(train_csv, train_img_dir, train_json, 'train')
    
    # Convert val
    val_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'val.csv')
    val_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'val')
    val_json = os.path.join(ann_dir, 'val.json')
    convert_to_coco(val_csv, val_img_dir, val_json, 'val')
    
    return train_json, val_json


# ===================== MMDETECTION CONFIG =====================
def create_config():
    """Create MMDetection config file"""
    config_content = f'''
# Faster R-CNN with ResNet-50-FPN
_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# Model
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes={NUM_CLASSES})
    )
)

# Dataset
data_root = '{OUTPUT_DIR}/'
train_dataloader = dict(
    batch_size={BATCH_SIZE},
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='{INPUT_DIR}/images/images/train/'),
        metainfo=dict(classes={CLASS_NAMES}),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='{INPUT_DIR}/images/images/val/'),
        metainfo=dict(classes={CLASS_NAMES}),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox'
)

# Optimizer - Following paper config
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr={BASE_LR}, momentum=0.9, weight_decay=0.0001)
)

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 14], gamma=0.1)
]

# Training config
train_cfg = dict(max_epochs={EPOCHS}, val_interval=2)

# Work directory
work_dir = '{OUTPUT_DIR}/work_dirs'

# Load pretrained weights
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
'''
    
    config_path = os.path.join(OUTPUT_DIR, 'faster_rcnn_config.py')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Config saved: {config_path}")
    return config_path


# ===================== TRAIN WITH MMDETECTION =====================
def train_mmdet(config_path):
    """Train using MMDetection"""
    from mmdet.apis import init_detector, train_detector
    from mmdet.registry import RUNNERS
    from mmengine.config import Config
    from mmengine.runner import Runner
    
    print("\n🚀 Starting training...")
    
    cfg = Config.fromfile(config_path)
    
    # Build runner
    runner = Runner.from_cfg(cfg)
    
    # Train
    runner.train()
    
    print("\n✅ Training complete!")
    return os.path.join(OUTPUT_DIR, 'work_dirs', 'epoch_24.pth')


# ===================== ALTERNATIVE: SIMPLE PYTORCH TRAINING =====================
def train_simple():
    """Simpler training using torchvision Faster R-CNN"""
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import Dataset, DataLoader
    
    print("\n🚀 Training with torchvision Faster R-CNN...")
    
    # Create model
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # Replace head for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)  # +1 for background
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Dataset class
    class RIVADataset(Dataset):
        def __init__(self, csv_path, img_dir, transforms=None):
            self.df = pd.read_csv(csv_path)
            self.img_dir = img_dir
            self.transforms = transforms
            self.image_files = self.df['image_filename'].unique()
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            h_orig, w_orig = cv2.imread(img_path).shape[:2]
            scale_x = IMG_SIZE / w_orig
            scale_y = IMG_SIZE / h_orig
            
            # Get annotations
            ann = self.df[self.df['image_filename'] == img_name]
            
            boxes = []
            labels = []
            for _, row in ann.iterrows():
                x_c = row['x'] * scale_x
                y_c = row['y'] * scale_y
                w = row['width'] * scale_x
                h = row['height'] * scale_y
                
                x1 = x_c - w/2
                y1 = y_c - h/2
                x2 = x_c + w/2
                y2 = y_c + h/2
                
                boxes.append([x1, y1, x2, y2])
                labels.append(int(row['class']) + 1)  # +1 for background
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            target = {
                'boxes': boxes,
                'labels': labels,
            }
            
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            return img, target
    
    # Create datasets
    train_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv')
    train_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'train')
    val_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'val.csv')
    val_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'val')
    
    train_dataset = RIVADataset(train_csv, train_img_dir)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, collate_fn=collate_fn)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=BASE_LR, momentum=0.9, weight_decay=0.0001)
    
    # LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 14], gamma=0.1)
    
    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        lr_scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'frcnn_epoch_{epoch+1}.pth'))
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, 'faster_rcnn_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Model saved: {final_path}")
    
    return final_path, model


# ===================== INFERENCE =====================
def generate_submission(model, model_path):
    """Generate submission file"""
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    print("\n🔮 Generating submission...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
        model.load_state_dict(torch.load(model_path))
    
    model.to(device)
    model.eval()
    
    test_dir = os.path.join(INPUT_DIR, 'images', 'images', 'test')
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))])
    
    results = []
    pred_id = 0
    
    for img_name in tqdm(test_images, desc="Inference"):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(img_tensor)[0]
        
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        # Scale back to original size
        scale_x = w_orig / IMG_SIZE
        scale_y = h_orig / IMG_SIZE
        
        if len(boxes) == 0:
            results.append({
                'id': pred_id, 'image_filename': img_name,
                'class': 0, 'x': w_orig/2, 'y': h_orig/2,
                'width': 100, 'height': 100, 'conf': 0.001
            })
            pred_id += 1
            continue
        
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.001:
                continue
            
            x1, y1, x2, y2 = box
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            results.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': int(label) - 1,  # Remove background offset
                'x': float(x_center),
                'y': float(y_center),
                'width': float(width),
                'height': float(height),
                'conf': float(score)
            })
            pred_id += 1
    
    df = pd.DataFrame(results)
    submission_path = os.path.join(BASE_DIR, 'submission_frcnn.csv')
    df.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"   Total predictions: {len(df)}")
    
    return submission_path


# ===================== MAIN =====================
if __name__ == "__main__":
    print("="*60)
    print("🏆 FASTER R-CNN BASELINE TRAINING")
    print("="*60)
    
    # Option 1: Simple PyTorch training (recommended for Kaggle)
    model_path, model = train_simple()
    
    # Generate submission
    generate_submission(model, model_path)
    
    print("\n" + "="*60)
    print("🎉 DONE!")
    print("="*60)
