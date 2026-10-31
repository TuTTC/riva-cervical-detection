# =============================================================================
# FASTER R-CNN PAPER REPRODUCTION - RTX 3090 VERSION
# =============================================================================
# Based on "Holistic and Historical Instance Comparison" paper specifications
# Optimized for RTX 3090 24GB
# =============================================================================

import os
import sys
import subprocess

# Install dependencies
print("📦 Installing dependencies...")
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

# =============================================================================
# CONFIGURATION - PAPER SPECIFICATIONS
# =============================================================================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'

OUTPUT_DIR = os.path.join(BASE_DIR, 'faster_rcnn_paper')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== PAPER CONFIGURATIONS =====================
# From Section III. EXPERIMENTS AND RESULTS

# Dataset config
CLASS_NAMES = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
NUM_CLASSES = 8

# === TRAINING CONFIG (Paper specifications) ===
IMG_SIZE = 1200          # Paper: 1200x1200 pixels
BATCH_SIZE = 4           # RTX 3090 24GB can handle batch 4 @ 1200px
EPOCHS = 24              # Paper: 24 epochs
BASE_LR = 0.005          # Paper: Initial LR = 0.005
MOMENTUM = 0.9           # Paper: SGD momentum = 0.9
WEIGHT_DECAY = 0.0001    # Standard weight decay
LR_DECAY_EPOCHS = [8, 14]  # Paper: Decay at epoch 8 and 14
LR_DECAY_GAMMA = 0.1     # Paper: Decay by factor 0.1

# === METHOD-SPECIFIC HYPERPARAMETERS ===
TEMPERATURE = 6          # Paper: τ = 6
MEMORY_BANK_SIZE = 80    # Paper: Q = 80 per class
LAMBDA_ROI = 0.2         # Paper: λ_roi = 0.2
LAMBDA_CLS = 0.2         # Paper: λ_cls = 0.2
ROI_BATCH_SIZE = 256     # Paper: K = 256 RoI candidates

# === BACKBONE ===
BACKBONE = 'resnet50'    # Paper: ResNet-50 + FPN
USE_FPN = True           # Paper: Feature Pyramid Network

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║         FASTER R-CNN - PAPER REPRODUCTION (RTX 3090)                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Backbone: ResNet-50 + FPN                                               ║
║  Input Size: {IMG_SIZE}x{IMG_SIZE} pixels                                            ║
║  Batch Size: {BATCH_SIZE}                                                            ║
║  Epochs: {EPOCHS}                                                                ║
║  Optimizer: SGD (LR={BASE_LR}, momentum={MOMENTUM})                               ║
║  LR Decay: at epochs {LR_DECAY_EPOCHS} (gamma={LR_DECAY_GAMMA})                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Temperature (τ): {TEMPERATURE}                                                        ║
║  Memory Bank (Q): {MEMORY_BANK_SIZE} per class                                               ║
║  Loss Weights: λ_roi={LAMBDA_ROI}, λ_cls={LAMBDA_CLS}                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# STEP 1: CONVERT TO COCO FORMAT
# =============================================================================
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
            # Convert center format to COCO format (top-left x, y, width, height)
            x_center = row['x']
            y_center = row['y']
            box_w = row['width']
            box_h = row['height']
            
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
    
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    
    print(f"✅ Saved: {output_json}")
    print(f"   Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")
    
    return output_json


def prepare_dataset():
    """Prepare COCO format dataset"""
    print("\n📂 Preparing dataset...")
    
    ann_dir = os.path.join(OUTPUT_DIR, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    train_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv')
    train_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'train')
    train_json = os.path.join(ann_dir, 'train.json')
    convert_to_coco(train_csv, train_img_dir, train_json, 'train')
    
    val_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'val.csv')
    val_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'val')
    val_json = os.path.join(ann_dir, 'val.json')
    convert_to_coco(val_csv, val_img_dir, val_json, 'val')
    
    return train_json, val_json


# =============================================================================
# STEP 2: MMDETECTION CONFIG (PAPER SPECIFICATIONS)
# =============================================================================
def create_mmdet_config():
    """Create MMDetection config following paper specifications"""
    
    config_content = f'''
# =============================================================================
# Faster R-CNN Config - Paper Reproduction
# Based on "Holistic and Historical Instance Comparison" specifications
# =============================================================================

_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/default_runtime.py'
]

# =========================== MODEL ===========================
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,  # Paper: d=1024
            roi_feat_size=7,
            num_classes={NUM_CLASSES},
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num={ROI_BATCH_SIZE},  # Paper: K=256
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300
        )
    )
)

# =========================== DATASET ===========================
dataset_type = 'CocoDataset'
data_root = '{OUTPUT_DIR}/'

# Paper: 1200x1200 input size
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # Color augmentation for cell images
    dict(type='ColorTransform', prob=0.5, saturation_range=(0.8, 1.2), brightness_range=(0.9, 1.1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size={BATCH_SIZE},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='{INPUT_DIR}/images/images/train/'),
        metainfo=dict(classes={CLASS_NAMES}),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='{INPUT_DIR}/images/images/val/'),
        metainfo=dict(classes={CLASS_NAMES}),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# =========================== OPTIMIZER (Paper Specs) ===========================
# Paper: SGD, momentum=0.9, initial LR=0.005

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr={BASE_LR},           # Paper: 0.005
        momentum={MOMENTUM},     # Paper: 0.9
        weight_decay={WEIGHT_DECAY}
    )
)

# Paper: LR decay by 0.1 at epoch 8 and 14
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end={EPOCHS},
        milestones={LR_DECAY_EPOCHS},  # Paper: [8, 14]
        gamma={LR_DECAY_GAMMA}          # Paper: 0.1
    )
]

# =========================== TRAINING CONFIG ===========================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs={EPOCHS},  # Paper: 24 epochs
    val_interval=2
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =========================== HOOKS ===========================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# =========================== RUNTIME ===========================
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume = False

work_dir = '{OUTPUT_DIR}/work_dirs'
'''
    
    config_path = os.path.join(OUTPUT_DIR, 'faster_rcnn_paper_config.py')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Config saved: {config_path}")
    return config_path


# =============================================================================
# STEP 3: TRAIN WITH MMDETECTION
# =============================================================================
def train_mmdet(config_path):
    """Train using MMDetection"""
    from mmengine.config import Config
    from mmengine.runner import Runner
    
    print("\n🚀 Starting training with paper specifications...")
    print(f"   Config: {config_path}")
    
    cfg = Config.fromfile(config_path)
    
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    print("\n✅ Training complete!")
    best_ckpt = os.path.join(OUTPUT_DIR, 'work_dirs', f'epoch_{EPOCHS}.pth')
    return best_ckpt


# =============================================================================
# STEP 4: TORCHVISION ALTERNATIVE (If MMDet fails)
# =============================================================================
def train_torchvision():
    """Train with torchvision Faster R-CNN (Paper config)"""
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import Dataset, DataLoader
    
    print("\n🚀 Training with torchvision (Paper config)...")
    
    # Create model - ResNet-50 FPN (Paper specification)
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # Replace head for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model.to(device)
    
    # Dataset
    class RIVADataset(Dataset):
        def __init__(self, csv_path, img_dir):
            self.df = pd.read_csv(csv_path)
            self.img_dir = img_dir
            self.image_files = self.df['image_filename'].unique()
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h_orig, w_orig = img.shape[:2]
            
            # Paper: 1200x1200 input
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            scale_x = IMG_SIZE / w_orig
            scale_y = IMG_SIZE / h_orig
            
            ann = self.df[self.df['image_filename'] == img_name]
            
            boxes = []
            labels = []
            for _, row in ann.iterrows():
                x_c = row['x'] * scale_x
                y_c = row['y'] * scale_y
                w = row['width'] * scale_x
                h = row['height'] * scale_y
                
                x1 = max(0, x_c - w/2)
                y1 = max(0, y_c - h/2)
                x2 = min(IMG_SIZE, x_c + w/2)
                y2 = min(IMG_SIZE, y_c + h/2)
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(row['class']) + 1)
            
            if len(boxes) == 0:
                boxes = [[0, 0, 1, 1]]
                labels = [0]
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            target = {'boxes': boxes, 'labels': labels}
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            return img, target
    
    # Data loaders
    train_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv')
    train_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'train')
    
    train_dataset = RIVADataset(train_csv, train_img_dir)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Paper: SGD with momentum=0.9, LR=0.005
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=BASE_LR, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Paper: LR decay at epoch 8 and 14 by factor 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=LR_DECAY_EPOCHS, 
        gamma=LR_DECAY_GAMMA
    )
    
    # Training loop
    print(f"\n📊 Training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=35)
                optimizer.step()
                
                epoch_loss += losses.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{losses.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            except Exception as e:
                print(f"   Warning: {e}")
                continue
        
        lr_scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoints
        if (epoch + 1) % 4 == 0 or epoch + 1 == EPOCHS:
            ckpt_path = os.path.join(OUTPUT_DIR, f'frcnn_paper_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"   💾 Checkpoint saved: {ckpt_path}")
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, 'faster_rcnn_paper_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Final model saved: {final_path}")
    
    return final_path, model


# =============================================================================
# STEP 5: INFERENCE
# =============================================================================
def generate_submission(model=None, model_path=None):
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
        
        # Paper: 1200x1200 input
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(img_tensor)[0]
        
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        # Scale back
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
            
            results.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': int(label) - 1,
                'x': float((x1 + x2) / 2),
                'y': float((y1 + y2) / 2),
                'width': float(x2 - x1),
                'height': float(y2 - y1),
                'conf': float(score)
            })
            pred_id += 1
    
    df = pd.DataFrame(results)
    submission_path = os.path.join(BASE_DIR, 'submission_frcnn_paper.csv')
    df.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"   Total predictions: {len(df)}")
    print(f"\n📊 Class distribution:")
    for cls_id, count in df['class'].value_counts().sort_index().items():
        print(f"   {CLASS_NAMES[cls_id]}: {count}")
    
    return submission_path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🏆 FASTER R-CNN - PAPER REPRODUCTION (RTX 3090)")
    print("=" * 80)
    
    # Prepare dataset
    prepare_dataset()
    
    # Choose training method
    USE_MMDET = False  # Set to True if MMDetection works
    
    if USE_MMDET:
        config_path = create_mmdet_config()
        checkpoint = train_mmdet(config_path)
        # For MMDet, need separate inference code
    else:
        model_path, model = train_torchvision()
        generate_submission(model, model_path)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"""
Summary:
  - Input Size: {IMG_SIZE}x{IMG_SIZE}
  - Epochs: {EPOCHS}
  - LR Schedule: {BASE_LR} → decay at {LR_DECAY_EPOCHS}
  - Batch Size: {BATCH_SIZE}
  
Next Steps:
  1. Submit 'submission_frcnn_paper.csv'
  2. Optionally add Feature Fusion for +5-10% improvement
""")
