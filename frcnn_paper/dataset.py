# =============================================================================
# FASTER R-CNN PAPER - DATASET
# =============================================================================
# Dataset classes and data preparation utilities
# With Box Augmentation following paper specifications
# =============================================================================

import os
import cv2
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import *


# =============================================================================
# BOX AUGMENTATION (Paper Specification)
# =============================================================================
# Paper: b' = b + δ (generate augmented boxes from ground truth)
# =============================================================================

class BoxAugmentation:
    """
    Box Augmentation following paper specifications.
    
    Generates augmented boxes from ground truth to enrich positive samples
    for contrastive learning.
    
    Paper formula: b' = b + δ
    Where δ is random noise applied to box coordinates.
    
    Args:
        num_augments: Number of augmented boxes per GT box (default: 5)
        max_shift_ratio: Maximum shift as ratio of box size (default: 0.1)
        max_scale_ratio: Maximum scale change ratio (default: 0.1)
        iou_threshold: Minimum IoU with original box (default: 0.7)
    """
    
    def __init__(self, num_augments=5, max_shift_ratio=0.1, max_scale_ratio=0.1, iou_threshold=0.7):
        self.num_augments = num_augments
        self.max_shift_ratio = max_shift_ratio
        self.max_scale_ratio = max_scale_ratio
        self.iou_threshold = iou_threshold
    
    def __call__(self, boxes, labels, image_size):
        """
        Apply box augmentation.
        
        Args:
            boxes: Original boxes [N, 4] (x1, y1, x2, y2)
            labels: Original labels [N]
            image_size: (H, W) of image
        
        Returns:
            augmented_boxes: [N + M, 4] where M = N * num_augments
            augmented_labels: [N + M]
            is_augmented: [N + M] boolean mask (True for augmented boxes)
        """
        if len(boxes) == 0:
            return boxes, labels, torch.zeros(0, dtype=torch.bool)
        
        h, w = image_size
        aug_boxes = [boxes]
        aug_labels = [labels]
        is_augmented = [torch.zeros(len(boxes), dtype=torch.bool)]
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.numpy() if torch.is_tensor(box) else box
            box_w = x2 - x1
            box_h = y2 - y1
            
            valid_augments = []
            attempts = 0
            max_attempts = self.num_augments * 3
            
            while len(valid_augments) < self.num_augments and attempts < max_attempts:
                attempts += 1
                
                # Random shift δ (paper: b' = b + δ)
                dx = (random.random() * 2 - 1) * self.max_shift_ratio * box_w
                dy = (random.random() * 2 - 1) * self.max_shift_ratio * box_h
                
                # Random scale
                scale_x = 1 + (random.random() * 2 - 1) * self.max_scale_ratio
                scale_y = 1 + (random.random() * 2 - 1) * self.max_scale_ratio
                
                # Calculate center
                cx = (x1 + x2) / 2 + dx
                cy = (y1 + y2) / 2 + dy
                
                # New dimensions
                new_w = box_w * scale_x
                new_h = box_h * scale_y
                
                # New coordinates
                new_x1 = max(0, cx - new_w / 2)
                new_y1 = max(0, cy - new_h / 2)
                new_x2 = min(w, cx + new_w / 2)
                new_y2 = min(h, cy + new_h / 2)
                
                # Check validity
                if new_x2 <= new_x1 or new_y2 <= new_y1:
                    continue
                
                # Check IoU with original
                iou = self._compute_iou(
                    [x1, y1, x2, y2],
                    [new_x1, new_y1, new_x2, new_y2]
                )
                
                if iou >= self.iou_threshold:
                    valid_augments.append([new_x1, new_y1, new_x2, new_y2])
            
            if valid_augments:
                aug_boxes.append(torch.tensor(valid_augments, dtype=torch.float32))
                aug_labels.append(torch.full((len(valid_augments),), label, dtype=torch.int64))
                is_augmented.append(torch.ones(len(valid_augments), dtype=torch.bool))
        
        return (
            torch.cat(aug_boxes, dim=0),
            torch.cat(aug_labels, dim=0),
            torch.cat(is_augmented, dim=0)
        )
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


# =============================================================================
# IMAGE AUGMENTATION
# =============================================================================

class ImageAugmentation:
    """
    Image augmentation for training.
    
    Args:
        horizontal_flip: Probability of horizontal flip
        vertical_flip: Probability of vertical flip
        color_jitter: Whether to apply color jitter
        brightness: Brightness adjustment range
        contrast: Contrast adjustment range
        saturation: Saturation adjustment range
    """
    
    def __init__(
        self,
        horizontal_flip=0.5,
        vertical_flip=0.5,
        color_jitter=True,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
    
    def __call__(self, image, target):
        """
        Apply augmentations.
        
        Args:
            image: Image tensor [C, H, W]
            target: Dict with 'boxes' and 'labels'
        
        Returns:
            augmented_image, augmented_target
        """
        _, h, w = image.shape
        boxes = target['boxes']
        
        # Horizontal flip
        if random.random() < self.horizontal_flip:
            image = torch.flip(image, dims=[2])
            if len(boxes) > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        
        # Vertical flip
        if random.random() < self.vertical_flip:
            image = torch.flip(image, dims=[1])
            if len(boxes) > 0:
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        
        # Color jitter
        if self.color_jitter:
            image = self._apply_color_jitter(image)
        
        target['boxes'] = boxes
        return image, target
    
    def _apply_color_jitter(self, image):
        """Apply color jitter to image"""
        # Brightness
        if self.brightness > 0:
            factor = 1 + (random.random() * 2 - 1) * self.brightness
            image = image * factor
        
        # Contrast
        if self.contrast > 0:
            factor = 1 + (random.random() * 2 - 1) * self.contrast
            mean = image.mean()
            image = (image - mean) * factor + mean
        
        # Clamp values
        image = torch.clamp(image, 0, 1)
        
        return image


# =============================================================================
# COCO FORMAT CONVERSION
# =============================================================================
def convert_to_coco(csv_path, img_dir, output_json, split='train'):
    """
    Convert RIVA CSV format to COCO JSON format
    
    Args:
        csv_path: Path to annotation CSV
        img_dir: Path to image directory
        output_json: Output JSON path
        split: 'train' or 'val'
    
    Returns:
        output_json path
    """
    print(f"\n📄 Converting {split} to COCO format...")
    
    df = pd.read_csv(csv_path)
    
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
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        for _, row in group.iterrows():
            # Convert center format to COCO (top-left)
            x_center, y_center = row['x'], row['y']
            box_w, box_h = row['width'], row['height']
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


def prepare_coco_dataset():
    """Prepare COCO format annotations"""
    print("\n📂 Preparing COCO format dataset...")
    
    ann_dir = os.path.join(OUTPUT_DIR, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    train_json = os.path.join(ann_dir, 'train.json')
    val_json = os.path.join(ann_dir, 'val.json')
    
    convert_to_coco(TRAIN_CSV, TRAIN_IMG_DIR, train_json, 'train')
    convert_to_coco(VAL_CSV, VAL_IMG_DIR, val_json, 'val')
    
    return train_json, val_json


# =============================================================================
# PYTORCH DATASET
# =============================================================================
class RIVADataset(Dataset):
    """
    RIVA Cervical Cell Detection Dataset with Box Augmentation.
    
    Args:
        csv_path: Path to annotation CSV
        img_dir: Path to image directory
        img_size: Target image size (default: 1200)
        is_train: Whether this is training set (enables augmentation)
        use_box_augmentation: Whether to use box augmentation (paper method)
        use_image_augmentation: Whether to use image augmentation
        box_aug_config: Config dict for BoxAugmentation
    """
    
    def __init__(
        self,
        csv_path,
        img_dir,
        img_size=IMG_SIZE,
        is_train=True,
        use_box_augmentation=True,
        use_image_augmentation=True,
        box_aug_config=None
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        self.image_files = self.df['image_filename'].unique().tolist()
        
        # Initialize augmentations
        self.box_augmentation = None
        self.image_augmentation = None
        
        if is_train:
            if use_box_augmentation:
                box_config = box_aug_config or {
                    'num_augments': 5,
                    'max_shift_ratio': 0.1,
                    'max_scale_ratio': 0.1,
                    'iou_threshold': 0.7
                }
                self.box_augmentation = BoxAugmentation(**box_config)
                print(f"   ✓ Box Augmentation enabled: {box_config}")
            
            if use_image_augmentation:
                self.image_augmentation = ImageAugmentation(
                    horizontal_flip=0.5,
                    vertical_flip=0.5,
                    color_jitter=True,
                    brightness=0.2,
                    contrast=0.2
                )
                print("   ✓ Image Augmentation enabled")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # Load image with Unicode path support (fixes Vietnamese characters in path)
        # cv2.imread doesn't handle non-ASCII paths on Windows, so use np.fromfile + imdecode
        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        
        if img is None:
            # Image not found, return a random different sample
            new_idx = random.randint(0, len(self.image_files) - 1)
            while new_idx == idx:
                new_idx = random.randint(0, len(self.image_files) - 1)
            return self.__getitem__(new_idx)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        
        # Resize to target size
        img = cv2.resize(img, (self.img_size, self.img_size))
        scale_x = self.img_size / w_orig
        scale_y = self.img_size / h_orig
        
        # Get annotations
        ann = self.df[self.df['image_filename'] == img_name]
        
        boxes = []
        labels = []
        
        for _, row in ann.iterrows():
            # Scale coordinates
            x_c = row['x'] * scale_x
            y_c = row['y'] * scale_y
            w = row['width'] * scale_x
            h = row['height'] * scale_y
            
            # Convert to corner format
            x1 = max(0, x_c - w / 2)
            y1 = max(0, y_c - h / 2)
            x2 = min(self.img_size, x_c + w / 2)
            y2 = min(self.img_size, y_c + h / 2)
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(int(row['class']) + 1)  # +1 for background class
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply Box Augmentation (Paper: b' = b + δ)
        if self.box_augmentation is not None and self.is_train:
            boxes, labels, is_augmented = self.box_augmentation(
                boxes, labels, (self.img_size, self.img_size)
            )
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Convert image to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Apply Image Augmentation
        if self.image_augmentation is not None and self.is_train:
            img, target = self.image_augmentation(img, target)
        
        return img, target


def collate_fn(batch):
    """Custom collate function for detection"""
    return tuple(zip(*batch))


def get_data_loaders(use_box_augmentation=True, use_image_augmentation=True):
    """
    Create train and validation data loaders.
    
    Args:
        use_box_augmentation: Enable box augmentation (paper method)
        use_image_augmentation: Enable image augmentations
    
    Returns:
        train_loader, val_loader
    """
    print("\n📊 Creating data loaders...")
    
    train_dataset = RIVADataset(
        TRAIN_CSV,
        TRAIN_IMG_DIR,
        IMG_SIZE,
        is_train=True,
        use_box_augmentation=use_box_augmentation,
        use_image_augmentation=use_image_augmentation
    )
    
    val_dataset = RIVADataset(
        VAL_CSV,
        VAL_IMG_DIR,
        IMG_SIZE,
        is_train=False,  # No augmentation for validation
        use_box_augmentation=False,
        use_image_augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"   Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    print(f"   Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset with augmentation
    print("=" * 60)
    print("Testing dataset with Box Augmentation...")
    print("=" * 60)
    
    train_loader, val_loader = get_data_loaders(
        use_box_augmentation=True,
        use_image_augmentation=True
    )
    
    # Test one batch
    images, targets = next(iter(train_loader))
    print(f"\nBatch test:")
    print(f"  Images: {len(images)}")
    print(f"  First image shape: {images[0].shape}")
    print(f"  First target boxes: {targets[0]['boxes'].shape}")
    print(f"  First target labels: {targets[0]['labels'].shape}")
    
    # Compare with/without augmentation
    print("\n" + "=" * 60)
    print("Comparing augmentation effect:")
    print("=" * 60)
    
    # Without augmentation
    no_aug_dataset = RIVADataset(
        TRAIN_CSV, TRAIN_IMG_DIR, IMG_SIZE,
        is_train=False, use_box_augmentation=False
    )
    _, no_aug_target = no_aug_dataset[0]
    
    # With augmentation
    aug_dataset = RIVADataset(
        TRAIN_CSV, TRAIN_IMG_DIR, IMG_SIZE,
        is_train=True, use_box_augmentation=True
    )
    _, aug_target = aug_dataset[0]
    
    print(f"  Without augmentation: {len(no_aug_target['boxes'])} boxes")
    print(f"  With augmentation: {len(aug_target['boxes'])} boxes")
    print(f"  Augmentation multiplier: {len(aug_target['boxes']) / max(1, len(no_aug_target['boxes'])):.1f}x")

