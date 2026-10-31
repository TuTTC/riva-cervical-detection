# =============================================================================
# SWIN TRANSFORMER LARGE - DATASET
# =============================================================================
# COCO format dataset with augmentations and imbalance handling
# =============================================================================

import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler

from config import (
    CLASS_NAMES, NUM_CLASSES, IMG_SIZE,
    BOX_AUG_CONFIG, TRAIN_SCALES,
    FLIP_PROB_H, FLIP_PROB_V,
    USE_CLASS_AWARE_SAMPLING, CLASS_AWARE_REPEAT_FACTOR,
    USE_COPY_PASTE, COPY_PASTE_CONFIG,
    USE_MOSAIC, MOSAIC_CONFIG
)


# =============================================================================
# BOX AUGMENTATION (Paper: b' = b + delta)
# =============================================================================

class BoxAugmentation:
    """Box Augmentation following paper specifications."""
    
    def __init__(self, num_augments=3, max_shift_ratio=0.1, max_scale_ratio=0.1, iou_threshold=0.7):
        self.num_augments = num_augments
        self.max_shift_ratio = max_shift_ratio
        self.max_scale_ratio = max_scale_ratio
        self.iou_threshold = iou_threshold
    
    def __call__(self, boxes, labels, image_size):
        if len(boxes) == 0:
            return boxes, labels
        
        h, w = image_size
        aug_boxes = [boxes]
        aug_labels = [labels]
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            box_w = x2 - x1
            box_h = y2 - y1
            
            valid_augments = []
            valid_labels = []
            attempts = 0
            max_attempts = self.num_augments * 3
            
            while len(valid_augments) < self.num_augments and attempts < max_attempts:
                attempts += 1
                dx = (random.random() * 2 - 1) * self.max_shift_ratio * box_w
                dy = (random.random() * 2 - 1) * self.max_shift_ratio * box_h
                scale_x = 1 + (random.random() * 2 - 1) * self.max_scale_ratio
                scale_y = 1 + (random.random() * 2 - 1) * self.max_scale_ratio
                
                cx = (x1 + x2) / 2 + dx
                cy = (y1 + y2) / 2 + dy
                new_w = box_w * scale_x
                new_h = box_h * scale_y
                
                new_x1 = max(0, cx - new_w / 2)
                new_y1 = max(0, cy - new_h / 2)
                new_x2 = min(w, cx + new_w / 2)
                new_y2 = min(h, cy + new_h / 2)
                
                if new_x2 <= new_x1 or new_y2 <= new_y1:
                    continue
                
                iou = self._compute_iou([x1, y1, x2, y2], [new_x1, new_y1, new_x2, new_y2])
                
                if iou >= self.iou_threshold:
                    valid_augments.append([new_x1, new_y1, new_x2, new_y2])
                    valid_labels.append(label)
            
            if valid_augments:
                aug_boxes.append(np.array(valid_augments, dtype=np.float32))
                aug_labels.append(np.array(valid_labels, dtype=np.int64))
        
        return np.concatenate(aug_boxes, axis=0), np.concatenate(aug_labels, axis=0)
    
    def _compute_iou(self, box1, box2):
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
# COPY-PASTE AUGMENTATION (for rare classes)
# =============================================================================

class CopyPasteAugmentation:
    """
    Copy-paste augmentation for rare classes.
    Copies cell instances from rare classes and pastes them onto other images.
    """
    
    def __init__(self, cell_bank=None, prob=0.5, max_paste=5, 
                 target_classes=[4, 5, 6, 7], scale_range=(0.8, 1.2)):
        """
        Args:
            cell_bank: Dict of {class_id: [(crop, bbox_wh), ...]}
            prob: Probability of applying copy-paste
            max_paste: Maximum cells to paste per image
            target_classes: Classes to paste (rare classes)
            scale_range: Scale variation for pasted cells
        """
        self.cell_bank = cell_bank or {}
        self.prob = prob
        self.max_paste = max_paste
        self.target_classes = target_classes
        self.scale_range = scale_range
    
    def build_cell_bank(self, coco_json, image_dir, max_per_class=200):
        """Build cell bank from training data."""
        print("Building cell bank for copy-paste augmentation...")
        
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id to filename mapping
        id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Collect cells by class
        self.cell_bank = defaultdict(list)
        
        for ann in tqdm(coco_data['annotations'], desc='Extracting cells'):
            cls_id = ann['category_id']
            if cls_id not in self.target_classes:
                continue
            
            if len(self.cell_bank[cls_id]) >= max_per_class:
                continue
            
            img_file = id_to_file.get(ann['image_id'])
            if not img_file:
                continue
            
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            x, y, w, h = map(int, ann['bbox'])
            
            # Add padding
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            
            crop = img[y1:y2, x1:x2].copy()
            
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                self.cell_bank[cls_id].append({
                    'crop': crop,
                    'bbox_wh': (w, h),
                    'class_id': cls_id
                })
        
        for cls_id in self.target_classes:
            print(f"  Class {CLASS_NAMES[cls_id]}: {len(self.cell_bank[cls_id])} cells")
        
        return self.cell_bank
    
    def __call__(self, image, boxes, labels):
        """Apply copy-paste augmentation."""
        if random.random() > self.prob or not self.cell_bank:
            return image, boxes, labels
        
        h, w = image.shape[:2]
        image = image.copy()
        boxes = boxes.copy()
        labels = labels.copy()
        
        new_boxes = []
        new_labels = []
        
        # Select random cells to paste
        n_paste = random.randint(1, self.max_paste)
        
        for _ in range(n_paste):
            # Random class from target classes
            available_classes = [c for c in self.target_classes if len(self.cell_bank.get(c, [])) > 0]
            if not available_classes:
                break
            
            cls_id = random.choice(available_classes)
            cell = random.choice(self.cell_bank[cls_id])
            
            crop = cell['crop']
            orig_w, orig_h = cell['bbox_wh']
            
            # Random scale
            scale = random.uniform(*self.scale_range)
            new_crop_h = int(crop.shape[0] * scale)
            new_crop_w = int(crop.shape[1] * scale)
            
            if new_crop_h < 10 or new_crop_w < 10:
                continue
            
            crop_resized = cv2.resize(crop, (new_crop_w, new_crop_h))
            
            # Random position (avoid edges)
            max_x = w - new_crop_w - 10
            max_y = h - new_crop_h - 10
            
            if max_x < 10 or max_y < 10:
                continue
            
            paste_x = random.randint(10, max_x)
            paste_y = random.randint(10, max_y)
            
            # Check overlap with existing boxes
            paste_box = [paste_x, paste_y, paste_x + new_crop_w, paste_y + new_crop_h]
            has_overlap = False
            
            for box in boxes:
                iou = self._compute_iou(paste_box, box)
                if iou > 0.3:
                    has_overlap = True
                    break
            
            if has_overlap:
                continue
            
            # Paste with alpha blending at edges
            image[paste_y:paste_y+new_crop_h, paste_x:paste_x+new_crop_w] = crop_resized
            
            # Add new box
            new_boxes.append([paste_x, paste_y, paste_x + new_crop_w, paste_y + new_crop_h])
            new_labels.append(cls_id)
        
        if new_boxes:
            boxes = np.concatenate([boxes, np.array(new_boxes, dtype=np.float32)], axis=0)
            labels = np.concatenate([labels, np.array(new_labels, dtype=np.int64)], axis=0)
        
        return image, boxes, labels
    
    def _compute_iou(self, box1, box2):
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
# MOSAIC AUGMENTATION
# =============================================================================

class MosaicAugmentation:
    """
    Mosaic augmentation: combine 4 images into one.
    Increases diversity and helps with small object detection.
    """
    
    def __init__(self, dataset, prob=0.5, img_scale=(1280, 1280), 
                 center_ratio_range=(0.5, 1.5), pad_val=114.0):
        self.dataset = dataset
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
    
    def __call__(self, idx, image, boxes, labels):
        """Apply mosaic augmentation."""
        if random.random() > self.prob:
            return image, boxes, labels
        
        # Get 3 additional random images
        indices = [idx] + random.choices(range(len(self.dataset)), k=3)
        
        # Target size
        h, w = self.img_scale
        
        # Random center point
        cx = int(w * random.uniform(*self.center_ratio_range) / 2)
        cy = int(h * random.uniform(*self.center_ratio_range) / 2)
        
        # Create mosaic image
        mosaic_img = np.full((h, w, 3), self.pad_val, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []
        
        for i, index in enumerate(indices):
            if i == 0:
                img_i, boxes_i, labels_i = image, boxes, labels
            else:
                img_i, target_i = self.dataset[index]
                # Convert back from tensor
                if isinstance(img_i, torch.Tensor):
                    img_i = (img_i.permute(1, 2, 0).numpy() * 
                            np.array([58.395, 57.12, 57.375]) + 
                            np.array([123.675, 116.28, 103.53])).astype(np.uint8)
                boxes_i = target_i['boxes'].numpy() if isinstance(target_i['boxes'], torch.Tensor) else target_i['boxes']
                labels_i = target_i['labels'].numpy() if isinstance(target_i['labels'], torch.Tensor) else target_i['labels']
            
            h_i, w_i = img_i.shape[:2]
            
            # Position for this quadrant
            if i == 0:    # top-left
                x1, y1, x2, y2 = max(cx - w_i, 0), max(cy - h_i, 0), cx, cy
                crop_x1, crop_y1 = w_i - (x2 - x1), h_i - (y2 - y1)
                crop_x2, crop_y2 = w_i, h_i
            elif i == 1:  # top-right
                x1, y1, x2, y2 = cx, max(cy - h_i, 0), min(cx + w_i, w), cy
                crop_x1, crop_y1 = 0, h_i - (y2 - y1)
                crop_x2, crop_y2 = x2 - x1, h_i
            elif i == 2:  # bottom-left
                x1, y1, x2, y2 = max(cx - w_i, 0), cy, cx, min(cy + h_i, h)
                crop_x1, crop_y1 = w_i - (x2 - x1), 0
                crop_x2, crop_y2 = w_i, y2 - y1
            else:         # bottom-right
                x1, y1, x2, y2 = cx, cy, min(cx + w_i, w), min(cy + h_i, h)
                crop_x1, crop_y1 = 0, 0
                crop_x2, crop_y2 = x2 - x1, y2 - y1
            
            # Paste image
            mosaic_img[y1:y2, x1:x2] = img_i[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Adjust boxes
            if len(boxes_i) > 0:
                boxes_adj = boxes_i.copy()
                boxes_adj[:, [0, 2]] = boxes_adj[:, [0, 2]] - crop_x1 + x1
                boxes_adj[:, [1, 3]] = boxes_adj[:, [1, 3]] - crop_y1 + y1
                
                # Clip to mosaic bounds
                boxes_adj[:, [0, 2]] = np.clip(boxes_adj[:, [0, 2]], 0, w)
                boxes_adj[:, [1, 3]] = np.clip(boxes_adj[:, [1, 3]], 0, h)
                
                # Filter valid boxes
                valid = (boxes_adj[:, 2] > boxes_adj[:, 0]) & (boxes_adj[:, 3] > boxes_adj[:, 1])
                boxes_adj = boxes_adj[valid]
                labels_adj = labels_i[valid]
                
                mosaic_boxes.append(boxes_adj)
                mosaic_labels.append(labels_adj)
        
        if mosaic_boxes:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.zeros((0,), dtype=np.int64)
        
        return mosaic_img, mosaic_boxes, mosaic_labels


# =============================================================================
# CLASS-AWARE SAMPLER
# =============================================================================

class ClassAwareSampler(Sampler):
    """
    Sampler that oversamples rare classes using repeat factor sampling.
    """
    
    def __init__(self, dataset, repeat_factor_thresh=0.5):
        """
        Args:
            dataset: Dataset with annotations
            repeat_factor_thresh: Threshold for repeat factor calculation
        """
        self.dataset = dataset
        self.repeat_factor_thresh = repeat_factor_thresh
        
        # Calculate class frequencies
        class_counts = defaultdict(int)
        image_classes = []
        
        for i in range(len(dataset)):
            img_info = dataset.coco_data['images'][i]
            img_id = img_info['id']
            anns = dataset.img_id_to_anns.get(img_id, [])
            
            classes_in_image = set()
            for ann in anns:
                cls_id = ann['category_id']
                class_counts[cls_id] += 1
                classes_in_image.add(cls_id)
            
            image_classes.append(classes_in_image)
        
        # Calculate repeat factors
        total = sum(class_counts.values())
        class_freq = {c: count / total for c, count in class_counts.items()}
        
        # Repeat factor for each class
        repeat_factors = {}
        for cls_id, freq in class_freq.items():
            if freq < repeat_factor_thresh:
                repeat_factors[cls_id] = max(1, repeat_factor_thresh / freq)
            else:
                repeat_factors[cls_id] = 1.0
        
        # Calculate repeat factor for each image
        self.image_repeat_factors = []
        for classes in image_classes:
            if classes:
                rf = max(repeat_factors.get(c, 1.0) for c in classes)
            else:
                rf = 1.0
            self.image_repeat_factors.append(rf)
        
        # Build indices list with repetition
        self.indices = []
        for i, rf in enumerate(self.image_repeat_factors):
            self.indices.extend([i] * int(rf))
        
        print(f"ClassAwareSampler: {len(dataset)} images -> {len(self.indices)} samples")
        print(f"  Repeat factors: {repeat_factors}")
    
    def __iter__(self):
        indices = self.indices.copy()
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.indices)


# =============================================================================
# IMAGE AUGMENTATION
# =============================================================================

class ImageAugmentation:
    """Image-level augmentations for training"""
    
    def __init__(self, flip_h_prob=0.5, flip_v_prob=0.5, train_scales=None):
        self.flip_h_prob = flip_h_prob
        self.flip_v_prob = flip_v_prob
        self.train_scales = train_scales or [(1280, 1280)]
    
    def __call__(self, image, boxes):
        h, w = image.shape[:2]
        
        if random.random() < self.flip_h_prob:
            image = np.fliplr(image).copy()
            if len(boxes) > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        
        if random.random() < self.flip_v_prob:
            image = np.flipud(image).copy()
            if len(boxes) > 0:
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        
        target_size = random.choice(self.train_scales)
        scale_x = target_size[1] / w
        scale_y = target_size[0] / h
        
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        return image, boxes


# =============================================================================
# COCO FORMAT CONVERSION
# =============================================================================

def convert_to_coco_format(csv_path, image_dir, output_json, split='train'):
    """Convert CSV annotations to COCO JSON format"""
    
    print(f"Converting {split} data to COCO format...")
    
    df = pd.read_csv(csv_path)
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": CLASS_NAMES[i]} for i in range(NUM_CLASSES)]
    }
    
    image_id = 0
    annotation_id = 0
    
    for img_name in tqdm(df['image_filename'].unique(), desc=f'Processing {split}'):
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        height, width = img.shape[:2]
        
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        img_annotations = df[df['image_filename'] == img_name]
        
        for _, row in img_annotations.iterrows():
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(row['class']),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            
            annotation_id += 1
        
        image_id += 1
    
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)
    
    print(f"Saved COCO format to {output_json}")
    print(f"  Images: {len(coco_format['images'])}")
    print(f"  Annotations: {len(coco_format['annotations'])}")
    
    # Print class distribution
    class_counts = defaultdict(int)
    for ann in coco_format['annotations']:
        class_counts[ann['category_id']] += 1
    
    print(f"\n  Class distribution:")
    for cls_id in sorted(class_counts.keys()):
        print(f"    {CLASS_NAMES[cls_id]}: {class_counts[cls_id]}")
    
    return coco_format


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class RIVASwinDataset(Dataset):
    """Dataset for Swin Transformer training with optional imbalance handling."""
    
    def __init__(self, coco_json, image_dir, is_train=True, 
                 use_box_aug=True, use_copy_paste=False, use_mosaic=False):
        """
        Args:
            coco_json: Path to COCO format annotation JSON
            image_dir: Path to images directory
            is_train: Whether this is training set
            use_box_aug: Whether to use box augmentation
            use_copy_paste: Whether to use copy-paste augmentation
            use_mosaic: Whether to use mosaic augmentation
        """
        with open(coco_json, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_dir = image_dir
        self.is_train = is_train
        self.coco_json = coco_json
        
        # Build image_id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # Initialize augmenters
        if is_train:
            self.box_augmenter = BoxAugmentation(**BOX_AUG_CONFIG) if use_box_aug else None
            self.img_augmenter = ImageAugmentation(
                flip_h_prob=FLIP_PROB_H,
                flip_v_prob=FLIP_PROB_V,
                train_scales=TRAIN_SCALES
            )
            
            # Copy-paste augmentation
            self.copy_paste = None
            if use_copy_paste and USE_COPY_PASTE:
                self.copy_paste = CopyPasteAugmentation(**COPY_PASTE_CONFIG)
                self.copy_paste.build_cell_bank(coco_json, image_dir)
            
            # Mosaic augmentation
            self.mosaic = None
            if use_mosaic and USE_MOSAIC:
                self.mosaic = MosaicAugmentation(self, **MOSAIC_CONFIG)
        else:
            self.box_augmenter = None
            self.img_augmenter = None
            self.copy_paste = None
            self.mosaic = None
        
        # Mean and std for normalization
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
    
    def __len__(self):
        return len(self.coco_data['images'])
    
    def __getitem__(self, idx):
        img_info = self.coco_data['images'][idx]
        img_id = img_info['id']
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        anns = self.img_id_to_anns.get(img_id, [])
        
        if len(anns) > 0:
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
            
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        
        # Apply augmentations during training
        if self.is_train:
            # Mosaic augmentation (applied first)
            if self.mosaic is not None:
                image, boxes, labels = self.mosaic(idx, image, boxes, labels)
            
            # Copy-paste augmentation
            if self.copy_paste is not None:
                image, boxes, labels = self.copy_paste(image, boxes, labels)
            
            # Image augmentation
            if self.img_augmenter:
                image, boxes = self.img_augmenter(image, boxes)
            
            # Box augmentation
            if self.box_augmenter and len(boxes) > 0:
                boxes, labels = self.box_augmenter(boxes, labels, image.shape[:2])
        else:
            # Resize to fixed size for validation
            h, w = image.shape[:2]
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            if len(boxes) > 0:
                scale_x = IMG_SIZE / w
                scale_y = IMG_SIZE / h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
        
        # Normalize image
        image = (image.astype(np.float32) - self.mean) / self.std
        
        # Convert to tensor format
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long(),
            'image_id': torch.tensor([img_id])
        }
        
        return image, target


def collate_fn(batch):
    """Custom collate function for variable number of boxes"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


def get_class_aware_sampler(dataset):
    """Get class-aware sampler if enabled."""
    if USE_CLASS_AWARE_SAMPLING:
        return ClassAwareSampler(dataset, repeat_factor_thresh=CLASS_AWARE_REPEAT_FACTOR)
    return None


if __name__ == "__main__":
    from config import TRAIN_CSV, TRAIN_IMG_DIR, VAL_CSV, VAL_IMG_DIR, OUTPUT_DIR
    
    # Convert to COCO format
    convert_to_coco_format(
        TRAIN_CSV, TRAIN_IMG_DIR,
        os.path.join(OUTPUT_DIR, 'annotations_train.json'),
        split='train'
    )
    
    convert_to_coco_format(
        VAL_CSV, VAL_IMG_DIR,
        os.path.join(OUTPUT_DIR, 'annotations_val.json'),
        split='val'
    )
