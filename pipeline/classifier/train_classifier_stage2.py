# =============================================================================
# BƯỚC 2: TRAIN CLASSIFIER CHUYÊN BIỆT CHO PHÂN LOẠI TẾ BÀO
# =============================================================================
# Sau khi Detection tìm ra vị trí tế bào, Classifier sẽ phân loại chính xác
# Đây là bước quan trọng nhất để cải thiện điểm từ 0.22 lên cao hơn
# =============================================================================

import os
import sys
import subprocess

# Install dependencies
print("📦 Installing dependencies...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                       'timm', 'albumentations', 'pytorch-metric-learning', '-q'])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision import models

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import timm

# ===================== CONFIGURATION =====================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'

MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models', 'classifier')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Class names
CLASS_NAMES = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
NUM_CLASSES = 8

# Training params
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          STAGE 2: CELL CLASSIFIER TRAINING                   ║
╠══════════════════════════════════════════════════════════════╣
║  Model: EfficientNet-B5 with ArcFace/Focal Loss              ║
║  Target: Classification accuracy > 80%                       ║
║  Strategy: Distinguish ASCUS vs LSIL vs HSIL                 ║
╚══════════════════════════════════════════════════════════════╝
""")


# ===================== DATASET =====================
class CellCropDataset(Dataset):
    """Dataset for cropped cell images"""
    
    def __init__(self, csv_path, img_dir, transform=None, is_train=True):
        """
        Args:
            csv_path: Path to annotations CSV (with columns: image_filename, x, y, width, height, class)
            img_dir: Directory containing original images
            transform: Albumentations or torchvision transforms
            is_train: Whether this is training set
        """
        self.df = pd.read_csv(csv_path, engine='python')
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        # Group annotations by image
        self.samples = []
        for _, row in self.df.iterrows():
            self.samples.append({
                'image_filename': row['image_filename'],
                'x': row['x'],
                'y': row['y'],
                'width': row['width'],
                'height': row['height'],
                'class': int(row['class'])
            })
        
        print(f"📊 Loaded {len(self.samples)} cell samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, sample['image_filename'])
        img = cv2.imread(img_path)
        if img is None:
            # Return a black image if file not found
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), sample['class']
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]
        
        # Extract crop with padding
        x_c, y_c = sample['x'], sample['y']
        w, h = sample['width'], sample['height']
        
        # Add 20% padding
        pad_ratio = 0.2
        pad_w = int(w * pad_ratio)
        pad_h = int(h * pad_ratio)
        
        x1 = max(0, int(x_c - w/2 - pad_w))
        y1 = max(0, int(y_c - h/2 - pad_h))
        x2 = min(w_img, int(x_c + w/2 + pad_w))
        y2 = min(h_img, int(y_c + h/2 + pad_h))
        
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            crop = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        
        # Apply transforms
        if self.transform:
            crop = self.transform(Image.fromarray(crop))
        else:
            crop = T.ToTensor()(crop)
        
        return crop, sample['class']


class CroppedCellDataset(Dataset):
    """Dataset for pre-cropped cell images (from SAHI output)"""
    
    def __init__(self, crop_dir, csv_path=None, transform=None):
        """
        Args:
            crop_dir: Directory containing cropped cell images
            csv_path: Optional CSV with crop metadata (crop_filename, predicted_class)
            transform: Transforms to apply
        """
        self.crop_dir = crop_dir
        self.transform = transform
        
        if csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.samples = list(self.df.to_dict('records'))
        else:
            # Just list all images
            self.samples = [
                {'crop_filename': f, 'predicted_class': -1}
                for f in os.listdir(crop_dir) if f.endswith(('.jpg', '.png'))
            ]
        
        print(f"📊 Loaded {len(self.samples)} cropped cells")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_path = os.path.join(self.crop_dir, sample['crop_filename'])
        img = cv2.imread(img_path)
        
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            img = self.transform(Image.fromarray(img))
        else:
            img = T.ToTensor()(img)
        
        return img, sample['crop_filename'], sample.get('predicted_class', -1)


# ===================== TRANSFORMS =====================
def get_train_transforms():
    """Augmentation for training - focus on color and morphology variation"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),  # Cells can be any orientation
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """Validation transforms - no augmentation"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ===================== FOCAL LOSS =====================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses learning on hard examples (ASCUS vs LSIL vs HSIL)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # Class weights
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ===================== MODEL =====================
def build_classifier(model_name='efficientnet_b5', pretrained=True):
    """Build classifier model using timm"""
    print(f"🏗️ Building {model_name}...")
    
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params / 1e6:.2f}M")
    
    return model


# ===================== TRAINING =====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == label:
                    class_correct[label] += 1
    
    # Print per-class accuracy
    print("\n📊 Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"   [{i}] {name:6s}: {acc:5.1f}% ({class_correct[i]}/{class_total[i]})")
    
    return total_loss / len(loader), 100 * correct / total


def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset"""
    labels = [sample['class'] for sample in dataset.samples]
    class_counts = Counter(labels)
    
    total = len(labels)
    weights = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 1)
        weight = total / (NUM_CLASSES * count)
        weights.append(weight)
    
    print(f"\n📊 Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ===================== EARLY STOPPING =====================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"⚠️ EarlyStopping: {self.counter}/{self.patience} (best: {self.best_score:.2f}%)")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


# ===================== OVERFITTING DETECTOR =====================
class OverfittingDetector:
    """Detect and handle overfitting by monitoring train-val gap"""
    def __init__(self, gap_threshold=15.0, patience=5):
        """
        Args:
            gap_threshold: Maximum allowed gap between train_acc and val_acc (%)
            patience: Number of epochs before taking action
        """
        self.gap_threshold = gap_threshold
        self.patience = patience
        self.overfit_counter = 0
        self.is_overfitting = False
        
    def __call__(self, train_acc, val_acc, optimizer):
        gap = train_acc - val_acc
        
        if gap > self.gap_threshold:
            self.overfit_counter += 1
            print(f"🔥 Overfitting detected! Gap: {gap:.1f}% (train={train_acc:.1f}%, val={val_acc:.1f}%)")
            print(f"   Counter: {self.overfit_counter}/{self.patience}")
            
            if self.overfit_counter >= self.patience:
                self.is_overfitting = True
                # Reduce learning rate by half
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * 0.5
                    print(f"   📉 Reduced LR: {old_lr:.6f} → {param_group['lr']:.6f}")
                self.overfit_counter = 0  # Reset counter after action
        else:
            self.overfit_counter = max(0, self.overfit_counter - 1)  # Decay counter
            
        return self.is_overfitting


def train_classifier():
    """Main training function with Early Stopping"""
    print("\n🚀 Starting classifier training...")
    
    # Paths
    train_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv')
    val_csv = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'val.csv')
    train_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'train')
    val_img_dir = os.path.join(INPUT_DIR, 'images', 'images', 'val')
    
    # Datasets
    train_dataset = CellCropDataset(train_csv, train_img_dir, get_train_transforms(), is_train=True)
    val_dataset = CellCropDataset(val_csv, val_img_dir, get_val_transforms(), is_train=False)
    
    # Compute class weights for focal loss
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    
    # Weighted sampler for balanced batches
    sample_weights = [class_weights[sample['class']].item() for sample in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=4 if IS_KAGGLE else 2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4 if IS_KAGGLE else 2,
        pin_memory=True
    )
    
    # Model
    model = build_classifier('efficientnet_b5').to(DEVICE)
    
    # Loss with focal loss
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.1, mode='max')
    
    # Overfitting Detector
    overfit_detector = OverfittingDetector(gap_threshold=15.0, patience=5)
    
    # Training loop
    best_val_acc = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_classifier.pth')
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        print(f"\n📈 Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"📈 Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"📈 LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"📈 Gap: {train_acc - val_acc:.1f}%")
        
        # Check for overfitting
        overfit_detector(train_acc, val_acc, optimizer)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            print(f"💾 Saved best model (acc: {val_acc:.2f}%)")
        
        # Check early stopping
        if early_stopping(val_acc):
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}!")
            break
    
    print(f"\n✅ Training complete!")
    print(f"📁 Best model: {best_model_path}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    return best_model_path


if __name__ == "__main__":
    best_model = train_classifier()
    
    print("\n📝 Next steps:")
    print("   1. Run classifier on test crops")
    print("   2. Fuse detection + classification scores")
    print("   3. Submit final results")
