# =============================================================================
# TWO-STAGE PAP SMEAR - STAGE 2: EFFICIENTNET CLASSIFIER
# =============================================================================
# This notebook trains an EfficientNet-B3 classifier on cropped cell images.
# Key features: Class weighting, heavy augmentation, label smoothing
#
# Compatible with: Kaggle & Local
# =============================================================================

# %% [markdown]
# # Stage 2: Cell Classification with EfficientNet-B3
# 
# ## Key Features:
# - **Class-Weighted Loss**: Handle imbalanced data
# - **Heavy Augmentation**: Rotation, flip, color jitter
# - **Label Smoothing**: Prevent overconfidence
# - **Mixup**: Better generalization

# %%
# ===================== CONFIGURATION =====================
import os
import sys

IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA\two_stage_model'

CROP_DIR = os.path.join(BASE_DIR, 'cropped_cells')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'classifier_runs')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 32 if IS_KAGGLE else 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMG_SIZE = 128
NUM_CLASSES = 8
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

print(f"🔵 Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"📁 Crop directory: {CROP_DIR}")

# %%
# ===================== INSTALL DEPENDENCIES =====================
import subprocess

def install_if_needed(packages):
    for package in packages:
        try:
            __import__(package.split('==')[0].replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

install_if_needed(['timm', 'albumentations'])

# %%
# ===================== IMPORTS =====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Class names
CLASS_NAMES = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']

# %%
# ===================== CUSTOM DATASET WITH ALBUMENTATIONS =====================
class CellDataset(Dataset):
    """Custom dataset with Albumentations augmentation"""
    
    def __init__(self, root_dir, transform=None, is_train=True):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        
        # Load image with OpenCV (for albumentations)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

# %%
# ===================== AUGMENTATION PIPELINES =====================
# Heavy augmentation for training
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    
    # Geometric transforms
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
    
    # Color transforms
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
    ], p=0.7),
    
    # Noise and blur
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        A.GaussianBlur(blur_limit=(3, 5), p=1),
        A.MedianBlur(blur_limit=3, p=1),
    ], p=0.2),
    
    # Normalize and convert to tensor
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Simple transform for validation
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# %%
# ===================== DATA LOADERS =====================
print("📂 Loading datasets...")

train_dataset = CellDataset(os.path.join(CROP_DIR, 'train'), transform=train_transform, is_train=True)
val_dataset = CellDataset(os.path.join(CROP_DIR, 'val'), transform=val_transform, is_train=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Calculate class weights for imbalanced data
train_labels = [label for _, label in train_dataset.dataset.samples]
class_counts = Counter(train_labels)
total_samples = len(train_labels)

# Inverse frequency weighting
class_weights = []
for i in range(NUM_CLASSES):
    count = class_counts.get(i, 1)
    weight = total_samples / (NUM_CLASSES * count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\n📊 Class weights (for handling imbalance):")
for i, (name, weight) in enumerate(zip(CLASS_NAMES, class_weights)):
    count = class_counts.get(i, 0)
    print(f"   {name}: count={count}, weight={weight:.2f}")

# Data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
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

# %%
# ===================== MODEL =====================
print("\n🔧 Creating EfficientNet-B3 model...")

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %%
# ===================== LOSS AND OPTIMIZER =====================
# Cross-entropy with class weights and label smoothing
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

# AdamW optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# %%
# ===================== MIXUP FUNCTION =====================
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# %%
# ===================== TRAINING LOOP =====================
print("\n🚀 Starting training...")

best_val_acc = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    # ======= Training =======
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup with 50% probability
        use_mixup = np.random.random() < 0.5 and MIXUP_ALPHA > 0
        
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if not use_mixup:
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    scheduler.step()
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total if train_total > 0 else 0
    
    # ======= Validation =======
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    # Per-class accuracy tracking
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    
    # Save history
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
    print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # Per-class accuracy
    print('  Per-class accuracy:')
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f'    {name}: {acc:.4f} ({class_correct[i]}/{class_total[i]})')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(MODEL_SAVE_DIR, 'classifier_best.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': CLASS_NAMES,
        }, best_model_path)
        print(f'  ✅ Saved best model (val_acc: {val_acc:.4f})')

# %%
# ===================== SAVE FINAL MODEL =====================
print("\n📦 Saving final model...")

final_model_path = os.path.join(BASE_DIR, 'classifier_best.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': CLASS_NAMES,
    'img_size': IMG_SIZE,
}, final_model_path)

print(f"Model saved to: {final_model_path}")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# Save training history
history_path = os.path.join(MODEL_SAVE_DIR, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history, f)

print("\n🎉 Stage 2 Training Complete!")
print(f"📍 Classifier location: {final_model_path}")
