# Faster R-CNN Paper Reproduction

Implementation following **"Holistic and Historical Instance Comparison"** paper specifications.

## 📁 File Structure

```
frcnn_paper/
├── config.py      # All hyperparameters and paths
├── dataset.py     # Data loading, Box Augmentation, preprocessing
├── model.py       # Model creation and utilities
├── train.py       # Training loop
├── inference.py   # Inference and submission
├── run.py         # Main entry point
└── README.md      # This file
```

## 🔧 Configuration (Paper Specifications)

| Parameter    | Value               | Source             |
| ------------ | ------------------- | ------------------ |
| Input Size   | 1200×1200           | Paper              |
| Backbone     | ResNet-50 + FPN     | Paper              |
| Optimizer    | SGD                 | Paper              |
| Initial LR   | 0.005               | Paper              |
| Momentum     | 0.9                 | Paper              |
| LR Decay     | ×0.1 at epoch 8, 14 | Paper              |
| Total Epochs | 24                  | Paper              |
| Batch Size   | 4                   | RTX 3090 optimized |

### Method-Specific Hyperparameters
| Parameter       | Value | Description                  |
| --------------- | ----- | ---------------------------- |
| τ (Temperature) | 6     | Contrastive loss temperature |
| Q (Memory Bank) | 80    | Queue size per class         |
| λ_roi           | 0.2   | RoI comparison loss weight   |
| λ_cls           | 0.2   | Class comparison loss weight |
| K (RoI Batch)   | 256   | RoI candidates per image     |

## 📦 Box Augmentation (Paper Method)

Implemented following **paper formula: b' = b + δ**

```python
# Box Augmentation generates variations of GT boxes
# to enrich positive samples for training

BoxAugmentation(
    num_augments=5,         # 5 augmented boxes per GT
    max_shift_ratio=0.1,    # ±10% shift
    max_scale_ratio=0.1,    # ±10% scale
    iou_threshold=0.7       # Min IoU with original
)
```

**Effect:**
- Original: 10 boxes → With augmentation: 60 boxes (6x)
- Better positive sample diversity
- Improved detection robustness

### Image Augmentations
- Horizontal flip (50%)
- Vertical flip (50%)
- Color jitter (brightness, contrast)

## 🚀 Quick Start

### Option 1: Full Pipeline
```bash
python run.py --mode all
```

### Option 2: Train Only
```bash
python run.py --mode train
```

### Option 3: Inference Only
```bash
python run.py --mode inference --checkpoint path/to/checkpoint.pth
```

## 📊 Expected Results

- **Training Time**: ~3-4 hours on RTX 3090
- **VRAM Usage**: ~20GB with batch_size=4 at 1200×1200
- **Expected mAP**: 0.25-0.35

## 🔄 Pipeline Flow

```
1. Data Loading (dataset.py)
   └─> CSV → PyTorch DataLoader
   └─> Box Augmentation: b' = b + δ
   └─> Image Augmentation: flip, color jitter
   
2. Model Setup (model.py)
   └─> ResNet-50-FPN + FastRCNN Head (8 classes)
   
3. Training (train.py)
   └─> 24 epochs, SGD, LR decay at [8, 14]
   
4. Inference (inference.py)
   └─> Test images → Predictions → submission.csv
```

## ⚙️ Modifying Configuration

Edit `config.py` to change:
- Image size
- Batch size
- Learning rate
- Number of epochs
- Paths

## 📝 Output Files

```
faster_rcnn_paper/
├── work_dirs/
│   ├── epoch_4.pth
│   ├── epoch_8.pth
│   ├── ...
│   └── best.pth
├── faster_rcnn_final.pth
└── annotations/
    ├── train.json
    └── val.json
    
submission_frcnn_paper.csv  (in BASE_DIR)
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# In config.py, reduce:
BATCH_SIZE = 2  # or 1
IMG_SIZE = 800  # smaller size
```

### Disable Augmentation
```python
# In dataset.py get_data_loaders():
get_data_loaders(
    use_box_augmentation=False,
    use_image_augmentation=False
)
```

## 📚 References

- Paper: "Holistic and Historical Instance Comparison for Cervical Cell Detection"
- Framework: PyTorch + torchvision
- Backbone: ResNet-50 with FPN (pretrained on ImageNet)
