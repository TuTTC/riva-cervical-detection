# Swin Transformer Large Baseline

## Overview
Swin Transformer Large backbone with Cascade R-CNN for cervical cancer cell detection in the RIVA challenge. Includes optional imbalance handling techniques.

## Architecture
- **Backbone**: Swin Transformer Large (embed_dims=192, depths=[2,2,18,2])
- **Detector**: Cascade R-CNN (3 stages)
- **Neck**: FPN (channels: [192, 384, 768, 1536] → 256)
- **Loss**: Focal Loss (with optional class weights)

## Files
```
swin_large/
├── config.py           # Configuration with all options
├── dataset.py          # Dataset + augmentations + samplers
├── model.py            # MMDetection config generator
├── train.py            # Training script
├── inference.py        # Inference + TTA + LightGBM
├── run.py              # Main entry point
└── requirements.txt    # Dependencies
```

## Installation
```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet
pip install ensemble-boxes scikit-image lightgbm
```

## Quick Start
```bash
# Show config
python run.py --mode config

# Full pipeline
python run.py --mode all

# Train only
python run.py --mode train

# Inference with TTA
python run.py --mode inference --tta
```

---

## Imbalance Handling Options

Edit `config.py` to enable/disable these techniques:

| Option                  | Variable                   | Default | Effect                          |
| ----------------------- | -------------------------- | ------- | ------------------------------- |
| Class-aware sampling    | `USE_CLASS_AWARE_SAMPLING` | False   | Oversample rare classes         |
| Copy-paste augmentation | `USE_COPY_PASTE`           | False   | Copy rare cells to other images |
| Mosaic augmentation     | `USE_MOSAIC`               | False   | Combine 4 images into 1         |
| Class-balanced weights  | `USE_CLASS_WEIGHTS`        | False   | Weight loss by class rarity     |
| MixUp augmentation      | `USE_MIXUP`                | True    | Blend images for regularization |

### How to Enable
```python
# In config.py, change:
USE_CLASS_AWARE_SAMPLING = True  # Oversample rare classes
USE_COPY_PASTE = True            # Copy-paste for LSIL/HSIL/ASCH/SCC
USE_CLASS_WEIGHTS = True         # Higher loss for rare classes
```

### Class Weights (configurable)
```python
CLASS_WEIGHTS = {
    0: 0.5,   # NILM (very common)
    1: 1.0,   # ENDO
    2: 0.7,   # INFL
    3: 1.2,   # ASCUS
    4: 2.0,   # LSIL (rare)
    5: 2.5,   # HSIL (rare)
    6: 3.0,   # ASCH (very rare)
    7: 4.0,   # SCC (very rare)
}
```

---

## Inference Options

| Mode              | Command                          | Improvement |
| ----------------- | -------------------------------- | ----------- |
| Basic             | `python run.py --mode inference` | Baseline    |
| TTA               | `--tta`                          | +1-3% mAP   |
| TTA + Multi-scale | `--tta --tta-scales 0.8 1.0 1.2` | +2-4% mAP   |
| LightGBM Fusion   | `--fusion`                       | +5-10% mAP  |
| All               | `--tta --fusion`                 | Best        |

---

## Expected Performance

| Configuration    | mAP@50    | Training Time (A6000) |
| ---------------- | --------- | --------------------- |
| Baseline         | 0.25-0.30 | ~3-4 hours            |
| + Class handling | 0.28-0.35 | ~4 hours              |
| + TTA            | 0.30-0.38 | -                     |
| + All options    | 0.35-0.42 | -                     |

## Notes
- Swin-Large requires ~16GB+ VRAM
- Gradient checkpointing auto-enabled for smaller GPUs
- Copy-paste builds a "cell bank" from rare classes on first run
