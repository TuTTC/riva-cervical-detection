# =============================================================================
# SWIN TRANSFORMER LARGE CONFIG
# =============================================================================
# Baseline configuration for Swin Transformer Large backbone
# With optional imbalance handling techniques
# =============================================================================

import os

# =============================================================================
# PATHS
# =============================================================================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input/riva-track-a'
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # RIVA-TRACK-A/
    INPUT_DIR = os.path.join(BASE_DIR, 'data', 'riva-cervical-cytology-challenge-isbi-2026')

OUTPUT_DIR = os.path.join(BASE_DIR, 'swin_large_output')
WORK_DIR = os.path.join(OUTPUT_DIR, 'work_dirs')

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# DATASET CONFIG
# =============================================================================
CLASS_NAMES = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
NUM_CLASSES = 8

# Data paths
TRAIN_CSV = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv')
VAL_CSV = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'val.csv')
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, 'images', 'images', 'train')
VAL_IMG_DIR = os.path.join(INPUT_DIR, 'images', 'images', 'val')
TEST_IMG_DIR = os.path.join(INPUT_DIR, 'images', 'images', 'test')

# =============================================================================
# SWIN TRANSFORMER LARGE MODEL CONFIG
# =============================================================================
MODEL_TYPE = 'swin_large'
BACKBONE = 'swin_large'

SWIN_CONFIG = {
    'embed_dims': 192,
    'depths': [2, 2, 18, 2],
    'num_heads': [6, 12, 24, 48],
    'window_size': 7,
    'mlp_ratio': 4,
    'qkv_bias': True,
    'qk_scale': None,
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.3,
    'patch_norm': True,
    'out_indices': (0, 1, 2, 3),
    'with_cp': False,
    'convert_weights': True,
}

FPN_IN_CHANNELS = [192, 384, 768, 1536]
FPN_OUT_CHANNELS = 256

PRETRAINED_CHECKPOINT = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'

IMG_SIZE = 1280

# =============================================================================
# GPU AUTO-DETECTION
# =============================================================================
import torch
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    GPU_NAME = "CPU"
    GPU_MEMORY = 0

# =============================================================================
# TRAINING CONFIG - AUTO-OPTIMIZED FOR GPU
# =============================================================================
if GPU_MEMORY >= 70:      # A100 80GB
    BATCH_SIZE = 4
    NUM_WORKERS = 8
    ACCUMULATE_STEPS = 2
    USE_CHECKPOINT = False
elif GPU_MEMORY >= 45:    # A6000 48GB
    BATCH_SIZE = 2
    NUM_WORKERS = 6
    ACCUMULATE_STEPS = 4
    USE_CHECKPOINT = False
elif GPU_MEMORY >= 20:    # RTX 3090, 4090
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    ACCUMULATE_STEPS = 8
    USE_CHECKPOINT = True
elif GPU_MEMORY >= 14:    # T4
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    ACCUMULATE_STEPS = 8
    USE_CHECKPOINT = True
else:                     # Smaller GPUs
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    ACCUMULATE_STEPS = 16
    USE_CHECKPOINT = True

EPOCHS = 48
OPTIMIZER = 'adamw'
BASE_LR = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.05
EFFECTIVE_LR = BASE_LR * (BATCH_SIZE * ACCUMULATE_STEPS / 8)
WARMUP_EPOCHS = 3
WARMUP_RATIO = 0.001
MIN_LR = 1e-6
GRAD_CLIP_NORM = 35.0

# =============================================================================
# IMBALANCE HANDLING OPTIONS (NEW)
# =============================================================================
# Enable/disable each technique as needed

# 1. Class-aware sampling: Oversample rare classes
USE_CLASS_AWARE_SAMPLING = False
CLASS_AWARE_REPEAT_FACTOR = 0.5  # Threshold for repeat factor sampling

# 2. Copy-paste augmentation: Copy rare cell instances to other images
USE_COPY_PASTE = False
COPY_PASTE_CONFIG = {
    'prob': 0.5,                    # Probability of applying copy-paste
    'max_paste_per_image': 5,       # Max cells to paste per image
    'target_classes': [4, 5, 6, 7], # Rare classes: LSIL, HSIL, ASCH, SCC
    'scale_range': (0.8, 1.2),      # Scale variation for pasted cells
}

# 3. Mosaic augmentation: Combine 4 images into one
USE_MOSAIC = False
MOSAIC_CONFIG = {
    'prob': 0.5,
    'img_scale': (1280, 1280),
    'center_ratio_range': (0.5, 1.5),
    'pad_val': 114.0,
}

# 4. Class-balanced loss weights
USE_CLASS_WEIGHTS = True
# Weights inversely proportional to class frequency (approximate)
# Higher weight = more penalty for misclassifying rare classes
CLASS_WEIGHTS = {
    0: 0.5,   # NILM (very common)
    1: 1.0,   # ENDO
    2: 0.7,   # INFL (common)
    3: 1.2,   # ASCUS
    4: 2.0,   # LSIL (rare)
    5: 2.5,   # HSIL (rare)
    6: 3.0,   # ASCH (very rare)
    7: 4.0,   # SCC (very rare)
}

# 5. MixUp augmentation
USE_MIXUP = True  # Already enabled in base config
MIXUP_CONFIG = {
    'alpha': 1.0,
    'prob': 0.3,
}

# =============================================================================
# STANDARD AUGMENTATION CONFIG
# =============================================================================
TRAIN_SCALES = [(1024, 1024), (1152, 1152), (1280, 1280), (1408, 1408)]

BOX_AUG_CONFIG = {
    'num_augments': 3,
    'max_shift_ratio': 0.1,
    'max_scale_ratio': 0.1,
    'iou_threshold': 0.7
}

FLIP_PROB_H = 0.5
FLIP_PROB_V = 0.5
PHOTOMETRIC_DISTORTION = True
RANDOM_CROP = True
RANDOM_AFFINE = True
MIXUP = USE_MIXUP

# =============================================================================
# INFERENCE CONFIG
# =============================================================================
CONF_THRESHOLD = 0.001
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 300

# =============================================================================
# CHECKPOINT CONFIG
# =============================================================================
SAVE_INTERVAL = 4
SAVE_BEST = True

# =============================================================================
# EMA CONFIG
# =============================================================================
USE_EMA = True
EMA_MOMENTUM = 0.0002

# =============================================================================
# PRINT CONFIG
# =============================================================================
def print_config():
    """Print current configuration"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║              SWIN TRANSFORMER LARGE - BASELINE CONFIG                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  GPU: {GPU_NAME[:40]:<40} ║
║  VRAM: {GPU_MEMORY:.1f} GB                                                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Backbone: SWIN TRANSFORMER LARGE                                        ║
║    - Embed dims: {SWIN_CONFIG['embed_dims']}                                                      ║
║    - Depths: {SWIN_CONFIG['depths']}                                                ║
║    - Num heads: {SWIN_CONFIG['num_heads']}                                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Input Size: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}                          ║
║  LR: {BASE_LR} | Weight Decay: {WEIGHT_DECAY}                                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  IMBALANCE HANDLING OPTIONS:                                             ║
║    - Class-aware sampling: {str(USE_CLASS_AWARE_SAMPLING):<5}                                      ║
║    - Copy-paste augmentation: {str(USE_COPY_PASTE):<5}                                   ║
║    - Mosaic augmentation: {str(USE_MOSAIC):<5}                                       ║
║    - Class-balanced weights: {str(USE_CLASS_WEIGHTS):<5}                                    ║
║    - MixUp augmentation: {str(USE_MIXUP):<5}                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_config()
