# # =============================================================================
# # FASTER R-CNN PAPER CONFIG
# # =============================================================================
# # Configuration settings from "Holistic and Historical Instance Comparison" paper
# # =============================================================================

# import os

# # =============================================================================
# # PATHS
# # =============================================================================
# IS_KAGGLE = os.path.exists('/kaggle/working')

# if IS_KAGGLE:
#     BASE_DIR = '/kaggle/working'
#     INPUT_DIR = '/kaggle/input/riva-track-a'
# else:
#     BASE_DIR = r'd:\NĂM 3\RIVA'
#     INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'

# OUTPUT_DIR = os.path.join(BASE_DIR, 'faster_rcnn_paper')
# WORK_DIR = os.path.join(OUTPUT_DIR, 'work_dirs')

# # Create directories
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(WORK_DIR, exist_ok=True)

# # =============================================================================
# # DATASET CONFIG
# # =============================================================================
# CLASS_NAMES = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
# NUM_CLASSES = 8

# # Data paths
# TRAIN_CSV = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv')
# VAL_CSV = os.path.join(INPUT_DIR, 'annotations', 'annotations', 'val.csv')
# TRAIN_IMG_DIR = os.path.join(INPUT_DIR, 'images', 'images', 'train')
# VAL_IMG_DIR = os.path.join(INPUT_DIR, 'images', 'images', 'val')
# TEST_IMG_DIR = os.path.join(INPUT_DIR, 'images', 'images', 'test')

# # =============================================================================
# # MODEL CONFIG (Paper Specifications)
# # =============================================================================
# # Backbone: ResNet-50 + FPN
# BACKBONE = 'resnet50'
# USE_FPN = True

# # Input size: Paper uses 1200x1200
# IMG_SIZE = 1200

# # =============================================================================
# # GPU AUTO-DETECTION
# # =============================================================================
# import torch
# if torch.cuda.is_available():
#     GPU_NAME = torch.cuda.get_device_name(0)
#     GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
# else:
#     GPU_NAME = "CPU"
#     GPU_MEMORY = 0

# # =============================================================================
# # TRAINING CONFIG - AUTO-OPTIMIZED FOR GPU
# # =============================================================================
# # Batch size based on GPU VRAM:
# #   - A6000 48GB: batch 8-12
# #   - RTX 3090 24GB: batch 4
# #   - RTX 4090 24GB: batch 4
# #   - T4 16GB: batch 2

# if GPU_MEMORY >= 45:      # A6000, A100
#     BATCH_SIZE = 10       # Can push to 12 if needed
#     NUM_WORKERS = 8
#     ACCUMULATE_STEPS = 1  # No gradient accumulation needed
# elif GPU_MEMORY >= 20:    # RTX 3090, 4090
#     BATCH_SIZE = 4
#     NUM_WORKERS = 4
#     ACCUMULATE_STEPS = 2
# elif GPU_MEMORY >= 14:    # T4, RTX 3060
#     BATCH_SIZE = 2
#     NUM_WORKERS = 4
#     ACCUMULATE_STEPS = 4
# else:                     # Smaller GPUs
#     BATCH_SIZE = 1
#     NUM_WORKERS = 2
#     ACCUMULATE_STEPS = 8

# # Total epochs: Paper uses 24
# EPOCHS = 24

# # Optimizer: SGD with momentum
# OPTIMIZER = 'sgd'
# BASE_LR = 0.005          # Paper: Initial LR = 0.005
# MOMENTUM = 0.9           # Paper: momentum = 0.9
# WEIGHT_DECAY = 0.0001

# # Scale LR with batch size (linear scaling rule)
# # Paper uses batch 4, so scale LR proportionally
# EFFECTIVE_LR = BASE_LR * (BATCH_SIZE / 4)

# # Learning rate schedule: Decay at epoch 8 and 14
# LR_DECAY_EPOCHS = [8, 14]
# LR_DECAY_GAMMA = 0.1     # Decay by factor 0.1

# # Gradient clipping
# GRAD_CLIP_NORM = 35.0

# # =============================================================================
# # METHOD-SPECIFIC HYPERPARAMETERS (from ablation studies)
# # =============================================================================
# TEMPERATURE = 6          # τ = 6 (contrastive loss temperature)
# MEMORY_BANK_SIZE = 80    # Q = 80 per class
# LAMBDA_ROI = 0.2         # λ_roi = 0.2 (RoI comparison loss weight)
# LAMBDA_CLS = 0.2         # λ_cls = 0.2 (class comparison loss weight)
# ROI_BATCH_SIZE = 256     # K = 256 RoI candidates

# # =============================================================================
# # INFERENCE CONFIG
# # =============================================================================
# CONF_THRESHOLD = 0.001   # Low threshold to keep more predictions
# NMS_THRESHOLD = 0.5
# MAX_DETECTIONS = 300

# # =============================================================================
# # CHECKPOINT CONFIG
# # =============================================================================
# SAVE_INTERVAL = 4        # Save checkpoint every N epochs
# PRETRAINED_WEIGHTS = 'DEFAULT'  # Use ImageNet pretrained

# # =============================================================================
# # PRINT CONFIG
# # =============================================================================
# def print_config():
#     """Print current configuration"""
#     print(f"""
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║         FASTER R-CNN - PAPER REPRODUCTION CONFIG                         ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  Backbone: {BACKBONE.upper()} + FPN                                              ║
# ║  Input Size: {IMG_SIZE}x{IMG_SIZE} pixels                                            ║
# ║  Batch Size: {BATCH_SIZE}                                                            ║
# ║  Epochs: {EPOCHS}                                                                ║
# ║  Optimizer: SGD (LR={BASE_LR}, momentum={MOMENTUM})                               ║
# ║  LR Decay: at epochs {LR_DECAY_EPOCHS} (gamma={LR_DECAY_GAMMA})                              ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  Temperature (τ): {TEMPERATURE}                                                        ║
# ║  Memory Bank (Q): {MEMORY_BANK_SIZE} per class                                               ║
# ║  Loss Weights: λ_roi={LAMBDA_ROI}, λ_cls={LAMBDA_CLS}                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# """)


# if __name__ == "__main__":
#     print_config()


# =============================================================================
# FASTER R-CNN PAPER CONFIG
# =============================================================================
# Configuration settings from "Holistic and Historical Instance Comparison" paper
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
    # BASE_DIR = r'/home/combine2t/CamTu/RIVA-TRACK-A/frcnn_paper'
    # INPUT_DIR = r'/home/combine2t/CamTu/RIVA-TRACK-A/frcnn_paper/data'
    BASE_DIR = r'd:\NĂM 3\RIVA'
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'

OUTPUT_DIR = os.path.join(BASE_DIR, 'faster_rcnn_paper')
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
# MODEL CONFIG (Paper Specifications)
# =============================================================================
# Backbone: ResNet-50 + FPN
BACKBONE = 'resnet50'
USE_FPN = True

# Input size: Paper uses 1200x1200
IMG_SIZE = 1200

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
# Batch size based on GPU VRAM:
#   - A6000 48GB: batch 8-12
#   - RTX 3090 24GB: batch 4
#   - RTX 4090 24GB: batch 4
#   - T4 16GB: batch 2

if GPU_MEMORY >= 45:      # A6000, A100
    BATCH_SIZE = 10       # Can push to 12 if needed
    NUM_WORKERS = 8
    ACCUMULATE_STEPS = 1  # No gradient accumulation needed
elif GPU_MEMORY >= 20:    # RTX 3090, 4090
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    ACCUMULATE_STEPS = 2
elif GPU_MEMORY >= 14:    # T4, RTX 3060
    BATCH_SIZE = 2
    NUM_WORKERS = 4
    ACCUMULATE_STEPS = 4
else:                     # Smaller GPUs
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    ACCUMULATE_STEPS = 8

# Total epochs: Paper uses 24
EPOCHS = 24

# Optimizer: SGD with momentum
OPTIMIZER = 'sgd'
BASE_LR = 0.005          # Paper: Initial LR = 0.005
MOMENTUM = 0.9           # Paper: momentum = 0.9
WEIGHT_DECAY = 0.0001

# Scale LR with batch size (linear scaling rule)
# Paper uses batch 4, so scale LR proportionally
EFFECTIVE_LR = BASE_LR * (BATCH_SIZE / 4)

# Learning rate schedule: Decay at epoch 8 and 14
LR_DECAY_EPOCHS = [8, 14]
LR_DECAY_GAMMA = 0.1     # Decay by factor 0.1

# Gradient clipping
GRAD_CLIP_NORM = 35.0

# =============================================================================
# METHOD-SPECIFIC HYPERPARAMETERS (from ablation studies)
# =============================================================================
TEMPERATURE = 6          # τ = 6 (contrastive loss temperature)
MEMORY_BANK_SIZE = 80    # Q = 80 per class
LAMBDA_ROI = 0.2         # λ_roi = 0.2 (RoI comparison loss weight)
LAMBDA_CLS = 0.2         # λ_cls = 0.2 (class comparison loss weight)
ROI_BATCH_SIZE = 256     # K = 256 RoI candidates

# =============================================================================
# INFERENCE CONFIG
# =============================================================================
CONF_THRESHOLD = 0.001   # Low threshold to keep more predictions
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 300

# =============================================================================
# CHECKPOINT CONFIG
# =============================================================================
SAVE_INTERVAL = 4        # Save checkpoint every N epochs
PRETRAINED_WEIGHTS = 'DEFAULT'  # Use ImageNet pretrained

# =============================================================================
# PRINT CONFIG
# =============================================================================
def print_config():
    """Print current configuration"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║         FASTER R-CNN - PAPER REPRODUCTION CONFIG                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  GPU: {GPU_NAME[:40]:<40} ║
║  VRAM: {GPU_MEMORY:.1f} GB                                                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Backbone: {BACKBONE.upper()} + FPN                                              ║
║  Input Size: {IMG_SIZE}x{IMG_SIZE} pixels                                            ║
║  Batch Size: {BATCH_SIZE:<2}                                                          ║
║  Num Workers: {NUM_WORKERS}                                                          ║
║  Epochs: {EPOCHS}                                                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Optimizer: SGD                                                          ║
║  Base LR: {BASE_LR}                                                          ║
║  Effective LR: {EFFECTIVE_LR:.4f} (scaled for batch {BATCH_SIZE})                        ║
║  Momentum: {MOMENTUM}                                                            ║
║  LR Decay: at epochs {LR_DECAY_EPOCHS} (gamma={LR_DECAY_GAMMA})                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Temperature (τ): {TEMPERATURE}                                                        ║
║  Memory Bank (Q): {MEMORY_BANK_SIZE} per class                                               ║
║  Loss Weights: λ_roi={LAMBDA_ROI}, λ_cls={LAMBDA_CLS}                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_config()

