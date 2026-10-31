# =============================================================================
# SWIN TRANSFORMER LARGE - TRAINING SCRIPT
# =============================================================================
# Training pipeline with optional imbalance handling
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OUTPUT_DIR, WORK_DIR, CLASS_NAMES, NUM_CLASSES,
    TRAIN_CSV, VAL_CSV, TRAIN_IMG_DIR, VAL_IMG_DIR,
    BATCH_SIZE, NUM_WORKERS, EPOCHS,
    BASE_LR, WEIGHT_DECAY, WARMUP_EPOCHS,
    USE_CHECKPOINT, USE_EMA, EMA_MOMENTUM,
    SAVE_INTERVAL, GRAD_CLIP_NORM,
    USE_CLASS_AWARE_SAMPLING, USE_COPY_PASTE, USE_MOSAIC, USE_CLASS_WEIGHTS,
    print_config
)
from dataset import convert_to_coco_format


def setup_environment():
    """Setup training environment"""
    print("=" * 80)
    print("SWIN TRANSFORMER LARGE - TRAINING SETUP")
    print("=" * 80)
    
    print_config()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    
    # Print imbalance handling status
    print("\nImbalance handling:")
    print(f"  - Class-aware sampling: {USE_CLASS_AWARE_SAMPLING}")
    print(f"  - Copy-paste augmentation: {USE_COPY_PASTE}")
    print(f"  - Mosaic augmentation: {USE_MOSAIC}")
    print(f"  - Class-balanced weights: {USE_CLASS_WEIGHTS}")


def prepare_data():
    """Convert CSV to COCO format"""
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING DATA")
    print("=" * 80)
    
    train_json = os.path.join(OUTPUT_DIR, 'annotations_train.json')
    val_json = os.path.join(OUTPUT_DIR, 'annotations_val.json')
    
    if os.path.exists(train_json) and os.path.exists(val_json):
        print("COCO annotations already exist, skipping conversion...")
    else:
        convert_to_coco_format(TRAIN_CSV, TRAIN_IMG_DIR, train_json, split='train')
        convert_to_coco_format(VAL_CSV, VAL_IMG_DIR, val_json, split='val')
    
    return train_json, val_json


def prepare_config(train_json, val_json):
    """Generate MMDetection config"""
    print("\n" + "=" * 80)
    print("STEP 2: GENERATING CONFIG")
    print("=" * 80)
    
    from model import save_mmdet_config
    
    config_path = os.path.join(OUTPUT_DIR, 'swin_large_config.py')
    
    save_mmdet_config(
        output_path=config_path,
        work_dir=OUTPUT_DIR,
        train_json=os.path.basename(train_json),
        val_json=os.path.basename(val_json),
        train_img_prefix=TRAIN_IMG_DIR,
        val_img_prefix=VAL_IMG_DIR,
        class_names=CLASS_NAMES
    )
    
    return config_path


def train_model(config_path):
    """Train Swin Transformer Large model"""
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING MODEL")
    print("=" * 80)
    
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        
        print("Loading config...")
        cfg = Config.fromfile(config_path)
        
        cfg.work_dir = WORK_DIR
        cfg.train_dataloader.batch_size = BATCH_SIZE
        cfg.train_dataloader.num_workers = NUM_WORKERS
        cfg.val_dataloader.batch_size = BATCH_SIZE
        cfg.val_dataloader.num_workers = NUM_WORKERS
        
        cfg.optim_wrapper.optimizer.lr = BASE_LR
        cfg.optim_wrapper.optimizer.weight_decay = WEIGHT_DECAY
        cfg.optim_wrapper.clip_grad.max_norm = GRAD_CLIP_NORM
        
        cfg.train_cfg.max_epochs = EPOCHS
        cfg.default_hooks.checkpoint.interval = SAVE_INTERVAL
        
        if USE_CHECKPOINT:
            cfg.model.backbone.with_cp = True
            print("Gradient checkpointing: ENABLED")
        
        if USE_EMA:
            cfg.custom_hooks = [
                dict(type='EMAHook', ema_type='ExpMomentumEMA', 
                     momentum=EMA_MOMENTUM, priority='ABOVE_NORMAL')
            ]
            print(f"EMA: ENABLED (momentum={EMA_MOMENTUM})")
        
        # Note: Class-aware sampling is handled in dataset.py
        # For full integration, you would need to modify the MMDetection sampler
        if USE_CLASS_AWARE_SAMPLING:
            print("Class-aware sampling: ENABLED (via custom dataset)")
        
        print(f"\nStarting training...")
        print(f"  - Epochs: {EPOCHS}")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Learning rate: {BASE_LR}")
        print("")
        
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Model saved to: {WORK_DIR}")
        
        checkpoints = sorted([f for f in os.listdir(WORK_DIR) if f.endswith('.pth')])
        if checkpoints:
            best_checkpoint = os.path.join(WORK_DIR, checkpoints[-1])
            print(f"Latest checkpoint: {best_checkpoint}")
            return best_checkpoint
        
        return None
        
    except ImportError as e:
        print(f"\nError: MMDetection not installed!")
        print("Please install with:")
        print("  pip install -U openmim")
        print("  mim install mmengine")
        print("  mim install 'mmcv>=2.0.0'")
        print("  mim install mmdet")
        raise e


def main():
    """Main training pipeline"""
    setup_environment()
    train_json, val_json = prepare_data()
    config_path = prepare_config(train_json, val_json)
    checkpoint = train_model(config_path)
    
    if checkpoint:
        print(f"\nTraining successful!")
        print(f"Use the checkpoint for inference: {checkpoint}")
    
    return checkpoint


if __name__ == "__main__":
    main()
