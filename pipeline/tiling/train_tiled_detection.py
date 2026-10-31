# =============================================================================
# BƯỚC 1B: TRAIN DETECTION TRÊN TILED DATASET
# =============================================================================
# Train Cascade R-CNN Swin-Large hoặc YOLOv11x trên tiled dataset
# Mục tiêu: Detection mAP 0.55+ (từ 0.46)
# =============================================================================

import os
import sys
import subprocess

# ===================== CONFIGURATION =====================
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    TILED_DATASET = '/kaggle/working/tiled_dataset'
else:
    BASE_DIR = r'd:\NĂM 3\RIVA\models\tiled_detection'
    TILED_DATASET = r'd:\NĂM 3\RIVA\tiled_dataset'

os.makedirs(BASE_DIR, exist_ok=True)
os.environ['WANDB_MODE'] = 'disabled'

# ===================== INSTALL DEPENDENCIES =====================
print("📦 Installing dependencies...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                       'ultralytics', 'sahi', '-q'])

from ultralytics import YOLO
import yaml

print(f"""
╔══════════════════════════════════════════════════════════════╗
║       TRAIN HIGH-RESOLUTION DETECTION ON TILED DATA          ║
╠══════════════════════════════════════════════════════════════╣
║  Strategy: Train on 1024x1024 tiles at native resolution     ║
║  Model: YOLOv11x (largest) for maximum accuracy              ║
║  Target: Detection mAP 0.55+                                 ║
╚══════════════════════════════════════════════════════════════╝
""")


def verify_dataset():
    """Verify tiled dataset exists and is valid"""
    yaml_path = os.path.join(TILED_DATASET, 'data.yaml')
    
    if not os.path.exists(yaml_path):
        print(f"❌ Dataset not found: {yaml_path}")
        print("   Run offline_tiling.py first!")
        return None
    
    # Load and verify
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_dir = os.path.join(TILED_DATASET, 'images', 'train')
    val_dir = os.path.join(TILED_DATASET, 'images', 'val')
    
    n_train = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))])
    n_val = len([f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"✅ Dataset verified:")
    print(f"   Train tiles: {n_train}")
    print(f"   Val tiles: {n_val}")
    print(f"   Classes: {config['nc']} ({', '.join(config['names'])})")
    
    return yaml_path


def train_yolo_tiled(yaml_path, model_size='l', epochs=100):
    """
    Train YOLOv11 on tiled dataset
    
    MEMORY-OPTIMIZED for Kaggle T4 (16GB) / P100 (16GB)
    """
    print(f"\n🚀 Training YOLOv11{model_size} on tiled data...")
    
    # ==================== GPU PROFILES ====================
    # Uncomment the profile that matches your GPU
    
    # === KAGGLE T4/P100 (16GB) - SAFE ===
    IMGSZ = 640       # Reduced from 1024
    BATCH = 4         # Reduced from 8
    MODEL_SIZE = 'l'  # Use 'l' instead of 'x'
    
    # === KAGGLE T4x2 (32GB total) ===
    # IMGSZ = 768
    # BATCH = 6
    # MODEL_SIZE = 'l'
    
    # === COLAB A100 (40GB) / VAST.AI ===
    # IMGSZ = 1024
    # BATCH = 8
    # MODEL_SIZE = 'x'
    
    # ======================================================
    
    model = YOLO(f'yolo11{MODEL_SIZE}.pt')
    
    # Training config optimized for limited VRAM
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=30,
        project=os.path.join(BASE_DIR, 'runs'),
        name=f'yolo11{MODEL_SIZE}_tiled',
        exist_ok=True,
        
        # === MEMORY OPTIMIZATION ===
        cache=False,      # Don't cache images in RAM
        amp=True,         # Mixed precision (saves ~30% VRAM)
        
        # Optimized augmentation for tiled data
        augment=True,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15,
        translate=0.1,
        scale=0.2,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,       # DISABLE - breaks tile structure
        mixup=0.0,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        
        # Loss
        box=7.5,
        cls=1.0,
        dfl=1.5,
        
        # Other
        workers=2,        # Reduced workers
        verbose=True,
        plots=True,
        save_period=10,
    )
    
    print(f"\n💾 Memory used: Check nvidia-smi")
    
    best_path = os.path.join(BASE_DIR, 'runs', f'yolo11{model_size}_tiled', 'weights', 'best.pt')
    return best_path, results


def train_yolo_with_focal_loss(yaml_path, model_size='x', epochs=100):
    """
    Alternative: Train with class-weighted focal loss for imbalanced classes
    """
    print(f"\n🚀 Training YOLOv11{model_size} with Focal Loss for class imbalance...")
    
    model = YOLO(f'yolo11{model_size}.pt')
    
    # Class weights based on inverse frequency
    # [NILM, ENDO, INFL, ASCUS, LSIL, HSIL, ASCH, SCC]
    # Rare classes (HSIL, ASCH, SCC) get higher weights
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=1024,
        batch=8 if IS_KAGGLE else 4,
        patience=30,
        project=os.path.join(BASE_DIR, 'runs'),
        name=f'yolo11{model_size}_focal',
        exist_ok=True,
        
        # Focal loss is enabled by default in recent YOLO versions
        # Increase cls loss weight to emphasize classification
        cls=2.0,
        
        # Same augmentation as above
        augment=True,
        mosaic=0.0,
        mixup=0.0,
        
        optimizer='AdamW',
        lr0=0.001,
        workers=4 if IS_KAGGLE else 2,
    )
    
    best_path = os.path.join(BASE_DIR, 'runs', f'yolo11{model_size}_focal', 'weights', 'best.pt')
    return best_path, results


def main():
    print("\n🚀 Starting Tiled Detection Training...")
    
    # Step 1: Verify dataset
    yaml_path = verify_dataset()
    if yaml_path is None:
        return
    
    # Step 2: Train (choose one)
    # Option A: Standard training
    model_path, results = train_yolo_tiled(yaml_path, model_size='x', epochs=100)
    
    # Option B: With focal loss (uncomment if class imbalance is severe)
    # model_path, results = train_yolo_with_focal_loss(yaml_path, model_size='x', epochs=100)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Best model: {model_path}")
    print("\n📝 Next steps:")
    print("   1. Run SAHI inference on test images:")
    print("      python inference_sahi.py")
    print("   2. Crop detected cells for Stage 2 classifier")
    
    return model_path


if __name__ == "__main__":
    main()
