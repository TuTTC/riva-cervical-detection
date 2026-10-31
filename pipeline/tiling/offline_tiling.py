# =============================================================================
# OFFLINE TILING - BƯỚC 1: CẮT ẢNH GỐC THÀNH MẢNH 1024x1024
# =============================================================================
# Mục tiêu: Giữ nguyên độ phân giải gốc để model thấy chi tiết hạt nhân tế bào
# Kỹ thuật: Sliding window với overlap 20% để không miss cells ở biên
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import json
import shutil

# ===================== CONFIGURATION =====================
# Điều chỉnh theo môi trường của bạn
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    INPUT_DIR = '/kaggle/input/riva-track-a'
    OUTPUT_DIR = '/kaggle/working/tiled_dataset'
else:
    INPUT_DIR = r'd:\NĂM 3\RIVA\riva-cervical-cytology-challenge-isbi-2026'
    OUTPUT_DIR = r'd:\NĂM 3\RIVA\tiled_dataset'

# Tiling parameters
TILE_SIZE = 1024  # Kích thước tile (giữ độ phân giải gốc)
OVERLAP_RATIO = 0.2  # 20% overlap giữa các tile
OVERLAP = int(TILE_SIZE * OVERLAP_RATIO)
STRIDE = TILE_SIZE - OVERLAP  # Bước nhảy = 1024 - 204 = 820px

# Minimum box size after clipping (bỏ các box quá nhỏ sau khi cắt)
MIN_BOX_AREA_RATIO = 0.3  # Box phải còn >= 30% diện tích gốc

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          OFFLINE TILING FOR HIGH-RES DETECTION               ║
╠══════════════════════════════════════════════════════════════╣
║  Tile Size: {TILE_SIZE}x{TILE_SIZE}                                           ║
║  Overlap: {OVERLAP_RATIO*100:.0f}% ({OVERLAP}px)                                         ║
║  Stride: {STRIDE}px                                             ║
║  Input: {INPUT_DIR[:50]}...                
║  Output: {OUTPUT_DIR[:50]}...               
╚══════════════════════════════════════════════════════════════╝
""")


def create_directory_structure():
    """Tạo cấu trúc thư mục output"""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
    print("✅ Created directory structure")


def clip_box_to_tile(box, tile_x, tile_y, tile_w, tile_h):
    """
    Clip box coordinates to tile boundaries và chuyển về tọa độ local
    
    Args:
        box: (x_center, y_center, width, height) trong tọa độ ảnh gốc
        tile_x, tile_y: Góc trên-trái của tile trong ảnh gốc
        tile_w, tile_h: Kích thước tile
    
    Returns:
        (x_center_local, y_center_local, w_clipped, h_clipped) hoặc None nếu box không nằm trong tile
    """
    x_c, y_c, w, h = box
    
    # Tính bounding box edges trong ảnh gốc
    x1_global = x_c - w / 2
    y1_global = y_c - h / 2
    x2_global = x_c + w / 2
    y2_global = y_c + h / 2
    
    # Tile boundaries
    tx1, ty1 = tile_x, tile_y
    tx2, ty2 = tile_x + tile_w, tile_y + tile_h
    
    # Check intersection
    if x2_global <= tx1 or x1_global >= tx2 or y2_global <= ty1 or y1_global >= ty2:
        return None  # No intersection
    
    # Clip to tile
    x1_clipped = max(x1_global, tx1)
    y1_clipped = max(y1_global, ty1)
    x2_clipped = min(x2_global, tx2)
    y2_clipped = min(y2_global, ty2)
    
    # Calculate clipped dimensions
    w_clipped = x2_clipped - x1_clipped
    h_clipped = y2_clipped - y1_clipped
    
    # Check minimum area
    original_area = w * h
    clipped_area = w_clipped * h_clipped
    if clipped_area < original_area * MIN_BOX_AREA_RATIO:
        return None  # Too small after clipping
    
    # Convert to local tile coordinates (center format)
    x_c_local = (x1_clipped + x2_clipped) / 2 - tile_x
    y_c_local = (y1_clipped + y2_clipped) / 2 - tile_y
    
    return (x_c_local, y_c_local, w_clipped, h_clipped)


def process_image(img_path, annotations_df, output_img_dir, output_lbl_dir, img_filename):
    """
    Cắt 1 ảnh thành nhiều tiles và map annotations
    
    Returns:
        int: Số tiles được tạo
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Cannot read: {img_path}")
        return 0
    
    h_img, w_img = img.shape[:2]
    
    # Get all annotations for this image
    img_annotations = annotations_df[annotations_df['image_filename'] == img_filename]
    
    # Parse annotations: (class, x_center, y_center, width, height)
    boxes = []
    for _, row in img_annotations.iterrows():
        cls = int(row['class'])
        x_c = float(row['x'])  # Assuming x,y are center coords
        y_c = float(row['y'])
        w = float(row['width'])
        h = float(row['height'])
        boxes.append((cls, x_c, y_c, w, h))
    
    tile_count = 0
    base_name = Path(img_filename).stem
    
    # Sliding window
    for y in range(0, h_img, STRIDE):
        for x in range(0, w_img, STRIDE):
            # Calculate tile boundaries
            tile_x1 = x
            tile_y1 = y
            tile_x2 = min(x + TILE_SIZE, w_img)
            tile_y2 = min(y + TILE_SIZE, h_img)
            
            tile_w = tile_x2 - tile_x1
            tile_h = tile_y2 - tile_y1
            
            # Skip tiny edge tiles
            if tile_w < TILE_SIZE * 0.5 or tile_h < TILE_SIZE * 0.5:
                continue
            
            # Crop tile
            tile_img = img[tile_y1:tile_y2, tile_x1:tile_x2]
            
            # Pad if needed (edge tiles)
            if tile_w < TILE_SIZE or tile_h < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                padded[:tile_h, :tile_w] = tile_img
                tile_img = padded
            
            # Map annotations to this tile
            tile_labels = []
            for (cls, x_c, y_c, w, h) in boxes:
                clipped = clip_box_to_tile(
                    (x_c, y_c, w, h), 
                    tile_x1, tile_y1, 
                    TILE_SIZE, TILE_SIZE
                )
                if clipped is not None:
                    x_c_local, y_c_local, w_clip, h_clip = clipped
                    # Normalize to [0, 1]
                    x_n = x_c_local / TILE_SIZE
                    y_n = y_c_local / TILE_SIZE
                    w_n = w_clip / TILE_SIZE
                    h_n = h_clip / TILE_SIZE
                    
                    # Clamp to valid range
                    x_n = max(0, min(1, x_n))
                    y_n = max(0, min(1, y_n))
                    w_n = max(0, min(1, w_n))
                    h_n = max(0, min(1, h_n))
                    
                    tile_labels.append(f"{cls} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}")
            
            # Only save tiles with annotations (during training)
            # For inference, we need all tiles
            if len(tile_labels) > 0:
                tile_name = f"{base_name}_tile_{x}_{y}"
                
                # Save tile image
                tile_img_path = os.path.join(output_img_dir, f"{tile_name}.jpg")
                cv2.imwrite(tile_img_path, tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save labels
                tile_lbl_path = os.path.join(output_lbl_dir, f"{tile_name}.txt")
                with open(tile_lbl_path, 'w') as f:
                    f.write('\n'.join(tile_labels))
                
                tile_count += 1
    
    return tile_count


def process_split(split):
    """Process all images in a split (train/val)"""
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} split...")
    print(f"{'='*60}")
    
    # Paths
    csv_path = os.path.join(INPUT_DIR, 'annotations', 'annotations', f'{split}.csv')
    img_dir = os.path.join(INPUT_DIR, 'images', 'images', split)
    output_img_dir = os.path.join(OUTPUT_DIR, 'images', split)
    output_lbl_dir = os.path.join(OUTPUT_DIR, 'labels', split)
    
    # Load annotations
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path, engine='python')
    print(f"📊 Loaded {len(df)} annotations")
    
    # Get unique images
    image_files = df['image_filename'].unique()
    print(f"📷 Found {len(image_files)} images")
    
    total_tiles = 0
    total_annotations = 0
    
    for img_filename in tqdm(image_files, desc=f"Tiling {split}"):
        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            continue
        
        n_tiles = process_image(
            img_path, df, 
            output_img_dir, output_lbl_dir, 
            img_filename
        )
        total_tiles += n_tiles
    
    # Count total annotations in output
    label_files = list(Path(output_lbl_dir).glob('*.txt'))
    for lbl_file in label_files:
        with open(lbl_file, 'r') as f:
            total_annotations += len(f.readlines())
    
    print(f"\n✅ {split.upper()} Complete:")
    print(f"   📷 Tiles created: {total_tiles}")
    print(f"   📋 Total annotations: {total_annotations}")
    
    return {'tiles': total_tiles, 'annotations': total_annotations}


def create_yaml_config():
    """Create data.yaml for YOLO training"""
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    
    config = {
        'path': OUTPUT_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 8,
        'names': ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
    }
    
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Created data.yaml: {yaml_path}")
    return yaml_path


def analyze_dataset():
    """Phân tích thống kê dataset sau khi tiling"""
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)
    
    for split in ['train', 'val']:
        lbl_dir = os.path.join(OUTPUT_DIR, 'labels', split)
        if not os.path.exists(lbl_dir):
            continue
        
        class_counts = {i: 0 for i in range(8)}
        total_boxes = 0
        
        for lbl_file in Path(lbl_dir).glob('*.txt'):
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        class_counts[cls] += 1
                        total_boxes += 1
        
        print(f"\n{split.upper()}:")
        print(f"  Total boxes: {total_boxes}")
        print("  Class distribution:")
        class_names = ['NILM', 'ENDO', 'INFL', 'ASCUS', 'LSIL', 'HSIL', 'ASCH', 'SCC']
        for cls, name in enumerate(class_names):
            pct = (class_counts[cls] / total_boxes * 100) if total_boxes > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"    [{cls}] {name:6s}: {class_counts[cls]:5d} ({pct:5.1f}%) {bar}")


def main():
    print("\n🚀 Starting Offline Tiling Pipeline...")
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Process train and val
    stats = {}
    for split in ['train', 'val']:
        stats[split] = process_split(split)
    
    # Step 3: Create YAML config
    yaml_path = create_yaml_config()
    
    # Step 4: Analyze
    analyze_dataset()
    
    print("\n" + "="*60)
    print("🎉 TILING COMPLETE!")
    print("="*60)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 YAML config: {yaml_path}")
    print("\n📝 Next steps:")
    print("   1. Train detection model on tiled dataset:")
    print(f"      python train_tiled_detection.py")
    print("   2. Use SAHI for inference on full-size test images")
    
    return OUTPUT_DIR


if __name__ == "__main__":
    main()
