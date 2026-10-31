"""
Đếm số ảnh trong từng folder của dataset RIVA
"""
import os
from pathlib import Path

# Đường dẫn đến data
DATA_DIR = r"/home/combine2t/CamTu/RIVA-TRACK-A/frcnn_paper/data"

# Hoặc nếu trên Kaggle
# DATA_DIR = "/kaggle/input/riva-track-a"

def count_images(directory):
    """Đếm số file ảnh trong folder"""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    count = 0
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in extensions:
            count += 1
    return count

def scan_folders(root_dir, depth=0):
    """Scan và đếm ảnh trong tất cả subfolder"""
    if not os.path.exists(root_dir):
        print(f" Không tìm thấy: {root_dir}")
        return
    
    indent = "  " * depth
    
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        
        if os.path.isdir(item_path):
            # Đếm ảnh trực tiếp trong folder này
            img_count = count_images(item_path)
            
            if img_count > 0:
                print(f"{indent} {item}: {img_count} ảnh")
            else:
                print(f"{indent} {item}/")
            
            # Đệ quy vào subfolder
            scan_folders(item_path, depth + 1)


if __name__ == "__main__":
    print("=" * 60)
    print(" THỐNG KÊ SỐ ẢNH TRONG DATASET RIVA")
    print("=" * 60)
    print(f"Scanning: {DATA_DIR}\n")
    
    scan_folders(DATA_DIR)
    
    print("\n" + "=" * 60)
    
    # Tổng hợp nhanh cho images folder
    images_dir = os.path.join(DATA_DIR, "images", "images")
    if os.path.exists(images_dir):
        print("\n TỔNG HỢP:")
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(images_dir, split)
            if os.path.exists(split_dir):
                count = count_images(split_dir)
                print(f"   {split}: {count} ảnh")
