# =============================================================================
# FASTER R-CNN PAPER - INFERENCE
# =============================================================================
# Inference and submission generation
# =============================================================================

import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import *
from model import create_model, load_model


def run_inference(model, device='cuda'):
    """
    Run inference on test images
    
    Args:
        model: Trained model
        device: Device to run on
    
    Returns:
        results: List of detection results
    """
    print("\n Running inference on test set...")
    
    model.to(device)
    model.eval()
    
    test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg'))])
    print(f"   Test images: {len(test_images)}")
    
    results = []
    pred_id = 0
    
    for img_name in tqdm(test_images, desc="Inference"):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            predictions = model(img_tensor)[0]
        
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        # Scale back to original size
        scale_x = w_orig / IMG_SIZE
        scale_y = h_orig / IMG_SIZE
        
        # Handle no detections
        if len(boxes) == 0:
            results.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': 0,
                'x': w_orig / 2,
                'y': h_orig / 2,
                'width': 100,
                'height': 100,
                'conf': 0.001
            })
            pred_id += 1
            continue
        
        # Process detections
        for box, label, score in zip(boxes, labels, scores):
            if score < CONF_THRESHOLD:
                continue
            
            x1, y1, x2, y2 = box
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            
            results.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': int(label) - 1,  # Remove background offset
                'x': float((x1 + x2) / 2),
                'y': float((y1 + y2) / 2),
                'width': float(x2 - x1),
                'height': float(y2 - y1),
                'conf': float(score)
            })
            pred_id += 1
    
    print(f"   Total predictions: {len(results)}")
    return results


def generate_submission(model=None, checkpoint_path=None, output_name='submission_frcnn_paper.csv'):
    """
    Generate submission CSV file
    
    Args:
        model: Trained model (optional)
        checkpoint_path: Path to checkpoint (if model not provided)
        output_name: Output filename
    
    Returns:
        submission_path: Path to saved submission
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if not provided
    if model is None:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(OUTPUT_DIR, 'faster_rcnn_final.pth')
        model = load_model(checkpoint_path, device)
    
    # Run inference
    results = run_inference(model, device)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]
    
    # Save
    submission_path = os.path.join(BASE_DIR, output_name)
    df.to_csv(submission_path, index=False)
    
    print(f"\n Submission saved: {submission_path}")
    print(f"   Total predictions: {len(df)}")
    
    # Class distribution
    print(f"\n  Class distribution:")
    for cls_id, count in df['class'].value_counts().sort_index().items():
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'Unknown'
        print(f"   {cls_name}: {count}")
    
    return submission_path


def visualize_predictions(model, num_images=5, save_dir=None):
    """
    Visualize predictions on sample images
    
    Args:
        model: Trained model
        num_images: Number of images to visualize
        save_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    test_images = sorted(os.listdir(TEST_IMG_DIR))[:num_images]
    
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    
    for img_name in test_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Inference
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(img_tensor)[0]
        
        # Draw predictions
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(img_resized)
        
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.3:
                continue
            
            x1, y1, x2, y2 = box
            color = colors[label % NUM_CLASSES]
            
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
            
            cls_name = CLASS_NAMES[label - 1] if label > 0 else 'BG'
            ax.text(x1, y1 - 5, f'{cls_name}: {score:.2f}',
                    color=color, fontsize=8, weight='bold')
        
        ax.set_title(f'{img_name}')
        ax.axis('off')
        
        save_path = os.path.join(save_dir, f'pred_{img_name}')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
    
    print(f" Visualizations saved to: {save_dir}")


if __name__ == "__main__":
    # Generate submission from trained model
    generate_submission()
