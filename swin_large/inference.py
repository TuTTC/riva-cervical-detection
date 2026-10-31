# =============================================================================
# SWIN TRANSFORMER LARGE - INFERENCE SCRIPT
# =============================================================================
# Run inference on test set with optional TTA and LightGBM fusion
# =============================================================================

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OUTPUT_DIR, WORK_DIR, TEST_IMG_DIR, CLASS_NAMES, NUM_CLASSES,
    CONF_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS, IMG_SIZE,
    TRAIN_CSV, TRAIN_IMG_DIR
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(config_path, checkpoint_path, device='cuda:0'):
    """Load trained Swin Transformer Large model"""
    print(f"Loading model from: {checkpoint_path}")
    
    try:
        from mmdet.apis import init_detector
        
        model = init_detector(config_path, checkpoint_path, device=device)
        print("Model loaded successfully!")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative loading method...")
        
        import torch
        from mmengine.config import Config
        from mmdet.registry import MODELS
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Build model
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        
        # Load weights
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        print("Model loaded via alternative method!")
        
        return model


# =============================================================================
# STANDARD INFERENCE
# =============================================================================

def run_inference(model, image_path, device='cuda:0'):
    """Run inference on a single image"""
    from mmdet.apis import inference_detector
    
    result = inference_detector(model, image_path)
    
    # Extract predictions
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    
    return bboxes, scores, labels


# =============================================================================
# TEST TIME AUGMENTATION (TTA)
# =============================================================================

class TTAInference:
    """
    Test Time Augmentation for improved predictions.
    
    Augmentations applied:
    - Original image
    - Horizontal flip
    - Vertical flip
    - Multi-scale (optional)
    
    Results are merged using Weighted Boxes Fusion (WBF).
    """
    
    def __init__(self, model, scales=[1.0], use_flip=True, device='cuda:0'):
        """
        Args:
            model: Detection model
            scales: List of scale factors for multi-scale inference
            use_flip: Whether to use flip augmentations
            device: Device to run inference on
        """
        self.model = model
        self.scales = scales
        self.use_flip = use_flip
        self.device = device
    
    def __call__(self, image_path):
        """
        Run TTA inference on an image.
        
        Returns:
            bboxes, scores, labels: Merged predictions
        """
        from mmdet.apis import inference_detector
        
        # Read original image
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for scale in self.scales:
            # Scale image
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(img, (new_w, new_h))
            else:
                scaled_img = img.copy()
                new_h, new_w = h, w
            
            # Original
            result = inference_detector(self.model, scaled_img)
            boxes, scores, labels = self._extract_predictions(result)
            if len(boxes) > 0:
                # Scale boxes back to original size
                boxes = boxes / scale
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
            
            if self.use_flip:
                # Horizontal flip
                flipped_h = np.fliplr(scaled_img).copy()
                result = inference_detector(self.model, flipped_h)
                boxes, scores, labels = self._extract_predictions(result)
                if len(boxes) > 0:
                    # Flip boxes back
                    boxes[:, [0, 2]] = new_w - boxes[:, [2, 0]]
                    boxes = boxes / scale
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)
                
                # Vertical flip
                flipped_v = np.flipud(scaled_img).copy()
                result = inference_detector(self.model, flipped_v)
                boxes, scores, labels = self._extract_predictions(result)
                if len(boxes) > 0:
                    # Flip boxes back
                    boxes[:, [1, 3]] = new_h - boxes[:, [3, 1]]
                    boxes = boxes / scale
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)
        
        if len(all_boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        
        # Merge predictions using WBF
        merged_boxes, merged_scores, merged_labels = self._weighted_boxes_fusion(
            all_boxes, all_scores, all_labels, (h, w)
        )
        
        return merged_boxes, merged_scores, merged_labels
    
    def _extract_predictions(self, result):
        """Extract predictions from MMDetection result"""
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        return bboxes, scores, labels
    
    def _weighted_boxes_fusion(self, boxes_list, scores_list, labels_list, image_size, 
                               iou_threshold=0.5, skip_box_threshold=0.001):
        """
        Weighted Boxes Fusion for merging predictions.
        """
        try:
            from ensemble_boxes import weighted_boxes_fusion
            
            h, w = image_size
            
            # Normalize boxes to [0, 1]
            norm_boxes_list = []
            for boxes in boxes_list:
                norm_boxes = boxes.copy()
                norm_boxes[:, [0, 2]] /= w
                norm_boxes[:, [1, 3]] /= h
                norm_boxes = np.clip(norm_boxes, 0, 1)
                norm_boxes_list.append(norm_boxes.tolist())
            
            scores_list = [s.tolist() for s in scores_list]
            labels_list = [l.tolist() for l in labels_list]
            
            # Run WBF
            boxes, scores, labels = weighted_boxes_fusion(
                norm_boxes_list, scores_list, labels_list,
                iou_thr=iou_threshold,
                skip_box_thr=skip_box_threshold
            )
            
            # Denormalize boxes
            boxes = np.array(boxes)
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            
            return boxes, np.array(scores), np.array(labels, dtype=np.int64)
            
        except ImportError:
            print("Warning: ensemble_boxes not installed, using simple NMS instead")
            return self._simple_merge(boxes_list, scores_list, labels_list)
    
    def _simple_merge(self, boxes_list, scores_list, labels_list):
        """Simple merge using NMS as fallback"""
        import torch
        from torchvision.ops import nms
        
        all_boxes = np.concatenate(boxes_list, axis=0)
        all_scores = np.concatenate(scores_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        
        # NMS per class
        keep_indices = []
        for cls in np.unique(all_labels):
            cls_mask = all_labels == cls
            cls_boxes = torch.tensor(all_boxes[cls_mask])
            cls_scores = torch.tensor(all_scores[cls_mask])
            
            keep = nms(cls_boxes, cls_scores, 0.5)
            keep_indices.extend(np.where(cls_mask)[0][keep.numpy()])
        
        keep_indices = np.array(keep_indices)
        
        return all_boxes[keep_indices], all_scores[keep_indices], all_labels[keep_indices]


# =============================================================================
# LIGHTGBM FEATURE FUSION
# =============================================================================

class LightGBMFusion:
    """
    LightGBM-based feature fusion for improved classification.
    
    Extracts radiomics + deep features from detected cells,
    trains a LightGBM classifier, and re-classifies detections.
    """
    
    def __init__(self, model, model_path=None, device='cuda:0'):
        """
        Args:
            model: Detection model (for feature extraction)
            model_path: Path to saved LightGBM model (optional)
            device: Device for deep feature extraction
        """
        self.det_model = model
        self.device = device
        self.lgb_model = None
        self.scaler = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def extract_features(self, image, bbox):
        """
        Extract features from a detected cell crop.
        
        Returns:
            features: numpy array of combined features
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        pad = int(max(x2 - x1, y2 - y1) * 0.1)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)
        
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        # Extract radiomics features
        radio_features = self._extract_radiomics(crop)
        
        return radio_features
    
    def _extract_radiomics(self, crop):
        """Extract radiomics features from crop"""
        try:
            from skimage import feature as skimage_feature
            from skimage.measure import regionprops, label as skimage_label
        except ImportError:
            # Return dummy features if skimage not available
            return np.zeros(31, dtype=np.float32)
        
        features = []
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # Shape features (8)
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            labeled = skimage_label(binary)
            
            if labeled.max() > 0:
                props = regionprops(labeled)[0]
                features.extend([
                    props.area,
                    props.perimeter,
                    props.eccentricity,
                    props.solidity,
                    props.extent,
                    props.major_axis_length,
                    props.minor_axis_length,
                    props.major_axis_length / (props.minor_axis_length + 1e-6),
                ])
            else:
                features.extend([0] * 8)
        except:
            features.extend([0] * 8)
        
        # LBP texture features (10)
        try:
            lbp = skimage_feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            features.extend(lbp_hist.tolist())
        except:
            features.extend([0] * 10)
        
        # Intensity features (7)
        features.extend([
            gray.mean(),
            gray.std(),
            gray.min(),
            gray.max(),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
        ])
        
        # Color features (6)
        if len(crop.shape) == 3:
            for channel in range(3):
                ch = crop[:, :, channel]
                features.extend([ch.mean(), ch.std()])
        else:
            features.extend([0] * 6)
        
        return np.array(features, dtype=np.float32)
    
    def train(self, train_csv, train_img_dir, output_path=None):
        """
        Train LightGBM classifier on training data.
        
        Args:
            train_csv: Path to training annotations CSV
            train_img_dir: Path to training images
            output_path: Path to save trained model
        """
        import lightgbm as lgb
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        print("\n" + "=" * 60)
        print("TRAINING LIGHTGBM CLASSIFIER")
        print("=" * 60)
        
        # Load annotations
        df = pd.read_csv(train_csv)
        
        features_list = []
        labels_list = []
        
        print("Extracting features from training data...")
        
        for img_name in tqdm(df['image_filename'].unique(), desc='Extracting features'):
            img_path = os.path.join(train_img_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            img_anns = df[df['image_filename'] == img_name]
            
            for _, row in img_anns.iterrows():
                x, y, w, h = row['x'], row['y'], row['width'], row['height']
                bbox = [x, y, x + w, y + h]
                
                feats = self.extract_features(img, bbox)
                if feats is not None:
                    features_list.append(feats)
                    labels_list.append(row['class'])
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"Extracted {len(X)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': NUM_CLASSES,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        print("Training LightGBM...")
        
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Evaluate
        y_pred = self.lgb_model.predict(X_val).argmax(axis=1)
        accuracy = (y_pred == y_val).mean()
        print(f"\nValidation accuracy: {accuracy:.4f}")
        
        # Save model
        if output_path:
            self.save(output_path)
        
        return accuracy
    
    def reclassify(self, image, bboxes, scores, labels):
        """
        Re-classify detections using LightGBM.
        
        Args:
            image: Original image
            bboxes: Detection boxes
            scores: Detection scores
            labels: Detection labels
        
        Returns:
            new_labels, new_scores: Updated predictions
        """
        if self.lgb_model is None:
            print("Warning: LightGBM model not trained/loaded!")
            return labels, scores
        
        new_labels = []
        new_scores = []
        
        for bbox, score, label in zip(bboxes, scores, labels):
            feats = self.extract_features(image, bbox)
            
            if feats is not None:
                feats_scaled = self.scaler.transform(feats.reshape(1, -1))
                probs = self.lgb_model.predict(feats_scaled)[0]
                
                # Combine detector score with LightGBM prediction
                new_label = probs.argmax()
                new_score = score * 0.5 + probs[new_label] * 0.5
                
                new_labels.append(new_label)
                new_scores.append(new_score)
            else:
                new_labels.append(label)
                new_scores.append(score)
        
        return np.array(new_labels, dtype=np.int64), np.array(new_scores)
    
    def save(self, path):
        """Save LightGBM model and scaler"""
        save_dict = {
            'lgb_model': self.lgb_model,
            'scaler': self.scaler
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Saved LightGBM model to: {path}")
    
    def load(self, path):
        """Load LightGBM model and scaler"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        self.lgb_model = save_dict['lgb_model']
        self.scaler = save_dict['scaler']
        print(f"Loaded LightGBM model from: {path}")


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================

def generate_submission(
    model,
    test_dir,
    output_csv,
    conf_threshold=0.001,
    device='cuda:0',
    use_tta=False,
    tta_scales=[1.0],
    use_fusion=False,
    fusion_model_path=None
):
    """
    Generate submission CSV from test images.
    
    Args:
        model: Detection model
        test_dir: Path to test images
        output_csv: Output CSV path
        conf_threshold: Confidence threshold
        device: Device to use
        use_tta: Whether to use TTA
        tta_scales: Scales for TTA
        use_fusion: Whether to use LightGBM fusion
        fusion_model_path: Path to LightGBM model
    """
    print(f"\nGenerating predictions on test set...")
    print(f"  TTA: {'ENABLED' if use_tta else 'DISABLED'}")
    print(f"  LightGBM Fusion: {'ENABLED' if use_fusion else 'DISABLED'}")
    
    # Initialize TTA if enabled
    tta_inference = None
    if use_tta:
        tta_inference = TTAInference(model, scales=tta_scales, use_flip=True, device=device)
        print(f"  TTA scales: {tta_scales}")
    
    # Initialize LightGBM fusion if enabled
    lgb_fusion = None
    if use_fusion:
        lgb_fusion = LightGBMFusion(model, model_path=fusion_model_path, device=device)
        if lgb_fusion.lgb_model is None:
            print("  Training LightGBM classifier...")
            lgb_fusion.train(
                TRAIN_CSV, TRAIN_IMG_DIR,
                os.path.join(OUTPUT_DIR, 'lgb_classifier.pkl')
            )
    
    # Get test images
    test_images = sorted([
        f for f in os.listdir(test_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    print(f"Found {len(test_images)} test images")
    
    results_list = []
    pred_id = 0
    
    for img_name in tqdm(test_images, desc='Inference'):
        img_path = os.path.join(test_dir, img_name)
        
        try:
            # Run inference
            if use_tta:
                bboxes, scores, labels = tta_inference(img_path)
            else:
                bboxes, scores, labels = run_inference(model, img_path, device)
            
            if len(bboxes) == 0:
                # No detections - add dummy prediction
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                results_list.append({
                    'id': pred_id,
                    'image_filename': img_name,
                    'class': 0,
                    'x': w / 2,
                    'y': h / 2,
                    'width': 100,
                    'height': 100,
                    'conf': 0.001
                })
                pred_id += 1
                continue
            
            # Apply LightGBM fusion if enabled
            if use_fusion and lgb_fusion is not None:
                img = cv2.imread(img_path)
                labels, scores = lgb_fusion.reclassify(img, bboxes, scores, labels)
            
            # Filter by confidence
            mask = scores >= conf_threshold
            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Convert to submission format
            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox
                
                # Convert to center format
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                results_list.append({
                    'id': pred_id,
                    'image_filename': img_name,
                    'class': int(label),
                    'x': float(cx),
                    'y': float(cy),
                    'width': float(w),
                    'height': float(h),
                    'conf': float(score)
                })
                pred_id += 1
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            # Add dummy prediction
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': 0,
                'x': 512,
                'y': 512,
                'width': 100,
                'height': 100,
                'conf': 0.001
            })
            pred_id += 1
    
    # Create DataFrame
    submission_df = pd.DataFrame(results_list)
    submission_df = submission_df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]
    
    # Save submission
    submission_df.to_csv(output_csv, index=False)
    
    print(f"\nSubmission saved to: {output_csv}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Print class distribution
    print("\nClass distribution:")
    for cls_id in sorted(submission_df['class'].unique()):
        count = len(submission_df[submission_df['class'] == cls_id])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Class_{cls_id}'
        print(f"  {cls_name}: {count}")
    
    return submission_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main inference pipeline with optional TTA and fusion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Swin Transformer Large Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic inference
  python inference.py
  
  # With TTA
  python inference.py --tta
  
  # With TTA multi-scale
  python inference.py --tta --tta-scales 0.8 1.0 1.2
  
  # With LightGBM fusion
  python inference.py --fusion
  
  # With both TTA and fusion
  python inference.py --tta --fusion
  
  # Train LightGBM classifier only
  python inference.py --train-fusion
'''
    )
    
    parser.add_argument('--config', type=str, 
                       default=os.path.join(OUTPUT_DIR, 'swin_large_config.py'),
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, 
                       default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--test-dir', type=str, 
                       default=TEST_IMG_DIR,
                       help='Path to test images directory')
    parser.add_argument('--output', type=str, 
                       default=None,
                       help='Output CSV path')
    parser.add_argument('--conf-threshold', type=float, 
                       default=CONF_THRESHOLD,
                       help='Confidence threshold')
    parser.add_argument('--device', type=str, 
                       default='cuda:0',
                       help='Device to use')
    
    # TTA options
    parser.add_argument('--tta', action='store_true',
                       help='Enable Test Time Augmentation')
    parser.add_argument('--tta-scales', type=float, nargs='+',
                       default=[1.0],
                       help='Scales for TTA (e.g., 0.8 1.0 1.2)')
    
    # LightGBM fusion options
    parser.add_argument('--fusion', action='store_true',
                       help='Enable LightGBM feature fusion')
    parser.add_argument('--fusion-model', type=str,
                       default=os.path.join(OUTPUT_DIR, 'lgb_classifier.pkl'),
                       help='Path to LightGBM model')
    parser.add_argument('--train-fusion', action='store_true',
                       help='Train LightGBM classifier only (no inference)')
    
    args = parser.parse_args()
    
    # Set default output path based on options
    if args.output is None:
        suffix = ''
        if args.tta:
            suffix += '_tta'
        if args.fusion:
            suffix += '_fusion'
        args.output = os.path.join(OUTPUT_DIR, f'submission_swin_large{suffix}.csv')
    
    # Find checkpoint if not specified
    if args.checkpoint is None:
        checkpoints = sorted([
            f for f in os.listdir(WORK_DIR) 
            if f.endswith('.pth')
        ]) if os.path.exists(WORK_DIR) else []
        
        if checkpoints:
            best_ckpts = [f for f in checkpoints if 'best' in f.lower()]
            if best_ckpts:
                args.checkpoint = os.path.join(WORK_DIR, best_ckpts[-1])
            else:
                args.checkpoint = os.path.join(WORK_DIR, checkpoints[-1])
            print(f"Using checkpoint: {args.checkpoint}")
        else:
            if not args.train_fusion:
                print("Error: No checkpoint found!")
                return
    
    print("=" * 80)
    print("SWIN TRANSFORMER LARGE - INFERENCE")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"TTA: {'ENABLED' if args.tta else 'DISABLED'}")
    if args.tta:
        print(f"  Scales: {args.tta_scales}")
    print(f"LightGBM Fusion: {'ENABLED' if args.fusion else 'DISABLED'}")
    print("=" * 80)
    
    # Train fusion model only
    if args.train_fusion:
        print("\nTraining LightGBM classifier...")
        model = load_model(args.config, args.checkpoint, args.device) if args.checkpoint else None
        lgb_fusion = LightGBMFusion(model, device=args.device)
        lgb_fusion.train(TRAIN_CSV, TRAIN_IMG_DIR, args.fusion_model)
        print("\nLightGBM training complete!")
        return
    
    # Load model
    model = load_model(args.config, args.checkpoint, args.device)
    
    # Generate submission
    submission_df = generate_submission(
        model=model,
        test_dir=args.test_dir,
        output_csv=args.output,
        conf_threshold=args.conf_threshold,
        device=args.device,
        use_tta=args.tta,
        tta_scales=args.tta_scales,
        use_fusion=args.fusion,
        fusion_model_path=args.fusion_model if args.fusion else None
    )
    
    print("\nInference complete!")
    
    return submission_df


if __name__ == "__main__":
    main()
