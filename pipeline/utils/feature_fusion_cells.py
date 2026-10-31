# ============================================================================
# FEATURE FUSION IMPLEMENTATION - Add this as new cells in your notebook
# ============================================================================

# CELL 1: Install Required Libraries
!pip install -q scikit-image radiomics lightgbm

# CELL 2: Import Libraries
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage.measure import regionprops, label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pickle
from tqdm import tqdm
import os

# CELL 3: Radiomics Feature Extractor
class RadiomicsExtractor:
    """Extract radiomics features from cell crops"""
    
    def extract_features(self, crop_image):
        """
        Extract shape, texture, and intensity features
        Returns: numpy array of features
        """
        features = []
        
        # Convert to grayscale if needed
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_image
        
        # === SHAPE FEATURES ===
        # Create binary mask (simple thresholding)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        labeled = label(binary)
        
        if labeled.max() > 0:
            props = regionprops(labeled)[0]
            
            # Basic shape features
            features.extend([
                props.area,
                props.perimeter,
                props.eccentricity,
                props.solidity,
                props.extent,
                props.major_axis_length,
                props.minor_axis_length,
                props.major_axis_length / (props.minor_axis_length + 1e-6),  # aspect ratio
            ])
        else:
            features.extend([0] * 8)
        
        # === TEXTURE FEATURES (GLCM) ===
        # Local Binary Pattern
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
        features.extend(lbp_hist.tolist())
        
        # === INTENSITY FEATURES ===
        features.extend([
            gray.mean(),
            gray.std(),
            gray.min(),
            gray.max(),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
        ])
        
        # === COLOR FEATURES (if RGB) ===
        if len(crop_image.shape) == 3:
            for channel in range(3):
                ch = crop_image[:, :, channel]
                features.extend([
                    ch.mean(),
                    ch.std(),
                ])
        else:
            features.extend([0] * 6)
        
        return np.array(features)

# CELL 4: Deep Feature Extractor
class DeepFeatureExtractor:
    """Extract deep features from YOLOv8 backbone"""
    
    def __init__(self, model_path):
        """Load YOLOv8 model"""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        
    def extract_features(self, crop_image):
        """
        Extract deep features from crop
        Returns: numpy array of features (512-dim)
        """
        # Resize to model input size
        resized = cv2.resize(crop_image, (640, 640))
        
        # Get model prediction (which includes feature extraction)
        results = self.model(resized, verbose=False)
        
        # Extract features from the model's bottleneck layer
        # This is a simplified approach - you might need to modify based on your needs
        # For now, we'll use the detection features as a proxy
        if len(results[0].boxes) > 0:
            # Use confidence and class distribution as features
            boxes = results[0].boxes
            conf_features = boxes.conf.cpu().numpy()
            cls_features = boxes.cls.cpu().numpy()
            
            # Create a simple feature vector
            features = np.zeros(512)
            if len(conf_features) > 0:
                features[:8] = np.bincount(cls_features.astype(int), minlength=8)
                features[8:16] = [conf_features.mean(), conf_features.std(), 
                                 conf_features.min(), conf_features.max(),
                                 0, 0, 0, 0]
        else:
            features = np.zeros(512)
            
        return features

# CELL 5: Create Training Dataset
def create_feature_dataset(model_path, train_csv_path, train_images_dir, output_path):
    """
    Extract features from training data
    """
    # Initialize extractors
    radio_extractor = RadiomicsExtractor()
    deep_extractor = DeepFeatureExtractor(model_path)
    
    # Load training annotations
    train_df = pd.read_csv(train_csv_path)
    
    features_list = []
    labels_list = []
    
    print("Extracting features from training data...")
    
    for img_name in tqdm(train_df['image_filename'].unique()):
        # Load image
        img_path = os.path.join(train_images_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Get all annotations for this image
        img_annotations = train_df[train_df['image_filename'] == img_name]
        
        for _, row in img_annotations.iterrows():
            # Crop cell region
            x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
            
            # Ensure valid crop region
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Extract features
            radio_features = radio_extractor.extract_features(crop)
            deep_features = deep_extractor.extract_features(crop)
            
            # Combine features
            combined_features = np.concatenate([radio_features, deep_features])
            
            features_list.append(combined_features)
            labels_list.append(int(row['class']))
    
    # Save features
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    np.savez(output_path, features=features_array, labels=labels_array)
    print(f"Saved {len(features_array)} feature vectors to {output_path}")
    
    return features_array, labels_array

# CELL 6: Train LightGBM Classifier
def train_classifier(features_path, model_save_path):
    """
    Train LightGBM classifier on extracted features
    """
    # Load features
    data = np.load(features_path)
    X = data['features']
    y = data['labels']
    
    print(f"Training on {len(X)} samples")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Calculate class weights for imbalance
    class_weights = {}
    for cls in range(8):
        count = (y_train == cls).sum()
        if count > 0:
            class_weights[cls] = len(y_train) / (8 * count)
        else:
            class_weights[cls] = 1.0
    
    sample_weights = np.array([class_weights[cls] for cls in y_train])
    
    # Train LightGBM
    train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
    
    params = {
        'objective': 'multiclass',
        'num_class': 8,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print("Training LightGBM...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Evaluate
    y_pred = gbm.predict(X_val_scaled, num_iteration=gbm.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import classification_report
    print("\\nValidation Results:")
    print(classification_report(y_val, y_pred_class, 
                               target_names=[CLASS_NAMES[i] for i in range(8)]))
    
    # Save model and scaler
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': gbm, 'scaler': scaler}, f)
    
    print(f"Model saved to {model_save_path}")
    
    return gbm, scaler

# CELL 7: Two-Stage Inference
def two_stage_inference(yolo_model_path, classifier_path, test_images_dir, output_csv):
    """
    Two-stage inference: YOLOv8 detection + Feature fusion classification
    """
    from ultralytics import YOLO
    
    # Load models
    yolo_model = YOLO(yolo_model_path)
    
    with open(classifier_path, 'rb') as f:
        checkpoint = pickle.load(f)
        classifier = checkpoint['model']
        scaler = checkpoint['scaler']
    
    # Initialize extractors
    radio_extractor = RadiomicsExtractor()
    deep_extractor = DeepFeatureExtractor(yolo_model_path)
    
    # Get test images
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg'))]
    
    results_list = []
    pred_id = 0
    
    print("Running two-stage inference...")
    
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_images_dir, img_name)
        img = cv2.imread(img_path)
        
        # Stage 1: YOLOv8 Detection
        yolo_results = yolo_model.predict(img_path, conf=0.001, iou=0.4, verbose=False)
        
        if len(yolo_results[0].boxes) == 0:
            # No detections
            h, w = img.shape[:2]
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': 0,
                'x': w/2, 'y': h/2,
                'width': 100, 'height': 100,
                'conf': 0.001
            })
            pred_id += 1
            continue
        
        boxes = yolo_results[0].boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Crop cell
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Stage 2: Feature Extraction + Classification
            radio_features = radio_extractor.extract_features(crop)
            deep_features = deep_extractor.extract_features(crop)
            combined_features = np.concatenate([radio_features, deep_features])
            
            # Normalize and classify
            features_scaled = scaler.transform([combined_features])
            class_probs = classifier.predict(features_scaled, num_iteration=classifier.best_iteration)[0]
            
            # Get final class and confidence
            final_class = np.argmax(class_probs)
            final_conf = class_probs[final_class]
            
            # Convert to center format
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            results_list.append({
                'id': pred_id,
                'image_filename': img_name,
                'class': int(final_class),
                'x': float(x_center),
                'y': float(y_center),
                'width': float(width),
                'height': float(height),
                'conf': float(final_conf)
            })
            pred_id += 1
    
    # Save submission
    submission_df = pd.DataFrame(results_list)
    submission_df = submission_df[['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']]
    submission_df.to_csv(output_csv, index=False)
    
    print(f"Submission saved to {output_csv}")
    print(f"Total predictions: {len(submission_df)}")
    
    return submission_df

# CELL 8: Run the Full Pipeline
# Step 1: Create feature dataset
features_path = os.path.join(BASE_DIR, 'features_train.npz')
create_feature_dataset(
    model_path=MODEL_PATH,
    train_csv_path=os.path.join(INPUT_DIR, 'annotations', 'annotations', 'train.csv'),
    train_images_dir=os.path.join(INPUT_DIR, 'images', 'images', 'train'),
    output_path=features_path
)

# Step 2: Train classifier
classifier_path = os.path.join(BASE_DIR, 'lgbm_classifier.pkl')
train_classifier(features_path, classifier_path)

# Step 3: Run two-stage inference
submission_df = two_stage_inference(
    yolo_model_path=MODEL_PATH,
    classifier_path=classifier_path,
    test_images_dir=os.path.join(INPUT_DIR, 'images', 'images', 'test'),
    output_csv='submission_feature_fusion.csv'
)

print("\\nFeature Fusion Pipeline Complete!")
print(f"Class distribution:")
for cls_id, count in submission_df['class'].value_counts().sort_index().items():
    print(f"  {CLASS_NAMES[cls_id]}: {count}")
