# =============================================================================
# FASTER R-CNN PAPER - MODEL
# =============================================================================
# Model creation and configuration
# =============================================================================

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import *


def create_model(pretrained=True):
    """
    Create Faster R-CNN model with ResNet-50-FPN backbone
    
    Args:
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: Faster R-CNN model
    """
    print("\n🔧 Creating Faster R-CNN model...")
    
    # Load pretrained model
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        print("   ✓ Loaded pretrained ImageNet weights")
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
    
    # Replace classification head for our classes
    # NUM_CLASSES + 1 for background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    
    print(f"   ✓ Modified for {NUM_CLASSES} classes (+1 background)")
    print(f"   ✓ Box predictor input features: {in_features}")
    
    return model


def load_model(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    print(f"\n Loading model from {checkpoint_path}...")
    
    model = create_model(pretrained=False)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    print("   ✓ Model loaded successfully")
    
    return model


def get_optimizer(model):
    """
    Create optimizer following paper specifications
    
    Args:
        model: Model to optimize
    
    Returns:
        optimizer: SGD optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(
        params,
        lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    print(f"\n Optimizer: SGD")
    print(f"   LR: {BASE_LR}, Momentum: {MOMENTUM}, Weight Decay: {WEIGHT_DECAY}")
    
    return optimizer


def get_scheduler(optimizer):
    """
    Create learning rate scheduler following paper specifications
    
    Args:
        optimizer: Optimizer to schedule
    
    Returns:
        scheduler: MultiStepLR scheduler
    """
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_DECAY_EPOCHS,
        gamma=LR_DECAY_GAMMA
    )
    
    print(f" LR Scheduler: MultiStepLR")
    print(f"   Milestones: {LR_DECAY_EPOCHS}, Gamma: {LR_DECAY_GAMMA}")
    
    return scheduler


def model_summary(model):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n Model Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")


if __name__ == "__main__":
    # Test model creation
    model = create_model(pretrained=True)
    model_summary(model)
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n Forward pass successful")
    print(f"   Output keys: {output[0].keys()}")
