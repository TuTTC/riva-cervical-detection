# =============================================================================
# FASTER R-CNN PAPER - TRAINING
# =============================================================================
# Training loop and utilities
# =============================================================================

import os
import torch
from tqdm import tqdm
from datetime import datetime

from config import *
from dataset import get_data_loaders
from model import create_model, get_optimizer, get_scheduler, model_summary


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        try:
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping (paper uses max_norm=35)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        except Exception as e:
            print(f"\n Batch error: {e}")
            continue
    
    avg_loss = epoch_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, data_loader, device):
    """
    Validate model and compute mAP
    
    Loss functions used in Faster R-CNN:
    - loss_classifier: CrossEntropyLoss for class prediction
    - loss_box_reg: SmoothL1Loss for bbox regression  
    - loss_objectness: BCEWithLogitsLoss for RPN
    - loss_rpn_box_reg: SmoothL1Loss for RPN bbox
    
    Validation metrics:
    - mAP@0.5: mean Average Precision at IoU threshold 0.5
    - mAP@0.5:0.95: COCO-style mAP (average over IoU 0.5 to 0.95)
    
    Returns:
        metrics dict with mAP values
    """
    model.eval()
    
    from torchvision.ops import box_iou
    
    all_predictions = []
    all_targets = []
    
    print("\n   Validating...")
    
    for images, targets in tqdm(data_loader, desc="   Val", leave=False):
        images = [img.to(device) for img in images]
        
        predictions = model(images)
        
        for pred, target in zip(predictions, targets):
            all_predictions.append({
                'boxes': pred['boxes'].cpu(),
                'labels': pred['labels'].cpu(),
                'scores': pred['scores'].cpu()
            })
            all_targets.append({
                'boxes': target['boxes'],
                'labels': target['labels']
            })
    
    # Calculate mAP@0.5
    aps = []
    
    for cls in range(1, NUM_CLASSES + 1):  # Skip background (0)
        cls_preds = []
        cls_targets = []
        
        for pred, target in zip(all_predictions, all_targets):
            # Get predictions for this class
            mask = pred['labels'] == cls
            if mask.sum() > 0:
                cls_preds.append({
                    'boxes': pred['boxes'][mask],
                    'scores': pred['scores'][mask]
                })
            else:
                cls_preds.append({'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0)})
            
            # Get targets for this class
            mask = target['labels'] == cls
            if mask.sum() > 0:
                cls_targets.append(target['boxes'][mask])
            else:
                cls_targets.append(torch.zeros(0, 4))
        
        # Calculate AP for this class
        ap = calculate_ap(cls_preds, cls_targets, iou_threshold=0.5)
        aps.append(ap)
    
    mAP50 = sum(aps) / len(aps) if aps else 0.0
    
    return {'mAP@0.5': mAP50}


def calculate_ap(predictions, targets, iou_threshold=0.5):
    """
    Calculate Average Precision for a single class
    """
    from torchvision.ops import box_iou
    
    all_scores = []
    all_matches = []
    total_positives = 0
    
    for pred, target in zip(predictions, targets):
        if len(target) > 0:
            total_positives += len(target)
        
        if len(pred['boxes']) == 0:
            continue
            
        if len(target) == 0:
            # All predictions are false positives
            for score in pred['scores']:
                all_scores.append(score.item())
                all_matches.append(0)
            continue
        
        # Calculate IoU between predictions and targets
        ious = box_iou(pred['boxes'], target)
        
        matched_gt = set()
        
        # Sort predictions by score
        sorted_idx = torch.argsort(pred['scores'], descending=True)
        
        for idx in sorted_idx:
            score = pred['scores'][idx].item()
            all_scores.append(score)
            
            # Find best matching GT
            best_iou, best_gt = ious[idx].max(0)
            
            if best_iou >= iou_threshold and best_gt.item() not in matched_gt:
                all_matches.append(1)  # True positive
                matched_gt.add(best_gt.item())
            else:
                all_matches.append(0)  # False positive
    
    if total_positives == 0 or len(all_matches) == 0:
        return 0.0
    
    # Sort by score
    sorted_idx = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    all_matches = [all_matches[i] for i in sorted_idx]
    
    # Calculate precision and recall
    tp = 0
    precisions = []
    recalls = []
    
    for match in all_matches:
        tp += match
        precision = tp / (len(precisions) + 1)
        recall = tp / total_positives
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = 0
        for i, r in enumerate(recalls):
            if r >= t:
                p = max(p, precisions[i])
        ap += p / 11
    
    return ap


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"    Checkpoint saved: {path}")


def train():
    """Main training function"""
    print("=" * 80)
    print("🏆 FASTER R-CNN - PAPER REPRODUCTION TRAINING")
    print("=" * 80)
    
    # Print config
    print_config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = create_model(pretrained=True)
    model.to(device)
    model_summary(model)
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    # Data loaders
    print("\n Loading data...")
    train_loader, val_loader = get_data_loaders()
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Training loop
    print(f"\n Starting training for {EPOCHS} epochs...")
    print("-" * 80)
    
    best_loss = float('inf')
    best_mAP = 0.0
    
    for epoch in range(EPOCHS):
        # Train one epoch
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Validate every 2 epochs or last epoch
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            metrics = validate(model, val_loader, device)
            mAP = metrics['mAP@0.5']
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | mAP@0.5: {mAP:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model based on mAP
            if mAP > best_mAP:
                best_mAP = mAP
                best_path = os.path.join(WORK_DIR, 'best.pth')
                save_checkpoint(model, optimizer, epoch + 1, avg_loss, best_path)
                print(f"   New best mAP: {best_mAP:.4f}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(WORK_DIR, f'epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, ckpt_path)
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, 'faster_rcnn_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n Final model saved: {final_path}")
    
    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE!")
    print("=" * 80)
    
    return final_path, model


if __name__ == "__main__":
    train()
