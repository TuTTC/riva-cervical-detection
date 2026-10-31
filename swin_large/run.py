# =============================================================================
# SWIN TRANSFORMER LARGE - RUN SCRIPT
# =============================================================================
# Main entry point for training and inference
# =============================================================================

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='Swin Transformer Large for Cervical Cancer Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run complete pipeline (train + inference)
  python run.py --mode all
  
  # Training only
  python run.py --mode train
  
  # Inference only (requires trained checkpoint)
  python run.py --mode inference --checkpoint path/to/checkpoint.pth
  
  # Inference with TTA
  python run.py --mode inference --tta
  
  # Inference with TTA + LightGBM fusion
  python run.py --mode inference --tta --fusion
  
  # Show configuration
  python run.py --mode config
'''
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all',
        choices=['all', 'train', 'inference', 'config', 'data'],
        help='Mode to run: all, train, inference, config, or data'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for inference mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (cuda:0 or cpu)'
    )
    
    # TTA options
    parser.add_argument('--tta', action='store_true',
                       help='Enable Test Time Augmentation')
    parser.add_argument('--tta-scales', type=float, nargs='+',
                       default=[1.0],
                       help='Scales for TTA (e.g., 0.8 1.0 1.2)')
    
    # LightGBM fusion options
    parser.add_argument('--fusion', action='store_true',
                       help='Enable LightGBM feature fusion')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SWIN TRANSFORMER LARGE - BASELINE")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    if args.tta:
        print(f"TTA: ENABLED (scales: {args.tta_scales})")
    if args.fusion:
        print("LightGBM Fusion: ENABLED")
    print("=" * 80)
    
    if args.mode == 'config':
        # Just show configuration
        from config import print_config
        print_config()
        
    elif args.mode == 'data':
        # Prepare data only
        from dataset import convert_to_coco_format
        from config import (
            OUTPUT_DIR, TRAIN_CSV, VAL_CSV, 
            TRAIN_IMG_DIR, VAL_IMG_DIR
        )
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        convert_to_coco_format(
            TRAIN_CSV, TRAIN_IMG_DIR,
            os.path.join(OUTPUT_DIR, 'annotations_train.json'),
            split='train'
        )
        
        convert_to_coco_format(
            VAL_CSV, VAL_IMG_DIR,
            os.path.join(OUTPUT_DIR, 'annotations_val.json'),
            split='val'
        )
        
    elif args.mode == 'train':
        # Training only
        from train import main as train_main
        train_main()
        
    elif args.mode == 'inference':
        # Inference only
        inference_args = ['--device', args.device]
        
        if args.checkpoint:
            inference_args.extend(['--checkpoint', args.checkpoint])
        
        if args.tta:
            inference_args.append('--tta')
            if args.tta_scales != [1.0]:
                inference_args.extend(['--tta-scales'] + [str(s) for s in args.tta_scales])
        
        if args.fusion:
            inference_args.append('--fusion')
        
        # Add to sys.argv for argparse in inference.py
        sys.argv = [sys.argv[0]] + inference_args
        
        from inference import main as inference_main
        inference_main()
        
    elif args.mode == 'all':
        # Complete pipeline
        from train import main as train_main
        from config import WORK_DIR
        
        # Train
        checkpoint = train_main()
        
        # Inference
        if checkpoint:
            inference_args = ['--checkpoint', checkpoint, '--device', args.device]
            
            if args.tta:
                inference_args.append('--tta')
                if args.tta_scales != [1.0]:
                    inference_args.extend(['--tta-scales'] + [str(s) for s in args.tta_scales])
            
            if args.fusion:
                inference_args.append('--fusion')
            
            sys.argv = [sys.argv[0]] + inference_args
            
            from inference import main as inference_main
            inference_main()
        else:
            print("Warning: No checkpoint found after training!")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
