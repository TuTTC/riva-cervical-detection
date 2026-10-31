# =============================================================================
# FASTER R-CNN PAPER - MAIN ENTRY POINT
# =============================================================================
# Run this file to train and generate submission
# =============================================================================

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import print_config, OUTPUT_DIR
from train import train
from inference import generate_submission


def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN Paper Reproduction')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['train', 'inference', 'all'],
                        help='Mode: train, inference, or all')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for inference')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" FASTER R-CNN - PAPER REPRODUCTION")
    print("=" * 80)
    
    print_config()
    
    if args.mode in ['train', 'all']:
        print("\n PHASE 1: TRAINING")
        print("-" * 80)
        model_path, model = train()
    else:
        model = None
        model_path = args.checkpoint or os.path.join(OUTPUT_DIR, 'faster_rcnn_final.pth')
    
    if args.mode in ['inference', 'all']:
        print("\n PHASE 2: INFERENCE")
        print("-" * 80)
        generate_submission(model, model_path)
    
    print("\n" + "=" * 80)
    print(" COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
