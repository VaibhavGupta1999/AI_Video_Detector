"""
Feature Extraction Script

Extract EfficientNet-B3 features from all videos in a directory.
Saves features as .npy files for use with the temporal model.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet_b3_extractor import EfficientNetB3Extractor, extract_features_from_video


def main():
    parser = argparse.ArgumentParser(description="Extract EfficientNet-B3 features from videos")
    parser.add_argument('--video_dir', type=str, default='./Dataset',
                       help='Directory containing videos (with Fake/Real subdirs)')
    parser.add_argument('--output_dir', type=str, default='./Features',
                       help='Directory to save extracted features')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to extract per video')
    args = parser.parse_args()
    
    # Setup
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load extractor
    print("Loading EfficientNet-B3 extractor...")
    extractor = EfficientNetB3Extractor(device=device)
    
    # Find all videos
    video_files = []
    
    # Check for Fake/Real subdirectories
    for subdir in ['Fake', 'Real']:
        subdir_path = os.path.join(args.video_dir, subdir)
        if os.path.exists(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append({
                        'path': os.path.join(subdir_path, filename),
                        'label': subdir.lower()
                    })
    
    # Also check root directory
    for filename in os.listdir(args.video_dir):
        filepath = os.path.join(args.video_dir, filename)
        if os.path.isfile(filepath) and filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Infer label from filename
            label = 'fake' if 'fake' in filename.lower() else 'real'
            video_files.append({
                'path': filepath,
                'label': label
            })
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process videos
    metadata = []
    
    for video_info in tqdm(video_files, desc="Extracting features"):
        video_path = video_info['path']
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(args.output_dir, f"{video_id}.npy")
        
        try:
            n_frames = extract_features_from_video(
                video_path, extractor, output_path, max_frames=args.max_frames
            )
            metadata.append({
                'video_id': video_id,
                'n_frames': n_frames,
                'feature_path': output_path,
                'label': video_info['label']
            })
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
    
    # Save metadata
    if metadata:
        import pandas as pd
        meta_path = os.path.join(args.output_dir, "features_metadata.csv")
        pd.DataFrame(metadata).to_csv(meta_path, index=False)
        print(f"\nMetadata saved to: {meta_path}")
    
    print(f"\nExtracted features for {len(metadata)} videos")
    print(f"Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
