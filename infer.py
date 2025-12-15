"""
Inference Script for Deepfake Detection

Load a trained model and run predictions on new videos or features.
Outputs predictions with confidence scores.
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.temporal_transformer import TemporalTransformer
from models.efficientnet_b3_extractor import EfficientNetB3Extractor, extract_features_from_video
from utils.metrics import save_predictions_csv


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    
    model = TemporalTransformer(
        feature_dim=config.get('FEATURE_DIM', 1536),
        seq_len=config.get('SEQ_LEN', 64),
        d_model=config.get('D_MODEL', 256),
        nhead=config.get('NHEAD', 4),
        num_layers=config.get('NUM_LAYERS', 3),
        dropout=config.get('DROPOUT', 0.2)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"Best F1 score: {checkpoint.get('best_f1', '?'):.4f}")
    
    return model


def pad_or_truncate(features, seq_len=64):
    """Pad or truncate features to fixed length."""
    n = features.shape[0]
    feature_dim = features.shape[1]
    mask = np.zeros(seq_len, dtype=bool)
    
    if n == seq_len:
        return features, mask
    
    if n < seq_len:
        pad = np.zeros((seq_len - n, feature_dim), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
        mask[n:] = True
    else:
        indices = np.linspace(0, n - 1, seq_len).astype(int)
        features = features[indices]
    
    return features, mask


def predict_from_features(model, feature_path, device, seq_len=64):
    """Make prediction from pre-extracted features."""
    features = np.load(feature_path).astype(np.float32)
    features, mask = pad_or_truncate(features, seq_len)
    
    # Convert to tensors
    features = torch.from_numpy(features).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features, padding_mask=mask)
        prob = torch.sigmoid(logits).item()
    
    pred_label = 1 if prob > 0.5 else 0
    pred_class = "Fake" if pred_label == 1 else "Real"
    
    return pred_label, pred_class, prob


def predict_from_video(model, extractor, video_path, device, seq_len=64, temp_dir="./temp_features"):
    """Make prediction directly from video file."""
    os.makedirs(temp_dir, exist_ok=True)
    
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    feature_path = os.path.join(temp_dir, f"{video_id}.npy")
    
    # Extract features
    try:
        extract_features_from_video(video_path, extractor, feature_path)
    except Exception as e:
        print(f"Error extracting features from {video_path}: {e}")
        return None, None, None
    
    # Predict
    return predict_from_features(model, feature_path, device, seq_len)


def main():
    parser = argparse.ArgumentParser(description="Run inference on videos or features")
    parser.add_argument('--model_path', type=str, default='./outputs/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing .npy features or videos')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Single video or feature file to process')
    parser.add_argument('--output_csv', type=str, default='./outputs/predictions.csv',
                       help='Path to save predictions CSV')
    parser.add_argument('--from_video', action='store_true',
                       help='Input is video files (not pre-extracted features)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        return
    
    model = load_model(args.model_path, device)
    
    # Load feature extractor if processing videos
    extractor = None
    if args.from_video:
        print("Loading EfficientNet-B3 extractor...")
        extractor = EfficientNetB3Extractor(device=device)
    
    # Collect input files
    input_files = []
    if args.input_file:
        input_files = [args.input_file]
    elif args.input_dir:
        extension = ('.mp4', '.avi', '.mov') if args.from_video else ('.npy',)
        for filename in os.listdir(args.input_dir):
            if filename.endswith(extension):
                input_files.append(os.path.join(args.input_dir, filename))
    
    if not input_files:
        print("No input files found.")
        return
    
    print(f"Processing {len(input_files)} files...")
    
    # Run predictions
    video_ids = []
    predictions = []
    confidences = []
    pred_classes = []
    
    for filepath in tqdm(input_files, desc="Predicting"):
        video_id = os.path.splitext(os.path.basename(filepath))[0]
        
        if args.from_video:
            pred_label, pred_class, prob = predict_from_video(
                model, extractor, filepath, device
            )
        else:
            pred_label, pred_class, prob = predict_from_features(
                model, filepath, device
            )
        
        if pred_label is not None:
            video_ids.append(video_id)
            predictions.append(pred_label)
            pred_classes.append(pred_class)
            confidences.append(prob)
            
            print(f"  {video_id}: {pred_class} (confidence: {prob:.3f})")
    
    # Save results
    if video_ids:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        
        import pandas as pd
        df = pd.DataFrame({
            'video_id': video_ids,
            'predicted_label': predictions,
            'predicted_class': pred_classes,
            'confidence': confidences
        })
        df.to_csv(args.output_csv, index=False)
        print(f"\nPredictions saved to: {args.output_csv}")
        
        # Summary
        n_fake = sum(predictions)
        n_real = len(predictions) - n_fake
        print(f"\nSummary: {n_fake} Fake, {n_real} Real out of {len(predictions)} videos")


if __name__ == "__main__":
    main()
