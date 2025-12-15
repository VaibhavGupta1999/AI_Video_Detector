import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.temporal_transformer import TemporalTransformer, FocalLoss
from datasets.video_sequence_dataset import VideoSequenceDataset, create_dataloader
from utils.augmentation import TemporalAugmentor
from utils.metrics import (
    compute_metrics, plot_confusion_matrix, save_predictions_csv,
    print_metrics_summary, MetricsTracker
)


# ============== Configuration ==============
def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    This makes experiments reproducible and traceable.
    """
    config = {
        # Default values if config.yaml doesn't exist
        'dataset': {
            'feature_dir': './Features',
            'sequence_length': 64,
            'feature_dim': 1536,
            'val_split': 0.2
        },
        'training': {
            'batch_size': 8,
            'epochs': 30,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'early_stop_patience': 7,
            'min_epochs': 10,           # Minimum epochs before early stopping
            'stratified_split': True    # Use stratified train/val split
        },
        'model': {
            'num_layers': 4,           # Deep: 4 layers for hierarchical patterns
            'num_heads': 4,
            'hidden_dim': 256,
            'feedforward_dim': 512,
            'dropout': 0.2,
            'use_attention_pooling': True  # Attention pooling learns which frames matter
        },
        'loss': {
            'type': 'focal',           # 'focal' or 'bce'
            'focal_alpha': 0.5,        # Balance factor for positive class
            'focal_gamma': 2.0         # Focusing parameter for hard examples
        },
        'augmentation': {
            'enabled': True,
            'frame_drop_prob': 0.05,
            'noise_std': 0.002,
            'jitter_prob': 0.3,
            'feature_dropout': 0.05
        },
        'output': {
            'dir': './outputs',
            'model_name': 'best_model.pth'
        }
    }
    
    # Try to load from YAML file
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Deep merge yaml_config into config
            for section, values in yaml_config.items():
                if section in config and isinstance(values, dict):
                    config[section].update(values)
                else:
                    config[section] = values
            
            print(f"Loaded config from: {config_path}")
        except ImportError:
            print("PyYAML not installed. Using default config. Install with: pip install pyyaml")
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
    
    return config


class Config:
    """Config object that wraps dictionary for backward compatibility."""
    def __init__(self, config_dict):
        # Flatten for easy access
        self.FEATURE_DIR = config_dict['dataset']['feature_dir']
        self.SEQ_LEN = config_dict['dataset']['sequence_length']
        self.FEATURE_DIM = config_dict['dataset']['feature_dim']
        self.VAL_SPLIT = config_dict['dataset']['val_split']
        
        self.BATCH_SIZE = config_dict['training']['batch_size']
        self.EPOCHS = config_dict['training']['epochs']
        self.LEARNING_RATE = config_dict['training']['learning_rate']
        self.WEIGHT_DECAY = config_dict['training']['weight_decay']
        self.GRADIENT_CLIP = config_dict['training']['grad_clip']
        self.EARLY_STOP_PATIENCE = config_dict['training']['early_stop_patience']
        self.MIN_EPOCHS = config_dict['training'].get('min_epochs', 10)
        self.STRATIFIED_SPLIT = config_dict['training'].get('stratified_split', True)
        
        self.NUM_LAYERS = config_dict['model']['num_layers']
        self.NHEAD = config_dict['model']['num_heads']
        self.D_MODEL = config_dict['model']['hidden_dim']
        self.DROPOUT = config_dict['model']['dropout']
        self.USE_ATTENTION_POOLING = config_dict['model'].get('use_attention_pooling', True)
        
        # Loss configuration
        loss_config = config_dict.get('loss', {})
        self.LOSS_TYPE = loss_config.get('type', 'focal')
        self.FOCAL_ALPHA = loss_config.get('focal_alpha', 0.5)
        self.FOCAL_GAMMA = loss_config.get('focal_gamma', 2.0)
        
        self.AUGMENT_TRAIN = config_dict['augmentation']['enabled']
        self.AUG_CONFIG = config_dict['augmentation']
        
        self.OUTPUT_DIR = config_dict['output']['dir']
        self.MODEL_SAVE_PATH = os.path.join(self.OUTPUT_DIR, config_dict['output']['model_name'])


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced data.
    Higher weight for minority class to balance loss contribution.
    """
    labels = np.array(labels)
    n_samples = len(labels)
    n_positive = labels.sum()
    n_negative = n_samples - n_positive
    
    if n_positive == 0 or n_negative == 0:
        return 1.0
    
    # Weight for positive class (fake)
    pos_weight = n_negative / n_positive
    return pos_weight


def train_one_epoch(model, train_loader, criterion, optimizer, device, gradient_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for features, labels, masks in pbar:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(features, padding_mask=masks)
        loss = criterion(logits, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Run validation and compute all metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, masks in val_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            masks = masks.to(device)
            
            logits = model(features, padding_mask=masks)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics, all_labels, all_preds, all_probs


def train(config, train_paths, train_labels, val_paths, val_labels, video_ids=None):
    """Main training function."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Setup augmentation
    augmentor = TemporalAugmentor() if config.AUGMENT_TRAIN else None
    
    # Create data loaders
    train_loader = create_dataloader(
        train_paths, train_labels,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=True,
        augment=config.AUGMENT_TRAIN,
        augment_fn=augmentor
    )
    
    val_loader = create_dataloader(
        val_paths, val_labels,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=False,
        augment=False
    )
    
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    # Create model with attention pooling
    model = TemporalTransformer(
        feature_dim=config.FEATURE_DIM,
        seq_len=config.SEQ_LEN,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_attention_pooling=config.USE_ATTENTION_POOLING
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Compute class weights for imbalanced data
    pos_weight = compute_class_weights(train_labels)
    print(f"Class weight (positive/fake): {pos_weight:.2f}")
    
    # Loss function - Focal Loss for hard example mining
    loss_type = getattr(config, 'LOSS_TYPE', 'focal')
    if loss_type == 'focal':
        focal_alpha = getattr(config, 'FOCAL_ALPHA', 0.5)
        focal_gamma = getattr(config, 'FOCAL_GAMMA', 2.0)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"Using Focal Loss (α={focal_alpha}, γ={focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        print("Using BCEWithLogitsLoss")
    
    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Metrics tracking
    metrics_tracker = MetricsTracker()
    best_f1 = 0.0
    patience_counter = 0
    
    # Training log
    log_path = os.path.join(config.OUTPUT_DIR, "training_logs.txt")
    with open(log_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Accuracy,Precision,Recall,F1,AUC,LR\n")
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.GRADIENT_CLIP
        )
        
        # Validate
        val_loss, val_metrics, y_true, y_pred, y_prob = validate(
            model, val_loader, criterion, device
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler based on F1 score
        scheduler.step(val_metrics['f1'])
        
        # Track metrics
        metrics_tracker.update(
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_metrics['accuracy'],
            val_precision=val_metrics['precision'],
            val_recall=val_metrics['recall'],
            val_f1=val_metrics['f1'],
            val_auc=val_metrics.get('auc', 0),
            learning_rate=current_lr
        )
        
        # Log to file
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},"
                   f"{val_metrics['accuracy']:.4f},{val_metrics['precision']:.4f},"
                   f"{val_metrics['recall']:.4f},{val_metrics['f1']:.4f},"
                   f"{val_metrics.get('auc', 0):.4f},{current_lr:.6f}\n")
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Acc: {val_metrics['accuracy']:.4f}, P: {val_metrics['precision']:.4f}, "
              f"R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': vars(config)
            }, config.MODEL_SAVE_PATH)
            print(f"  ** New best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping (only after minimum epochs)
        min_epochs = getattr(config, 'MIN_EPOCHS', 10)
        if epoch >= min_epochs and patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (after {min_epochs} minimum epochs)")
            break
    
    # Load best model for final evaluation
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    _, final_metrics, y_true, y_pred, y_prob = validate(model, val_loader, criterion, device)
    
    # Generate outputs
    print("\n" + "="*60)
    print("Generating Outputs")
    print("="*60)
    
    # Confusion matrix
    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Predictions CSV
    if video_ids is None:
        video_ids = [f"video_{i}" for i in range(len(y_true))]
    csv_path = os.path.join(config.OUTPUT_DIR, "sample_predictions.csv")
    save_predictions_csv(video_ids[-len(y_true):], y_true, y_pred, y_prob, csv_path)
    
    # Training curves
    curves_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")
    metrics_tracker.plot_training_curves(curves_path)
    
    # Print final metrics
    print_metrics_summary(y_true, y_pred, y_prob)
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    
    return model, final_metrics


import numpy as np

def _diagnose_frame_counts(feature_paths, labels):
    """
    Check for potential data leakage from frame count differences.
    If fake and real videos have very different frame counts, the model
    might learn to classify based on padding patterns instead of actual artifacts.
    """
    fake_counts = []
    real_counts = []
    
    for path, label in zip(feature_paths, labels):
        try:
            features = np.load(path)
            n_frames = features.shape[0]
            if label == 1:  # Fake
                fake_counts.append(n_frames)
            else:  # Real
                real_counts.append(n_frames)
        except Exception as e:
            continue
    
    if len(fake_counts) > 0 and len(real_counts) > 0:
        fake_avg = np.mean(fake_counts)
        real_avg = np.mean(real_counts)
        fake_std = np.std(fake_counts)
        real_std = np.std(real_counts)
        
        print(f"\n{'='*60}")
        print("FRAME COUNT ANALYSIS (Data Leakage Check)")
        print(f"{'='*60}")
        print(f"  Fake videos: {len(fake_counts)} samples, avg {fake_avg:.1f} frames (std: {fake_std:.1f})")
        print(f"  Real videos: {len(real_counts)} samples, avg {real_avg:.1f} frames (std: {real_std:.1f})")
        
        # Check for significant difference
        ratio = max(fake_avg, real_avg) / max(min(fake_avg, real_avg), 1)
        if ratio > 2.0:
            print(f"\n  ⚠️  WARNING: Large frame count difference (ratio: {ratio:.1f}x)")
            print(f"  ⚠️  The model may learn padding patterns instead of deepfake artifacts!")
            print(f"  ⚠️  Consider normalizing frame counts or using more diverse data.")
        else:
            print(f"\n  ✓  Frame count distribution looks balanced (ratio: {ratio:.1f}x)")
        print(f"{'='*60}\n")


def prepare_dataset_from_videos(video_dir, feature_dir):
    """Extract features from videos and prepare dataset."""
    from models.efficientnet_b3_extractor import EfficientNetB3Extractor, extract_features_from_video
    
    device = get_device()
    extractor = EfficientNetB3Extractor(device=device)
    
    feature_paths = []
    labels = []
    video_ids = []
    
    # Process Fake videos
    fake_dir = os.path.join(video_dir, "Fake")
    if os.path.exists(fake_dir):
        for filename in tqdm(os.listdir(fake_dir), desc="Extracting Fake"):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(fake_dir, filename)
                video_id = os.path.splitext(filename)[0]
                output_path = os.path.join(feature_dir, f"{video_id}.npy")
                
                try:
                    extract_features_from_video(video_path, extractor, output_path)
                    feature_paths.append(output_path)
                    labels.append(1)  # Fake
                    video_ids.append(video_id)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    # Process Real videos
    real_dir = os.path.join(video_dir, "Real")
    if os.path.exists(real_dir):
        for filename in tqdm(os.listdir(real_dir), desc="Extracting Real"):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(real_dir, filename)
                video_id = os.path.splitext(filename)[0]
                output_path = os.path.join(feature_dir, f"{video_id}.npy")
                
                try:
                    extract_features_from_video(video_path, extractor, output_path)
                    feature_paths.append(output_path)
                    labels.append(0)  # Real
                    video_ids.append(video_id)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    return feature_paths, labels, video_ids


def main():
    parser = argparse.ArgumentParser(description="Train Temporal Transformer for Deepfake Detection")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--video_dir', type=str, default='./Dataset',
                       help='Directory containing Fake/Real video subdirs')
    parser.add_argument('--feature_dir', type=str, default=None,
                       help='Directory to save/load extracted features (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract features from videos first')
    parser.add_argument('--val_split', type=float, default=None,
                       help='Validation split ratio (overrides config)')
    args = parser.parse_args()
    
    # Load config from YAML file
    config_dict = load_config(args.config)
    
    # Apply CLI overrides
    if args.feature_dir:
        config_dict['dataset']['feature_dir'] = args.feature_dir
    if args.output_dir:
        config_dict['output']['dir'] = args.output_dir
    if args.epochs:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size:
        config_dict['training']['batch_size'] = args.batch_size
    if args.lr:
        config_dict['training']['learning_rate'] = args.lr
    if args.val_split:
        config_dict['dataset']['val_split'] = args.val_split
    
    # Create config object
    config = Config(config_dict)
    
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save config alongside model for reproducibility
    config_save_path = os.path.join(config.OUTPUT_DIR, "training_config.yaml")
    try:
        import yaml
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Saved training config to: {config_save_path}")
    except ImportError:
        pass
    
    # Extract features if requested
    if args.extract_features:
        print("Extracting features from videos...")
        feature_paths, labels, video_ids = prepare_dataset_from_videos(
            args.video_dir, config.FEATURE_DIR
        )
    else:
        # Load existing features
        print("Loading existing features...")
        feature_paths = []
        labels = []
        video_ids = []
        
        for filename in os.listdir(config.FEATURE_DIR):
            if filename.endswith('.npy'):
                path = os.path.join(config.FEATURE_DIR, filename)
                feature_paths.append(path)
                video_id = filename.replace('.npy', '')
                video_ids.append(video_id)
                
                # Infer label from filename
                if 'hotshot' in video_id.lower():
                    labels.append(1)  # Fake
                elif 'msrvtt' in video_id.lower():
                    labels.append(0)  # Real
                else:
                    # Default: check if fake/real in name
                    labels.append(1 if 'fake' in video_id.lower() else 0)
    
    if len(feature_paths) == 0:
        print("No features found. Run with --extract_features flag first.")
        return
    
    print(f"Found {len(feature_paths)} video features")
    print(f"  Fake: {sum(labels)}, Real: {len(labels) - sum(labels)}")
    
    # Diagnostic: Check for frame count imbalance (potential data leakage)
    _diagnose_frame_counts(feature_paths, labels)
    
    # Split into train/val using stratified split for balanced classes
    if getattr(config, 'STRATIFIED_SPLIT', True):
        print("Using stratified train/val split for balanced classes")
        train_paths, val_paths, train_labels, val_labels, train_ids, val_video_ids = train_test_split(
            feature_paths, labels, video_ids,
            test_size=config.VAL_SPLIT,
            stratify=labels,
            random_state=42
        )
    else:
        # Fallback to random split
        np.random.seed(42)
        indices = np.random.permutation(len(feature_paths))
        split_idx = int(len(indices) * (1 - config.VAL_SPLIT))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_paths = [feature_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_paths = [feature_paths[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        val_video_ids = [video_ids[i] for i in val_indices]
    
    # Print split statistics
    print(f"\nTrain set: {len(train_paths)} samples (Fake: {sum(train_labels)}, Real: {len(train_labels) - sum(train_labels)})")
    print(f"Val set: {len(val_paths)} samples (Fake: {sum(val_labels)}, Real: {len(val_labels) - sum(val_labels)})")
    
    # Train
    train(config, train_paths, train_labels, val_paths, val_labels, val_video_ids)


if __name__ == "__main__":
    main()

