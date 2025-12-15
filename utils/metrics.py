"""
Metrics Utilities

Functions for computing classification metrics and generating
visualizations. These are crucial for evaluating deepfake detection
performance beyond simple accuracy.

Why these metrics matter for deepfake detection:
- Accuracy alone can be misleading with class imbalance
- False negatives (missing a fake) are often more costly than false positives
- F1 score balances precision and recall
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all relevant classification metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Optional prediction probabilities
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Can happen if only one class in y_true
            metrics['auc'] = 0.0
    
    return metrics


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Returns:
        Confusion matrix as numpy array (2x2)
        [[TN, FP], [FN, TP]]
    """
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['Real', 'Fake']):
    """
    Generate and save confusion matrix plot.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the PNG file
        class_names: Names for the classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix'
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    metrics_text = f"TN={tn}  FP={fp}\nFN={fn}  TP={tp}"
    ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes, 
            ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path):
    """
    Generate and save ROC curve plot.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        save_path: Path to save the PNG file
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")


def save_predictions_csv(video_ids, y_true, y_pred, y_prob, save_path):
    """
    Save predictions to CSV file.
    
    Args:
        video_ids: List of video identifiers
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        save_path: Path to save the CSV file
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'video_id': video_ids,
        'true_label': y_true,
        'predicted_label': y_pred,
        'confidence': y_prob,
        'correct': (np.array(y_true) == np.array(y_pred)).astype(int)
    })
    
    # Add label names
    df['true_class'] = df['true_label'].map({0: 'Real', 1: 'Fake'})
    df['predicted_class'] = df['predicted_label'].map({0: 'Real', 1: 'Fake'})
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Predictions saved to: {save_path}")


def print_metrics_summary(y_true, y_pred, y_prob=None):
    """Print a formatted summary of all metrics."""
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    print("\n" + "="*50)
    print("METRICS SUMMARY")
    print("="*50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:       {metrics['auc']:.4f}")
    print("="*50)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    return metrics


class MetricsTracker:
    """Track metrics across training epochs."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rate': []
        }
    
    def update(self, **kwargs):
        """Update metrics for current epoch."""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric='val_f1'):
        """Get the epoch with best value for given metric."""
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0
        return int(np.argmax(self.history[metric]))
    
    def save_to_file(self, save_path):
        """Save training history to file."""
        import json
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self, save_path):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', color='green')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision/Recall
        axes[1, 1].plot(self.history['val_precision'], label='Precision')
        axes[1, 1].plot(self.history['val_recall'], label='Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    # Quick test
    print("Metrics test")
    
    # Create dummy predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.4, 0.2, 0.7, 0.85, 0.55])
    
    # Test metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"Metrics: {metrics}")
    
    # Test confusion matrix
    print_metrics_summary(y_true, y_pred, y_prob)
    
    print("All tests passed!")
