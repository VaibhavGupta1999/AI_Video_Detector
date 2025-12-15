"""
Temporal Augmentation Utilities

Safe augmentations for video feature sequences that won't break
the temporal structure. These are applied during training to
improve model generalization.

Key design decisions:
- Frame drop rate is capped at 5% to preserve temporal continuity
- At least one frame always survives
- Gaussian noise is very light to avoid destroying learned features
"""

import numpy as np


def random_frame_drop(features, drop_rate=0.05, min_frames=1):
    """
    Randomly drop frames from a sequence.
    
    This simulates missing frames or variable-rate video sampling.
    We keep drop rate low (<=5%) to preserve temporal structure.
    
    Args:
        features: numpy array of shape (n_frames, feature_dim)
        drop_rate: Maximum fraction of frames to drop
        min_frames: Minimum number of frames to keep
    
    Returns:
        Augmented features array
    """
    n_frames = features.shape[0]
    
    if n_frames <= min_frames:
        return features
    
    # Create drop mask - True means keep the frame
    keep_mask = np.random.rand(n_frames) > drop_rate
    
    # Ensure at least min_frames survive
    if keep_mask.sum() < min_frames:
        # Randomly select min_frames indices to keep
        keep_indices = np.random.choice(n_frames, size=min_frames, replace=False)
        keep_mask = np.zeros(n_frames, dtype=bool)
        keep_mask[keep_indices] = True
    
    return features[keep_mask]


def add_gaussian_noise(features, std=0.002):
    """
    Add light Gaussian noise to features.
    
    This acts as a regularizer and helps the model be robust to
    small variations in the feature extraction. The standard
    deviation is very low to avoid destroying the signal.
    
    Args:
        features: numpy array of shape (n_frames, feature_dim)
        std: Standard deviation of noise
    
    Returns:
        Noisy features array
    """
    noise = np.random.normal(0, std, features.shape).astype(np.float32)
    return features + noise


def temporal_jitter(features, max_shift=2):
    """
    Apply small temporal jitter by shifting frame order slightly.
    
    This helps the model be robust to small temporal misalignments.
    Only adjacent frames are swapped to preserve overall structure.
    
    Args:
        features: numpy array of shape (n_frames, feature_dim)
        max_shift: Maximum number of positions to shift
    
    Returns:
        Jittered features array
    """
    n_frames = features.shape[0]
    if n_frames < 4:
        return features
    
    result = features.copy()
    
    # Randomly swap some adjacent frames
    n_swaps = np.random.randint(0, min(max_shift + 1, n_frames // 4))
    for _ in range(n_swaps):
        idx = np.random.randint(0, n_frames - 1)
        result[idx], result[idx + 1] = result[idx + 1].copy(), result[idx].copy()
    
    return result


def feature_dropout(features, drop_rate=0.1):
    """
    Randomly zero out some feature dimensions.
    
    This is similar to dropout but applied to input features.
    Helps prevent overfitting to specific features.
    
    Args:
        features: numpy array of shape (n_frames, feature_dim)
        drop_rate: Fraction of features to zero out
    
    Returns:
        Features with some dimensions zeroed
    """
    mask = np.random.rand(*features.shape) > drop_rate
    return features * mask.astype(np.float32)


def temporal_augment(features, config=None):
    """
    Apply all temporal augmentations to a feature sequence.
    
    This is the main augmentation function used during training.
    Augmentations are applied with some probability to add variety.
    
    Args:
        features: numpy array of shape (n_frames, feature_dim)
        config: Optional dict with augmentation parameters
    
    Returns:
        Augmented features array
    """
    if config is None:
        config = {
            'frame_drop_rate': 0.05,
            'noise_std': 0.002,
            'jitter_prob': 0.3,
            'feature_dropout_rate': 0.05
        }
    
    # Frame dropping - always apply (with low rate)
    features = random_frame_drop(
        features, 
        drop_rate=config.get('frame_drop_rate', 0.05),
        min_frames=1
    )
    
    # Gaussian noise - always apply (very light)
    features = add_gaussian_noise(
        features,
        std=config.get('noise_std', 0.002)
    )
    
    # Temporal jitter - apply with probability
    if np.random.rand() < config.get('jitter_prob', 0.3):
        features = temporal_jitter(features, max_shift=2)
    
    # Feature dropout - apply to add regularization
    features = feature_dropout(
        features,
        drop_rate=config.get('feature_dropout_rate', 0.05)
    )
    
    return features.astype(np.float32)


# Create a simple callable for use with the dataset
class TemporalAugmentor:
    """Callable class for temporal augmentation."""
    
    def __init__(self, **kwargs):
        self.config = {
            'frame_drop_rate': kwargs.get('frame_drop_rate', 0.05),
            'noise_std': kwargs.get('noise_std', 0.002),
            'jitter_prob': kwargs.get('jitter_prob', 0.3),
            'feature_dropout_rate': kwargs.get('feature_dropout_rate', 0.05)
        }
    
    def __call__(self, features):
        return temporal_augment(features, self.config)


if __name__ == "__main__":
    # Quick test
    print("Augmentation test")
    
    # Create dummy features
    features = np.random.randn(50, 1536).astype(np.float32)
    print(f"Original shape: {features.shape}")
    
    # Test frame drop
    dropped = random_frame_drop(features, drop_rate=0.1)
    print(f"After frame drop: {dropped.shape}")
    
    # Test noise
    noisy = add_gaussian_noise(features, std=0.002)
    noise_level = np.abs(noisy - features).mean()
    print(f"Noise level: {noise_level:.6f}")
    
    # Test full augmentation
    augmentor = TemporalAugmentor()
    augmented = augmentor(features)
    print(f"After full augmentation: {augmented.shape}")
    
    print("All tests passed!")
