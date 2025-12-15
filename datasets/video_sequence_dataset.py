import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VideoSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading video feature sequences.
    
    Each sample is a tuple of (features, label) where:
    - features: tensor of shape (seq_len, feature_dim)
    - label: 0 for real, 1 for fake
    """
    
    def __init__(
        self,
        feature_paths,
        labels,
        seq_len=64,
        feature_dim=1536,
        augment=False,
        augment_fn=None
    ):
        """
        Args:
            feature_paths: List of paths to .npy feature files
            labels: List of labels (0=real, 1=fake)
            seq_len: Fixed sequence length to pad/truncate to
            feature_dim: Expected feature dimension
            augment: Whether to apply augmentation
            augment_fn: Optional custom augmentation function
        """
        self.feature_paths = feature_paths
        self.labels = labels
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.augment = augment
        self.augment_fn = augment_fn
        
        assert len(feature_paths) == len(labels), "Paths and labels must have same length"
    
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        path = self.feature_paths[idx]
        label = self.labels[idx]
        
        # Load features
        try:
            features = np.load(path).astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            features = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        
        # Apply augmentation if enabled
        if self.augment and self.augment_fn is not None:
            features = self.augment_fn(features)
        
        # Pad or truncate to fixed length
        features, mask = self._pad_or_truncate(features)
        
        # Convert to tensors
        features = torch.from_numpy(features)
        label = torch.tensor(label, dtype=torch.float32)
        mask = torch.from_numpy(mask)
        
        return features, label, mask
    
    def _pad_or_truncate(self, arr):
        """
        Pad or truncate sequence to fixed length.
        Returns (sequence, padding_mask) where mask is True for padded positions.
        
        When augmenting (training), uses random padding position to prevent
        the model from learning padding patterns as a proxy for the label.
        """
        n = arr.shape[0]
        mask = np.zeros(self.seq_len, dtype=bool)
        
        if n == self.seq_len:
            return arr, mask
        
        if n < self.seq_len:
            # Pad with zeros - use random position during training
            padded = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
            
            if self.augment:
                # Random start offset during training to prevent pattern learning
                max_offset = self.seq_len - n
                offset = np.random.randint(0, max_offset + 1)
            else:
                # Center padding during evaluation for consistency
                offset = (self.seq_len - n) // 2
            
            padded[offset:offset+n] = arr
            mask[:offset] = True  # Before content
            mask[offset+n:] = True  # After content
            arr = padded
        else:
            # Uniform sampling to reduce length
            indices = np.linspace(0, n - 1, self.seq_len).astype(int)
            arr = arr[indices]
        
        return arr, mask


def create_dataloader(
    feature_paths,
    labels,
    batch_size=16,
    seq_len=64,
    shuffle=True,
    augment=False,
    augment_fn=None,
    num_workers=0
):
    """
    Create a DataLoader for video sequence data.
    
    Args:
        feature_paths: List of paths to .npy files
        labels: List of labels
        batch_size: Batch size
        seq_len: Sequence length
        shuffle: Whether to shuffle data
        augment: Whether to augment
        augment_fn: Augmentation function
        num_workers: Number of workers for loading
    
    Returns:
        DataLoader instance
    """
    dataset = VideoSequenceDataset(
        feature_paths=feature_paths,
        labels=labels,
        seq_len=seq_len,
        augment=augment,
        augment_fn=augment_fn
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return loader


def load_dataset_from_directory(feature_dir, metadata_csv=None):
    """
    Load dataset from a directory of .npy feature files.
    
    If metadata_csv is provided, uses it for labels.
    Otherwise, infers labels from subdirectory structure (Real/Fake).
    
    Returns:
        Tuple of (feature_paths, labels, video_ids)
    """
    import pandas as pd
    
    feature_paths = []
    labels = []
    video_ids = []
    
    if metadata_csv and os.path.exists(metadata_csv):
        # Load from CSV
        df = pd.read_csv(metadata_csv)
        for _, row in df.iterrows():
            path = row.get('feature_path', '')
            if os.path.exists(path):
                feature_paths.append(path)
                # Handle various label formats
                label = row.get('label', row.get('label_int', 0))
                if isinstance(label, str):
                    label = 1 if label.lower() == 'fake' else 0
                labels.append(int(label))
                video_ids.append(row.get('video_id', os.path.basename(path).replace('.npy', '')))
    else:
        # Infer from directory structure
        for filename in os.listdir(feature_dir):
            if filename.endswith('.npy'):
                path = os.path.join(feature_dir, filename)
                feature_paths.append(path)
                video_id = filename.replace('.npy', '')
                video_ids.append(video_id)
                
                # Infer label from filename or parent directory
                if 'fake' in video_id.lower() or 'hotshot' in video_id.lower():
                    labels.append(1)
                else:
                    labels.append(0)
    
    return feature_paths, labels, video_ids


if __name__ == "__main__":
    # Quick test
    print("VideoSequenceDataset test")
    
    # Create dummy data
    os.makedirs("./test_features", exist_ok=True)
    for i in range(10):
        features = np.random.randn(np.random.randint(30, 100), 1536).astype(np.float32)
        np.save(f"./test_features/video_{i}.npy", features)
    
    paths = [f"./test_features/video_{i}.npy" for i in range(10)]
    labels = [i % 2 for i in range(10)]
    
    dataset = VideoSequenceDataset(paths, labels, seq_len=64)
    print(f"Dataset size: {len(dataset)}")
    
    features, label, mask = dataset[0]
    print(f"Features shape: {features.shape}")
    print(f"Label: {label}")
    print(f"Mask shape: {mask.shape}")
    
    loader = create_dataloader(paths, labels, batch_size=4)
    for batch_features, batch_labels, batch_masks in loader:
        print(f"Batch features: {batch_features.shape}")
        print(f"Batch labels: {batch_labels.shape}")
        print(f"Batch masks: {batch_masks.shape}")
        break
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_features")
    print("Test passed!")
