import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


class EfficientNetB3Extractor(nn.Module):
    """
    Frozen EfficientNet-B3 backbone for frame feature extraction.
    Outputs 1536-dimensional feature vectors per frame.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load pretrained EfficientNet-B3
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        base_model = models.efficientnet_b3(weights=weights)
        
        # Remove classifier head - keep only feature extractor
        # EfficientNet structure: features -> avgpool -> classifier
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Freeze all parameters - we don't want to train the backbone
        for param in self.parameters():
            param.requires_grad = False
        
        self.to(device)
        self.eval()  # always in eval mode
        
        # ImageNet normalization - must match training preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),  # B3 default input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x):
        """Forward pass through backbone."""
        with torch.no_grad():
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # (batch, 1536)
        return x
    
    def extract_from_image(self, img_path):
        """Extract features from a single image file."""
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return self.forward(tensor).cpu().numpy().squeeze()
    
    def extract_from_frames(self, frame_paths, batch_size=32):
        """
        Extract features from a list of frame paths.
        Returns numpy array of shape (n_frames, 1536).
        """
        all_features = []
        
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    tensor = self.transform(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Warning: failed to load {path}: {e}")
                    # Use zero tensor as fallback
                    batch_tensors.append(torch.zeros(3, 300, 300))
            
            batch = torch.stack(batch_tensors).to(self.device)
            features = self.forward(batch).cpu().numpy()
            all_features.append(features)
        
        return np.concatenate(all_features, axis=0).astype(np.float32)


def extract_features_from_video(video_path, extractor, output_path, max_frames=None):
    """
    Extract frames from video and save features as .npy file.
    
    Args:
        video_path: Path to input video file
        extractor: EfficientNetB3Extractor instance
        output_path: Where to save the .npy features
        max_frames: Optional limit on frames to extract
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from: {video_path}")
    
    # Extract features for all frames
    all_features = []
    
    for i in range(0, len(frames), 32):
        batch_frames = frames[i:i + 32]
        batch_tensors = []
        
        for frame in batch_frames:
            img = Image.fromarray(frame)
            tensor = extractor.transform(img)
            batch_tensors.append(tensor)
        
        batch = torch.stack(batch_tensors).to(extractor.device)
        features = extractor.forward(batch).cpu().numpy()
        all_features.append(features)
    
    features = np.concatenate(all_features, axis=0).astype(np.float32)
    
    # Save features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)
    
    return features.shape[0]


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    extractor = EfficientNetB3Extractor(device=device)
    print("EfficientNet-B3 extractor loaded successfully")
    print(f"Output feature dimension: 1536")
