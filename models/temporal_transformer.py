import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You Need".
    Creates fixed position embeddings that help the model understand frame order.
    """
    
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create position encodings once and register as buffer (not trained)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use log scale for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input tensor."""
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer.
    
    Instead of mean pooling, the model LEARNS which frames are important.
    Suspicious frames (with deepfake artifacts) should get higher attention weights.
    This is much better than treating all frames equally.
    """
    
    def __init__(self, d_model):
        super().__init__()
        # Learnable query vector for attention
        self.attention_query = nn.Linear(d_model, 1)
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            padding_mask: (batch, seq_len) - True for padded positions
        
        Returns:
            pooled: (batch, d_model)
            attention_weights: (batch, seq_len) - for visualization
        """
        # Compute attention scores
        scores = self.attention_query(x).squeeze(-1)  # (batch, seq_len)
        
        # Mask padded positions with -inf before softmax
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Handle all-masked case (prevent NaN)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Weighted sum
        pooled = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # (batch, d_model)
        
        return pooled, attention_weights


class TemporalTransformer(nn.Module):
    """
    Deep Transformer for deepfake detection with attention pooling.
    
    Architecture improvements:
    1. 4 layers (deeper = learns hierarchical patterns)
    2. Pre-norm (stable gradients through deep layers)
    3. Attention pooling (learns which frames matter)
    """
    
    def __init__(
        self,
        feature_dim=1536,      # EfficientNet-B3 output dimension
        seq_len=64,            # Fixed sequence length
        d_model=256,           # Internal transformer dimension
        nhead=4,               # Number of attention heads
        num_layers=4,          # Deeper: 4 layers for hierarchical patterns
        dim_feedforward=512,   # FFN hidden dimension
        dropout=0.2,           # Dropout rate
        use_attention_pooling=True,  # Use attention pooling vs mean pooling
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_attention_pooling = use_attention_pooling
        
        # Project input features to transformer dimension
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),  # GELU throughout for consistency
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Deep Transformer encoder (4 layers, Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN: critical for deep models
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Attention pooling - learns which frames are important
        if use_attention_pooling:
            self.pooling = AttentionPooling(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Raw logits
        )
        
        # Store attention weights for visualization
        self.last_attention_weights = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, padding_mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim)
            padding_mask: Optional mask for padded positions (batch, seq_len)
        
        Returns:
            Tensor of shape (batch, 1) with raw logits
        """
        # Project to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through deep transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Apply final norm
        x = self.final_norm(x)
        
        # Pooling
        if self.use_attention_pooling:
            x, attention_weights = self.pooling(x, padding_mask)
            self.last_attention_weights = attention_weights  # Save for visualization
        else:
            # Fallback to masked mean pooling
            if padding_mask is not None:
                mask = ~padding_mask
                mask = mask.unsqueeze(-1).float()
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_weights(self):
        """Get the last attention weights for visualization."""
        return self.last_attention_weights
    
    def predict_proba(self, x, padding_mask=None):
        """Get probability predictions (for inference)."""
        logits = self.forward(x, padding_mask)
        return torch.sigmoid(logits)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    Two key benefits:
    1. α (alpha): Handles class imbalance - give more weight to minority class
    2. γ (gamma): Focuses on hard examples - down-weight easy predictions
    
    For deepfake detection:
    - Easy examples: clearly fake/real videos
    - Hard examples: subtle deepfakes that fool the model initially
    
    By focusing on hard examples, the model learns to detect subtle artifacts.
    """
    
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Balance factor for positive class
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (before sigmoid), shape (batch, 1)
            targets: Ground truth labels (0 or 1), shape (batch, 1)
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute binary cross entropy (without reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute focal weight
        # For positive examples (fake): p_t = probs
        # For negative examples (real): p_t = 1 - probs
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for class balance
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing Deep Temporal Transformer with Attention Pooling")
    print("=" * 60)
    
    model = TemporalTransformer(num_layers=4, use_attention_pooling=True)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch = torch.randn(4, 64, 1536)
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check attention weights
    attention = model.get_attention_weights()
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention sum per sample: {attention.sum(dim=1)}")  # Should be ~1.0
    
    # Test with padding mask
    mask = torch.zeros(4, 64, dtype=torch.bool)
    mask[:, 50:] = True
    output_masked = model(batch, padding_mask=mask)
    print(f"Output with mask shape: {output_masked.shape}")
    
    # Test Focal Loss
    print("\nTesting Focal Loss")
    print("-" * 40)
    focal_loss = FocalLoss(alpha=0.5, gamma=2.0)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = focal_loss(output, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    print("\nAll tests passed!")
