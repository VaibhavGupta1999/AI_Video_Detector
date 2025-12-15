# Technical Report: Deepfake Video Detection Pipeline

## 1. Problem Understanding

Deepfake detection is a binary classification task: given a video, determine if it's real or synthetically generated (fake). The challenge is that modern deepfakes are visually convincing per-frame — the tells are in the **temporal domain** (inconsistencies across frames, unnatural movements, etc.).

This means any effective solution must model the sequence of frames, not just analyze individual frames.

---

## 2. My Approach

I built a two-stage pipeline:

### Stage 1: Feature Extraction
Use a pretrained EfficientNet-B3 (frozen) to extract visual features from each frame. This captures "what's in the frame" without needing to train on image classification.

### Stage 2: Temporal Modeling
Use a Transformer encoder to model relationships between frames. This captures "how frames relate to each other" — the key signal for detecting deepfakes.

---

## 3. Design Decisions

### 3.1 Why Freeze the Backbone?

With only 50 videos, fine-tuning EfficientNet would overfit. The ImageNet features are already excellent for detecting visual artifacts. I use them as-is.

### 3.2 Why a 4-Layer Transformer?

| Layers | Learning Capability |
|--------|---------------------|
| 1-2 | Local patterns only |
| 3-4 | Hierarchical + global patterns |

4 layers is deep enough to learn complex temporal dependencies while being computationally efficient.

### 3.3 Why Pre-Norm Architecture?

Standard Transformers use post-norm (LayerNorm after attention). For deep models, this causes gradient issues. Pre-norm (LayerNorm before attention) provides:
- Stable gradients through all layers
- Faster convergence
- Better final accuracy

### 3.4 Why Attention Pooling?

Mean pooling treats all frames equally. But deepfake artifacts appear in **specific frames**, not uniformly. Attention pooling lets the model:
- Learn which frames are suspicious
- Give higher weight to artifact-containing frames
- Ignore irrelevant frames (padding, static scenes)

### 3.5 Why Focal Loss?

Standard BCE loss treats all samples equally. Focal Loss adds two benefits:

1. **α (alpha=0.5):** Balances gradient contributions from each class
2. **γ (gamma=2.0):** Down-weights easy examples, focuses on subtle deepfakes

This forces the model to learn fine-grained artifacts instead of relying on obvious patterns.

---

## 4. Training Configuration

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Batch size | 8 | Balance between gradient stability and memory |
| Learning rate | 1e-4 | Standard for fine-tuning |
| Optimizer | AdamW | Weight decay for regularization |
| Gradient clipping | 1.0 | Prevent exploding gradients |
| Early stopping | patience=7 | Stop if F1 doesn't improve |

---

## 5. Results

Training on 50 videos (25 fake, 25 real):

| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| F1 Score | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |

**Note:** Perfect scores on such a small dataset are expected. The model effectively memorizes the patterns. Real-world evaluation would require testing on larger, unseen datasets.

---

## 6. Limitations

1. **Small Dataset:** 50 videos is insufficient for true generalization
2. **No Face Detection:** I extract features from full frames, not cropped faces
3. **Single Domain:** Model is trained on specific deepfake types

---

## 7. Future Improvements

1. **Face tracking:** Crop and align faces for better feature extraction
2. **Larger dataset:** Train on FaceForensics++, DFDC
3. **Multi-scale features:** Use features from multiple EfficientNet layers
4. **Attention visualization:** Show which frames the model focuses on

---

## 8. Conclusion

I built a deepfake detection pipeline with:
- **4-layer Transformer** for hierarchical temporal learning
- **Attention pooling** to focus on suspicious frames
- **Focal loss** for hard example mining
- **Config-driven training** for reproducibility

The architecture is designed for accuracy and stability, not over-engineering. Each component has a clear purpose and justifiable improvement over simpler alternatives.
