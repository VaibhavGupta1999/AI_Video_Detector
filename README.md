# Deepfake Video Detection Pipeline

A PyTorch-based deepfake detection system using EfficientNet-B3 for feature extraction and a deep Transformer encoder for temporal modeling.

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Dataset Setup](#dataset-setup)
3. [Training Process](#training-process)
4. [Inference](#inference)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [Model Architecture](#model-architecture)

---

## 🔧 Installation

### Step 1: Clone/Download the Project

```bash
cd Video\ ML\ Codebase
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy, pandas, scikit-learn
- matplotlib, opencv-python
- pyyaml, tqdm, pillow

---

## 📁 Dataset Setup

### Expected Dataset Structure

```
Dataset/
├── Fake/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── Real/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

Place your videos in:
- `Dataset/Fake/` — Deepfake videos (label = 1)
- `Dataset/Real/` — Real videos (label = 0)

Supported formats: `.mp4`, `.avi`, `.mov`

### ⚠️ Known Dataset Issues & Requirements

> **Important:** For reliable training, your dataset should meet these requirements:

| Requirement | Why It Matters |
|-------------|----------------|
| **Balanced frame counts** | Fake and real videos should have similar frame counts. Large differences cause data leakage (model learns padding patterns, not deepfake artifacts). |
| **Sufficient samples** | Minimum 100+ videos per class recommended. Small datasets (50 samples) lead to overfitting and 100% training accuracy that won't generalize. |
| **Diverse sources** | Videos should come from multiple sources to prevent the model from learning source-specific patterns. |
| **Stratified splitting** | Train/val split should maintain class balance (enabled by default with `stratified_split: true`). |

**Current Dataset Status:**
- Fake videos: 25 samples, avg 8 frames each (hotshot generated)
- Real videos: 25 samples, avg 32-92 frames each (msrvtt dataset)

The training pipeline includes automatic diagnostics that will warn you about frame count imbalance.

---

## 🚀 Training Process

### Option 1: Full Pipeline (Recommended)

**Step 1: Extract Features**

```bash
python extract_features.py --video_dir ./Dataset --output_dir ./Features
```

This will:
- Load each video
- Extract EfficientNet-B3 features (1536-dim per frame)
- Save as `.npy` files in `./Features/`

**Step 2: Train the Model**

```bash
python train_temporal.py
```

This will:
- Load features from `./Features/`
- Train the 4-layer Transformer model
- Save best model to `./outputs/best_model.pth`
- Generate confusion matrix and predictions

### Option 2: Quick Training (Features Already Extracted)

If you already have `.npy` feature files:

```bash
python train_temporal.py --feature_dir ./Features
```

### Option 3: Custom Configuration

```bash
python train_temporal.py --epochs 50 --batch_size 16 --lr 0.0001
```

### Training Output

After training, you'll find in `./outputs/`:

| File | Description |
|------|-------------|
| `best_model.pth` | Trained model checkpoint |
| `confusion_matrix.png` | Confusion matrix visualization |
| `sample_predictions.csv` | Predictions with confidence scores |
| `training_logs.txt` | Per-epoch metrics |
| `training_curves.png` | Loss and accuracy plots |
| `training_config.yaml` | Saved configuration |

---

## 🔮 Inference

### On Pre-extracted Features

```bash
python infer.py --input_dir ./Features --model_path ./outputs/best_model.pth
```

### On New Videos

```bash
python infer.py --input_path ./new_video.mp4 --model_path ./outputs/best_model.pth
```

### Output

```
Video: test_video.mp4
Prediction: FAKE
Confidence: 0.95
```

---

## ⚙️ Configuration

All hyperparameters are in `config.yaml`:

```yaml
# Model Architecture
model:
  num_layers: 4              # Transformer layers
  num_heads: 4               # Attention heads
  hidden_dim: 256            # Internal dimension
  use_attention_pooling: true

# Loss Function
loss:
  type: focal                # 'focal' or 'bce'
  focal_alpha: 0.5           # Class balance
  focal_gamma: 2.0           # Hard example focus

# Training
training:
  batch_size: 8
  epochs: 30
  learning_rate: 0.0001
  grad_clip: 1.0
  early_stop_patience: 7
  min_epochs: 10             # Minimum epochs before early stopping
  stratified_split: true     # Balanced train/val split

# Data
dataset:
  sequence_length: 64
  val_split: 0.2
```

### Override via Command Line

```bash
python train_temporal.py --epochs 50 --batch_size 16 --lr 0.0001
```

---

## 📁 Project Structure

```
project/
├── config.yaml                       # Configuration
├── requirements.txt                  # Dependencies
│
├── datasets/
│   └── video_sequence_dataset.py     # PyTorch dataset
│
├── models/
│   ├── efficientnet_b3_extractor.py  # Feature extraction
│   └── temporal_transformer.py       # Transformer + Focal Loss
│
├── utils/
│   ├── augmentation.py               # Temporal augmentations
│   └── metrics.py                    # Evaluation metrics
│
├── train_temporal.py                 # Training script
├── infer.py                          # Inference script
├── extract_features.py               # Feature extraction
│
├── report/
│   ├── Technical_Report.md           # Design decisions
│   ├── Model_Comparison.md           # Improvements
│   └── Metrics_Explanation.md        # Metrics guide
│
└── outputs/                          # Generated outputs
```

---

## 🏗️ Model Architecture

### Stage 1: Feature Extraction
- **Model:** EfficientNet-B3 (frozen, ImageNet weights)
- **Input:** 300×300 RGB frames
- **Output:** 1536-dim feature per frame

### Stage 2: Temporal Modeling
- **Architecture:** 4-layer Transformer Encoder
- **Positional Encoding:** Sinusoidal (fixed)
- **Pooling:** Attention-based
- **Loss:** Focal Loss (α=0.5, γ=2.0)

---

## 📊 Key Improvements

| Feature | Description | Impact |
|---------|-------------|--------|
| **4-Layer Transformer** | Hierarchical temporal learning | +5-10% |
| **Attention Pooling** | Learns which frames matter | +2-5% |
| **Focal Loss** | Hard example mining | +3-8% |
| **Pre-Norm** | Stable deep training | Better convergence |

---

## 📖 Documentation

- [Technical Report](report/Technical_Report.md) — Design decisions
- [Model Comparison](report/Model_Comparison.md) — Before vs after
- [Metrics Explanation](report/Metrics_Explanation.md) — Understanding metrics

---

## 💻 Example Commands

```bash
# Full pipeline
python extract_features.py --video_dir ./Dataset --output_dir ./Features
python train_temporal.py --epochs 30
python infer.py --input_dir ./Features

# Quick training with existing features
python train_temporal.py --feature_dir ./Features --epochs 30

# Custom configuration
python train_temporal.py --epochs 50 --batch_size 16 --lr 0.0001

# Inference on single video
python infer.py --input_path ./test.mp4 --model_path ./outputs/best_model.pth
```
