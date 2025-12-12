# Soccer Action Recognition - Project Architecture

---

## 🎯 System Overview

**Goal**: Temporal action recognition in soccer videos (PASS, DRIVE, BACKGROUND)

**Approach**: Two-stage pipeline with frozen YOLO detector + trainable Transformer classifier

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Soccer Video                          │
│                    (25 FPS, 224p resolution)                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │         STAGE 1: SPATIAL DETECTION             │
        │    (Frozen Pre-trained YOLOv8x/YOLOv11x)      │
        │                                                │
        │  - Detect players (class 0) & ball (class 32) │
        │  - Extract bounding boxes & confidence scores │
        │  - Extract spatial features (512-dim)         │
        │  - Cache features to disk for efficiency      │
        └───────────────┬────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────┐
        │         STAGE 2: TEMPORAL TRACKING              │
        │            (ByteTrack Algorithm)                │
        │                                                │
        │  - Track objects across frames (Kalman filter) │
        │  - Maintain track IDs, trajectories, history   │
        │  - Extract tracking features (128-dim)         │
        └───────────────┬────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────┐
        │      STAGE 3: TEMPORAL CLASSIFICATION           │
        │      (Trainable Transformer Encoder-Decoder)    │
        │                                                │
        │  Input: 32-frame temporal window               │
        │  - Combine spatial (512) + tracking (128) = 640│
        │  - Transformer Encoder (2 layers, 4 heads)     │
        │  - Hidden dim: 128                             │
        │  - Decoder: Temporal aggregation + classifier  │
        │  Output: 3 classes (PASS, DRIVE, BACKGROUND)   │
        └───────────────┬────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────┐
        │              OUTPUT: Action Label               │
        │         Softmax probabilities over 3 classes    │
        └────────────────────────────────────────────────┘
```

---

## 🏗️ Detailed Component Architecture

### 1. Configuration Layer

**File**: [toy_config.py](toy_config.py)

**Purpose**: Centralized configuration for all hyperparameters

```python
Key Parameters:
├── Dataset
│   ├── MAX_TRAIN_VIDEOS = 4
│   ├── MAX_VAL_VIDEOS = 2
│   ├── VIDEO_RESOLUTION = "224p"
│   └── FPS = 25
├── Temporal Window
│   ├── TEMPORAL_WINDOW_SIZE = 32
│   ├── TEMPORAL_STRIDE = 2
│   └── ACTION_CONTEXT_FRAMES = 8
├── Model Architecture
│   ├── FEATURE_DIM = 512 (YOLO spatial)
│   ├── HIDDEN_DIM = 128 (Transformer)
│   ├── NUM_LAYERS = 2 (Transformer)
│   ├── NUM_HEADS = 4 (Attention)
│   └── DROPOUT = 0.2
├── Training
│   ├── BATCH_SIZE = 4
│   ├── NUM_EPOCHS = 10
│   ├── LEARNING_RATE = 5e-6
│   └── WEIGHT_DECAY = 1e-5
├── Class Balancing (NEW)
│   ├── USE_CLASS_BALANCING = True
│   └── CLASS_BALANCE_RATIOS = {0: 1.0, 1: 1.5, 2: 3.0}
└── Label Mapping (13 → 3 classes)
    ├── 0: PASS (includes HIGH_PASS, HEADER, CROSS)
    ├── 1: DRIVE
    └── 2: BACKGROUND (includes OUT, THROW IN, SHOT, etc.)
```

---

### 2. Data Pipeline

#### 2.1 Dataset Loader

**File**: [utils/toy_dataset.py](utils/toy_dataset.py)

**Class**: `ToyActionDataset`

**Key Features**:
- Action-centric sampling around labeled actions (±8 frames context)
- Optional dense temporal sampling (stride=32)
- **Class balancing** via oversampling (DRIVE +50%, BACKGROUND +200%)
- Label mapping from 13 → 3 classes
- Metadata tracking (gameTime, video_id, action_id)

**Data Flow**:
```
Raw Video Dataset (SoccerNet BAS-2024)
         ↓
Parse JSON annotations (labels-ball.json)
         ↓
Map 13 labels → 3 classes (PASS/DRIVE/BACKGROUND)
         ↓
Create temporal windows (32 frames, stride 2)
         ↓
Apply class balancing (training only)
         ↓
Return batch: {video, label, metadata}
```

#### 2.2 Class Balancing

**Implementation**: `_apply_class_balancing()` method

**Before**:
- PASS: 52% (4,497 samples)
- DRIVE: 34% (2,942 samples)
- BACKGROUND: 14% (1,254 samples)

**After** (with ratios 1.0, 1.5, 3.0):
- PASS: ~40% (4,497 samples)
- DRIVE: ~35% (4,413 samples, +1,471)
- BACKGROUND: ~25% (3,762 samples, +2,508)

**Total**: 8,693 → 12,672 samples (+3,979 duplicates)

---

### 3. Detection & Tracking Pipeline

#### 3.1 Player & Ball Detector

**File**: [models/detector.py](models/detector.py)

**Class**: `PlayerBallDetector`

**Model**: YOLOv8x or YOLOv11x (frozen, pre-trained on COCO)

**Key Operations**:
```python
detect_frame(frame, return_features=True):
    Input: RGB frame (H, W, 3)
    ↓
    Run YOLO inference
    ↓
    Filter: classes [0=person, 32=ball]
    ↓
    Apply NMS (conf=0.15, iou=0.3)
    ↓
    Extract spatial features (512-dim from backbone)
    ↓
    Output: {
        'boxes': Nx4 (x1, y1, x2, y2),
        'classes': N,
        'scores': N,
        'features': 512-dim tensor
    }
```

**Feature Caching**:
- YOLO features cached to disk: `outputs/toy_experiment/yolo_cache/`
- Cache key: `{video_id}_{frame_idx}.pt`
- Speeds up training by ~3-5x

#### 3.2 Multi-Object Tracker

**File**: [models/tracker.py](models/tracker.py)

**Class**: `ByteTracker`

**Algorithm**: ByteTrack with Kalman filtering

**Key Features**:
- Track players & ball across frames
- Assign unique track IDs
- Maintain trajectory history
- Extract tracking features:
  - Position (cx, cy)
  - Size (w, h)
  - Velocity (vx, vy)
  - Track age, hits count

**Tracking Features (128-dim)**:
```python
TrackFeatureExtractor:
    Input: List of tracks with bboxes, velocities
    ↓
    Compute statistics:
        - Ball-player distances
        - Player positions (normalized)
        - Velocities
        - Track counts
    ↓
    MLP embedding: [raw_features] → 128-dim
    ↓
    Output: 128-dim track embedding per frame
```

---

### 4. Model Architecture

#### 4.1 Overall Model

**File**: [models/toy_action_classifier.py](models/toy_action_classifier.py)

**Class**: `ToyActionClassifier`

```
Input: Temporal window (32 frames)
    ├── Spatial features: (32, 512) from YOLO
    └── Detections & tracks: List[Dict] per frame

        ↓

Feature Fusion:
    ├── Spatial features: (32, 512)
    └── Track features: (32, 128) from TrackFeatureExtractor
        ↓
    Concatenate: (32, 640)

        ↓

Temporal Encoder (Transformer):
    ├── Input: (Batch, 32, 640)
    ├── 2 Transformer Encoder Layers
    ├── 4 Attention Heads
    ├── Hidden dim: 128
    ├── Dropout: 0.2
    └── Output: (Batch, 32, 128)

        ↓

Temporal Decoder:
    ├── GRU layer (1 layer, hidden=128)
    ├── Take last hidden state: (Batch, 128)
    └── Linear classifier: 128 → 3 classes

        ↓

Output: (Batch, 3) logits → Softmax → Probabilities
```

#### 4.2 Temporal Encoder

**File**: [models/action_classifier.py](models/action_classifier.py) (imported)

**Class**: `TemporalEncoder`

**Type**: Transformer (2 layers, 4 heads)

**Architecture**:
```python
TransformerEncoder:
    ├── Positional Encoding (learned)
    ├── Layer 1:
    │   ├── Multi-Head Self-Attention (4 heads)
    │   ├── LayerNorm
    │   ├── Feedforward (128 → 512 → 128)
    │   └── Residual connection
    ├── Layer 2: (same structure)
    └── Output: Contextualized sequence (32, 128)
```

#### 4.3 Temporal Decoder (simplified to a linear layer)

**Class**: `TemporalDecoder`

**Architecture**:
```python
GRU-based decoder:
    ├── GRU: (input=128, hidden=128, 1 layer)
    ├── Take final hidden state: h_T
    ├── Dropout (0.2)
    └── Linear: 128 → 3 classes
```

---

### 5. Training Pipeline

#### 5.1 Training Script

**File**: [toy_train.py](toy_train.py)

**Class**: `ToyTrainer`

**Loss Function**:
- Primary: CrossEntropyLoss (weighted by class frequency)
- Optional: Focal Loss (disabled - causes NaN)

**Optimizer**:
- Adam: lr=5e-6, weight_decay=1e-5
- Warmup: 2 epochs (1e-7 → 5e-6)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=7)

**Early Stopping**:
- Patience: 7 epochs
- Monitor: Validation loss

#### 5.2 Feature Extraction in Training

**Critical Dimension Handling** (Fixed):

**Location 1 - Cached Features**:
```python
# Load cached YOLO features
raw_f = torch.load(cache_path)

# Fix: Ensure at least 1D (prevent 0-dim tensors)
if not hasattr(raw_f, 'shape') or raw_f.dim() == 0:
    raw_f = raw_f.view(1) if hasattr(raw_f, 'view') else torch.tensor([raw_f])

# Project if needed
if raw_dim != FEATURE_DIM:
    f_proj = feature_proj(raw_f)
    f_proj = feature_norm(f_proj)
    frame_features.append(f_proj)
```

**Location 2 - Fresh YOLO Features**:
```python
# Extract features from YOLO
raw_features = detection_result['features']  # (C, H, W)

# Pool to (1, 1) spatial dims
feat = F.adaptive_avg_pool2d(raw_features.unsqueeze(0), (1, 1))

# Fix: Carefully remove only spatial and batch dims
feat = feat.squeeze(-1).squeeze(-1).squeeze(0)  # → (C,)

# Safety check for 0-dim tensors
if feat.dim() == 0:
    feat = feat.unsqueeze(0)
```

---

### 6. Evaluation & Metrics

#### 6.1 Unified Metrics Module

**File**: [utils/metrics.py](utils/metrics.py)

**Class**: `MetricsTracker`

**Computed Metrics**:
```python
MetricsTracker.compute() returns:
    ├── Basic
    │   ├── loss: Average loss
    │   └── accuracy: Top-1 accuracy (%)
    ├── F1 Scores
    │   ├── per_class_f1: [F1_PASS, F1_DRIVE, F1_BACKGROUND]
    │   └── macro_f1: Mean of per-class F1
    ├── Confusion Matrix
    │   └── confusion_matrix: 3x3 matrix
    ├── Average Precision
    │   ├── map_at_1: Mean AP at rank 1 (≈ accuracy)
    │   ├── tight_avg_map: Macro-averaged AP
    │   └── per_class_ap: [AP_PASS, AP_DRIVE, AP_BACKGROUND]
    ├── Classification Report
    │   └── Precision, Recall, F1 per class (sklearn format)
    └── num_samples: Total samples evaluated
```

**Visualization Methods**:
- `plot_confusion_matrix()` → RGB array (for TensorBoard)
- `plot_per_class_f1()` → Bar chart
- `plot_per_class_ap()` → Bar chart

**Usage Pattern**:
```python
# Initialize
tracker = MetricsTracker(class_names={0: "PASS", 1: "DRIVE", 2: "BACKGROUND"})

# Accumulate during epoch
for batch in dataloader:
    outputs, labels, loss = forward_pass(batch)
    tracker.update(outputs, labels, loss)

# Compute at end
metrics = tracker.compute()
tracker.print_summary("validation")

# Log to TensorBoard
cm_img = tracker.plot_confusion_matrix()
writer.add_image('ConfusionMatrix/val', cm_img, epoch, dataformats='HWC')

# Reset for next epoch
tracker.reset()
```

#### 6.2 Fast Evaluation Script (Optional - For quick sampling)

**File**: [evaluate_model_fast.py](evaluate_model_fast.py)

**Class**: `FastModelEvaluator`

**Purpose**: Quick evaluation on subset of data

**Features**:
- Evaluate N batches only (fast mode)
- Optional time filtering (e.g., minutes 10-13 of video)
- Generates all plots (confusion matrix, F1, AP, summary)
- Saves results to JSON

**Usage**:
```bash
python evaluate_model_fast.py \
    --model outputs/toy_experiment/models/toy_best.pth \
    --output evaluation_results_fast \
    --train-batches 50 \
    --val-batches 25 \
    --time-start 10 \
    --time-end 13
```

---

### 7. Inference Pipeline

#### 7.1 Headless Inference (Video Output)

**File**: [inference_headless.py](inference_headless.py)

**Class**: `HeadlessInference`

**Purpose**: Save annotated video with predictions (no live display)

**Key Features**:
- ✅ Confidence threshold parameter (default: 0.5)
- ✅ Always show all class counts (even zeros)
- ✅ Black text for all annotations
- Sliding window classification (32 frames, stride 8)
- Frame-by-frame annotation & writing

**Usage**:
```bash
python inference_headless.py \
    --video "Toy challenge.mp4" \
    --model outputs/toy_experiment/models/toy_best.pth \
    --output output_inference.mp4 \
    --window-size 32 \
    --stride 8 \
    --confidence-threshold 0.7
```

**Annotation Format**:
```
┌─────────────────────────────────┐
│ Action: DRIVE                   │ (Black text)
│                                 │
│ PASS: 142                       │ (Black text)
│ DRIVE: 87                       │ (Black text)
│ BACKGROUND: 23                  │ (Black text)
└─────────────────────────────────┘
```

#### 7.2 Dual-View Inference (Pending)

**Status**: NOT YET IMPLEMENTED

**Purpose**: Side-by-side visualization
- Left: Real-time video with predictions
- Right: 32-frame temporal window (what model sees)

---

### 8. Output & Results Structure

```
outputs/toy_experiment/
├── models/
│   ├── toy_best.pth       (Best model by val loss)
│   └── toy_latest.pth     (Latest epoch checkpoint)
├── results/
│   └── Toy_run_6_inference.mp4  (Inference video)
├── logs/
│   └── toy_run_20251210_131122/
│       └── events.out.tfevents.*  (TensorBoard logs)
├── visualizations/
│   └── (Empty - metrics not yet integrated)
└── yolo_cache/
    └── {video_id}_{frame_idx}.pt  (Cached YOLO features)
```

**Checkpoint Format**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'history': {
        'train_loss': [...],
        'val_loss': [...],
        'train_acc': [...],
        'val_acc': [...]
    }
}
```

---

## 🔄 Data Flow: Training

```
1. Dataset Loading
   ├── Parse JSON annotations
   ├── Map 13 labels → 3 classes
   ├── Create temporal windows (32 frames)
   ├── Apply class balancing (train only)
   └── Batch: {video, label, metadata}

2. Feature Extraction (per batch)
   ├── For each video in batch (shape: B, T, C, H, W)
   │   ├── For each frame in video (T=32)
   │   │   ├── Check cache: yolo_cache/{video_id}_{frame_idx}.pt
   │   │   ├── If cached:
   │   │   │   └── Load cached features (512-dim)
   │   │   └── Else:
   │   │       ├── Run YOLO detection
   │   │       ├── Extract spatial features (512-dim)
   │   │       ├── Run ByteTrack tracking
   │   │       └── Cache features to disk
   │   └── Extract tracking features (128-dim)
   └── Combine: spatial (512) + tracking (128) = 640-dim

3. Model Forward Pass
   ├── Input: (B, 32, 640)
   ├── Transformer Encoder: (B, 32, 640) → (B, 32, 128)
   ├── Linear layer: (B, 32, 128) → (B, 128)
   └── Classifier: (B, 128) → (B, 3)

4. Loss & Optimization
   ├── CrossEntropyLoss(outputs, labels)
   ├── loss.backward()
   └── optimizer.step()

5. Metrics Tracking (NOT YET INTEGRATED)
   └── Will use MetricsTracker for comprehensive evaluation
```

---

## 🔄 Data Flow: Inference

```
1. Load Model
   └── ToyActionClassifier.load_state_dict(checkpoint)

2. Initialize Components
   ├── PlayerBallDetector (YOLOv8x)
   └── ByteTracker

3. Process Video
   ├── Open video file
   └── Create video writer

4. Sliding Window Classification
   ├── Read frame
   ├── Add to buffer
   ├── If buffer >= 32 frames:
   │   ├── Extract features (last 32 frames)
   │   │   ├── YOLO detection
   │   │   ├── ByteTrack tracking
   │   │   └── Combine spatial + tracking features
   │   ├── Forward pass: features → logits
   │   ├── Softmax: logits → probabilities
   │   ├── Check confidence threshold
   │   ├── Get prediction: argmax(probabilities)
   │   └── Update class counts
   ├── Annotate frame with prediction + counts
   ├── Write frame to output video
   └── Slide buffer by stride (8 frames)

5. Save Output
   └── Close video writer
```

---

## 🎨 Label Mapping

**From 13 Original Labels → 3 Final Classes**:

```python
LABEL_MAPPING_5_TO_3 = {
    # PASS class (0)
    0: 0,   # PASS → PASS
    2: 0,   # HIGH_PASS → PASS
    3: 0,   # HEADER → PASS
    7: 0,   # CROSS → PASS

    # DRIVE class (1)
    1: 1,   # DRIVE → DRIVE

    # BACKGROUND class (2)
    4: 2,   # BACKGROUND → BACKGROUND
    5: 2,   # OUT → BACKGROUND
    6: 2,   # THROW IN → BACKGROUND
    8: 2,   # BALL PLAYER BLOCK → BACKGROUND
    9: 2,   # SHOT → BACKGROUND
    10: 2,  # PLAYER SUCCESSFUL TACKLE → BACKGROUND
    11: 2,  # FREE KICK → BACKGROUND
    12: 2   # GOAL → BACKGROUND
}
```

**Class Distribution** (after balancing):
- PASS: ~40% (includes tactical passes, crosses, headers)
- DRIVE: ~35% (dribbling with ball)
- BACKGROUND: ~25% (all other actions, game stoppages)

---

## 🔧 Performance Optimizations

### 1. YOLO Feature Caching
- **Location**: `outputs/toy_experiment/yolo_cache/`
- **Format**: `{video_id}_{frame_idx}.pt`
- **Impact**: ~3-5x training speedup

### 2. Reduced Model Size
- Hidden dim: 256 → 128
- Transformer layers: 4 → 2
- Attention heads: 8 → 4
- **Impact**: Faster training, less GPU memory

### 3. Class Balancing
- Oversample minority classes (DRIVE +50%, BACKGROUND +200%)
- **Impact**: Better minority class performance

### 4. Intelligent Frame Filtering (Optional)
- Filter frames by ball-player distance
- `USE_INTELLIGENT_FILTERING = True`

---

## (Toy Run 7) Training Setup**:
- Videos: 4 train, 2 val
- Epochs: 10
- Batch size: 4
- LR: 5e-6
- Class balancing: Enabled

**Note**: Specific metrics available in TensorBoard at:
```bash
tensorboard --logdir=outputs/toy_experiment/logs
```

---

## 🚀 Pending Improvements

### High Priority
1. ✅ Dataset class balancing - COMPLETED
2. ✅ Unified metrics module - COMPLETED
3. ⏳ Integrate metrics into training loop
4. ✅ Add confidence threshold to inference - COMPLETED
5. ✅ Update inference visualization (black text, all counts) - COMPLETED

### Medium Priority
6. ⏳ Create dual-view inference script
7. ⏳ Implement mAP computation in training

### Optional
- Vision Transformer exploration (Task 4 - deferred)
- Hyperparameter tuning
- Multi-GPU training support

---

## 🛠️ Key Files Reference

### Core Models
- [models/toy_action_classifier.py](models/toy_action_classifier.py) - Main classifier
- [models/detector.py](models/detector.py) - YOLO wrapper
- [models/tracker.py](models/tracker.py) - ByteTrack implementation
- [models/action_classifier.py](models/action_classifier.py) - Encoder/Decoder components

### Data & Utils
- [toy_config.py](toy_config.py) - Configuration
- [utils/toy_dataset.py](utils/toy_dataset.py) - Dataset loader with balancing
- [utils/metrics.py](utils/metrics.py) - Unified metrics tracking
- [utils/frame_filter.py](utils/frame_filter.py) - Intelligent filtering (optional)

### Training & Evaluation
- [toy_train.py](toy_train.py) - Main training script
- [toy_train_resume.py](toy_train_resume.py) - Resume training from checkpoint
- [evaluate_model_fast.py](evaluate_model_fast.py) - Fast evaluation

### Inference
- [inference_headless.py](inference_headless.py) - Video output inference

---

**End of Architecture Document**
