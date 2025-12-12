# Soccer Action Recognition - YOLO + Transformer

Real-time soccer action recognition combining YOLOv8 object detection with Transformer-based temporal modeling.

## Project Overview

This project implements a deep learning pipeline for temporal action recognition in soccer videos. The system classifies player actions into three categories: **PASS**, **DRIVE**, and **BACKGROUND**.

**Key Features:**
- Frozen YOLOv8n detector for spatial feature extraction
- ByteTrack algorithm for multi-object tracking
- Transformer-based temporal encoder (2 layers, 4 heads)
- Efficient YOLO feature caching and intelligent frame filtering
- Class balancing for improved minority class performance

**Pipeline Stages:**
1. **Stage 1**: YOLOv8n detects players/ball and extracts 512-dim spatial features
2. **Stage 2**: ByteTrack maintains temporal tracking across frames
3. **Stage 3**: Transformer encoder models temporal sequences for action classification

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/superposition-fc23/Soccer_action_spotting.git
cd Soccer_action_spotting
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n soccer_action python=3.9
conda activate soccer_action
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch 2.0+ (with MPS/CUDA/CPU support)
- Ultralytics (YOLOv8)
- OpenCV
- scikit-learn, pandas, numpy
- matplotlib, seaborn, tensorboard
- huggingface_hub (for dataset download)

See [requirements.txt](requirements.txt) for complete list.

### Step 4: Download Dataset (Optional - only if training)
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='SoccerNet/SN-BAS-2024',
    repo_type='dataset',
    revision='main',
    local_dir='SoccerNet/SN-BAS-2024'
)
"
```

## How to Run

### Running the Demo

The project includes a comprehensive demo script that showcases all functionality:

```bash
# Run complete demo (recommended for first-time users)
python demo.py --demo all

# Or run specific demos:
python demo.py --demo overview      # Pipeline overview
python demo.py --demo config        # Configuration details
python demo.py --demo model         # Model architecture
python demo.py --demo detection     # Detection & tracking demo
python demo.py --demo training      # Training setup
python demo.py --demo inference     # Inference instructions
python demo.py --demo evaluation    # Evaluation guide

# Get help
python demo.py --help
```

### Running Inference on Your Own Video

```bash
python inference_headless.py \
  --video path/to/your/video.mp4 \
  --model outputs/checkpoints/toy_best.pth \
  --output output_inference.mp4 \
  --window-size 32 \
  --stride 8
```

### Training from Scratch

```bash
python toy_train.py
```

### Evaluating the Model

```bash
python evaluate_model_fast.py \
  --model outputs/checkpoints/toy_best.pth \
  --split val
```

## Expected Output

### Demo Script Output
Running `python demo.py --demo all` will display:

1. **Pipeline Overview**: ASCII diagram of the 3-stage architecture
2. **Configuration**: All hyperparameters and settings
3. **Model Architecture**: Parameter counts (~500K parameters), component breakdown
4. **Detection Demo**: Processes 100 frames, shows detection/tracking statistics
5. **Training Setup**: Dataset info, optimizer/scheduler configuration
6. **Inference Guide**: Sample commands with actual file paths
7. **Evaluation Guide**: Metrics and checkpoint information

### Inference Output
Running inference produces:
- **Annotated video** with frame-by-frame action predictions
- **Action labels** overlaid on each frame
- **Running counts** for each action class (PASS, DRIVE, BACKGROUND)
- Console output showing progress and final statistics

Example console output:
```
[PROGRESS] 500/1000 frames (50.0%) - Current: PASS
[COMPLETE] Output saved to: output_inference.mp4
[RESULTS] Class counts:
  PASS: 523
  DRIVE: 341
  BACKGROUND: 136
```

### Evaluation Output
Running evaluation produces:
- **Accuracy, Precision, Recall, F1-Score** (per class and macro-averaged)
- **Confusion Matrix** saved to `outputs/results/`
- **mAP metrics** (mean Average Precision)
- Console output with detailed metrics table

## Architecture Highlights

- **Temporal Window**: 32 frames @ stride 2
- **Model**: Transformer (2 layers, 4 heads, hidden=128)
- **Optimization**: YOLO feature caching, class balancing (DRIVE 1.5x, BACKGROUND 3.0x)
- **Tracking**: ByteTrack for multi-object tracking

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for complete technical details.

## Project Structure

```
├── models/           # Model architectures (detector, tracker, classifier)
├── utils/            # Dataset, metrics, filtering utilities
├── toy_train.py      # Training script
├── evaluate_model_fast.py    # Evaluation with metrics/plots
├── inference_headless.py     # Video inference
└── toy_config.py     # Central configuration
```

## Pre-trained Model

**Pre-trained models are included in this repository** at `outputs/checkpoints/`:

- **Best Model**: `outputs/checkpoints/toy_best.pth` (~11 MB)
  - Best validation loss checkpoint
  - Recommended for inference and evaluation

- **Latest Model**: `outputs/checkpoints/toy_latest.pth` (~11 MB)
  - Most recent training checkpoint
  - Useful for resuming training

**Model Details:**
- Architecture: Transformer-based (2 layers, 4 heads, hidden dim=128)
- Parameters: ~500K trainable parameters
- Training: 10 epochs on 4 training videos, 2 validation videos
- Performance: See metrics in `outputs/metrics/`

**To use the pre-trained model:**
```python
import torch
from models.toy_action_classifier import ToyActionClassifier

# Load model
model = ToyActionClassifier()
checkpoint = torch.load('outputs/checkpoints/toy_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Results

**Training results available in `outputs/metrics/`:**
- Training accuracy and loss graphs
- Validation accuracy and loss graphs
- mAP (mean Average Precision) and per-class precision metrics
- Confusion matrix visualization

**Key Performance Metrics:**
- See checkpoint file for validation accuracy and loss
- Per-class F1 scores available in evaluation output
- Detailed metrics can be regenerated with `evaluate_model_fast.py`

## Technical Stack

- **PyTorch** 2.0+ (MPS/CUDA/CPU support)
- **YOLOv8n** (Ultralytics) - Nano model for efficiency
- **ByteTrack** (Multi-object tracking)
- **Transformer** (Custom temporal encoder)

## Dataset

**SoccerNet Ball Action Spotting 2024**

Download from HuggingFace (optional, only needed for training):
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="SoccerNet/SN-BAS-2024",
    repo_type="dataset",
    revision="main",
    local_dir="SoccerNet/SN-BAS-2024"
)
```

**Dataset Structure:**
- 13 original action classes → mapped to 3 classes
- Classes: PASS (includes HIGH_PASS, HEADER, CROSS), DRIVE, BACKGROUND
- Training: 4 videos, Validation: 2 videos
- Video format: 224p, 25 FPS

## Reproducibility

### Hyperparameters

All hyperparameters are defined in [toy_config.py](toy_config.py):

**Model Architecture:**
- Feature dimension: 512 (from YOLOv8)
- Hidden dimension: 128
- Transformer layers: 2
- Attention heads: 4
- Dropout: 0.2

**Training Configuration:**
- Batch size: 4
- Learning rate: 5e-6
- Weight decay: 1e-5
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Epochs: 10
- Early stopping patience: 7

**Temporal Settings:**
- Window size: 32 frames
- Stride: 2 frames
- Context frames: 8

**Class Balancing:**
- PASS: 1.0x (baseline)
- DRIVE: 1.5x (oversampling)
- BACKGROUND: 3.0x (oversampling)

### Training Setup

To reproduce the results:
1. Follow setup instructions above
2. Ensure dataset is downloaded to `SoccerNet/SN-BAS-2024/`
3. Run `python toy_train.py`
4. Model checkpoints saved to `outputs/checkpoints/`
5. TensorBoard logs saved to `outputs/logs/`

**Training Environment:**
- Device: MPS (Apple Silicon) / CUDA / CPU auto-detected
- YOLOv8 features cached to disk for faster training
- Intelligent frame filtering reduces computation by ~40%

## Acknowledgments

This project builds upon several excellent open-source projects and datasets:

**Dataset:**
- [SoccerNet Ball Action Spotting 2024](https://huggingface.co/datasets/SoccerNet/SN-BAS-2024) - Soccer action dataset
- SoccerNet Challenge organizers for the comprehensive annotation and benchmark

**Models and Algorithms:**
- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics - Object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) by Zhang et al. - Multi-object tracking
- PyTorch team for the deep learning framework

**Research References:**
- [T-DEED](https://github.com/arturxe2/T-DEED) - 2024 SoccerNet winning model ([Paper](https://arxiv.org/pdf/2404.05392))
- [Ball Action Spotting](https://github.com/lRomul/ball-action-spotting) - 2023 SoccerNet winning approach

**Tools and Libraries:**
- PyTorch, torchvision
- OpenCV for video processing
- scikit-learn for metrics
- matplotlib, seaborn for visualization
- TensorBoard for training monitoring

## License

This project is for educational purposes as part of the Computer Vision course (Fall 2025).

---

**Author**: Mithun M
**Course**: Computer Vision and Temporal Architectures, Fall 2025
**Institution**: UCLA
