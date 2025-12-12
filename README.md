# Soccer Action Recognition - YOLO + Transformer

Real-time soccer action recognition combining YOLOv8 object detection with Transformer-based temporal modeling.

## Overview

A two-stage deep learning pipeline that classifies soccer player actions (PASS, DRIVE, BACKGROUND) from broadcast video:
- **Stage 1**: Frozen YOLOv8x extracts spatial features and detects players/ball
- **Stage 2**: Tracking algorithm using Bytetrack
- **Stage 3**: Trainable Transformer encoder models temporal sequences


## Quick Start

```bash
# Install dependencies
pip install torch torchvision ultralytics opencv-python scikit-learn

# Train model
python toy_train.py

# Evaluate
python evaluate_model_fast.py --model outputs/toy_experiment/models/toy_best.pth

# Run inference on video
python inference_headless.py --video input.mp4 --model toy_best.pth --output result.mp4
```

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

## Technical Stack

- **PyTorch** 2.0+ (MPS/CUDA/CPU support)
- **YOLOv8x** (Ultralytics)
- **ByteTrack** (Multi-object tracking)
- **Transformer** (Custom temporal encoder)

## Dataset

Dataset: 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="SoccerNet/SN-BAS-2024",
                  repo_type="dataset", revision="main",
                  local_dir="SoccerNet/SN-BAS-2024")

## Checkpoints

Trained weights available at : 
For latest weights -  /outputs/checkpoints/toy_latest.pth
For best weights - /outputs/checkpoints/toy_best.pth

## Demo

Input: Broadcast soccer match footage (224p, 25fps)
Output: Frame-by-frame action predictions with confidence scores
Available through the demo.py script

## Results

- Graphs for loss & accuracy as well as mAP scores available at : outputs/metrics/
- F1 score and confusion matrix also part of the code in outputs/metrics.py for future runs

## Citation

Soccernet Action Dataset 2024 : Use command from https://huggingface.co/datasets/SoccerNet/SN-BAS-2025 and change to "2024"

2024 Soccernet Winning Model (T-DEED) & Paper: https://github.com/arturxe2/T-DEED/tree/main
https://arxiv.org/pdf/2404.05392

2023 Soccernet Winning Model: https://github.com/lRomul/ball-action-spotting/tree/master

YOLOv8 & Usage: https://yolov8.com/

ByteTrack : https://github.com/FoundationVision/ByteTrack

---

**Author**: Mithun M | Computer Vision and Temporal architectures, Fall 2025
