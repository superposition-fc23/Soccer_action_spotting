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

## Demo

Input: Broadcast soccer match footage (224p, 25fps)
Output: Frame-by-frame action predictions with confidence scores

See `presentation_results/` for evaluation plots and `presentation_inference_output.mp4` for sample output.

## Citation

Dataset: [SoccerNet Ball Action Spotting 2024](https://www.soccer-net.org/)

---

**Author**: Mithun M | Computer Vision and Temporal architectures, Fall 2025
