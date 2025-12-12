"""
Toy Configuration for Quick Experimentation
Optimized for fast training on M1 Mac with intelligent frame filtering
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT / "SoccerNet" / "SN-BAS-2024"

# Data paths
TRAIN_DIR = DATASET_ROOT / "train_england_efl"
TEST_DIR = DATASET_ROOT / "test_england_efl"
VALID_DIR = DATASET_ROOT / "valid_england_efl 2"
CHALLENGE_DIR = DATASET_ROOT / "challenge_england_efl"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "toy_experiment"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Performance Optimizations (Phase 2a) - defined early for directory creation
USE_YOLO_CACHE = True
YOLO_CACHE_DIR = OUTPUT_DIR / "yolo_cache"
USE_BATCHED_YOLO = True

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR, VISUALIZATION_DIR, YOLO_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# TOY EXPERIMENT SETTINGS
# ============================================

# Limit dataset size for quick experimentation
MAX_TRAIN_VIDEOS = 4  # Use all 4 videos for training
MAX_VAL_VIDEOS = 2    # Use only 2 videos for validation
MAX_TEST_VIDEOS = 1   # Use only 1 video for testing

# Video processing
VIDEO_RESOLUTION = "224p"  # Lower resolution for speed
FPS = 25
FRAME_SAMPLE_RATE = 1  # Process all frames (filtering handles reduction)

# ============================================
# INTELLIGENT FRAME FILTERING
# ============================================
USE_INTELLIGENT_FILTERING = True  # Enable intelligent filtering to speed up training
BALL_PLAYER_DISTANCE_THRESHOLD = 0.1  # Normalized coords [0, 1]
MIN_FRAMES_PER_WINDOW = 20  # Minimum frames to keep even after filtering
FILTER_STRATEGY = "ball_player_proximity"  # Options: "ball_player_proximity", "none"

# YOLO Configuration
YOLO_VERSION = "yolov8"
YOLO_MODEL_SIZE = "n"  # Nano model for speed
YOLO_PRETRAINED = True
YOLO_CONFIDENCE_THRESHOLD = 0.15
YOLO_IOU_THRESHOLD = 0.3

# Detection classes
DETECT_CLASSES = {
    0: "person",
    1: "ball"
}

# Tracking Configuration
TRACKER_TYPE = "bytetrack"
TRACKER_MAX_AGE = 30
TRACKER_MIN_HITS = 3
TRACKER_IOU_THRESHOLD = 0.8


# Action Classification - REDUCED TO 3 CLASSES
# Original 5 classes: PASS, DRIVE, HIGH_PASS, HEADER, BACKGROUND
# New 3 classes: PASS (includes HIGH_PASS, HEADER), DRIVE, BACKGROUND
ACTION_CLASSES = {
    0: "PASS",       # Includes PASS, HIGH_PASS, HEADER
    1: "DRIVE",      # Unchanged
    2: "BACKGROUND"  # Unchanged
}

# Label mapping from original 5 classes to new 3 classes
LABEL_MAPPING_5_TO_3 = {
    0: 0,  # PASS → PASS
    1: 1,  # DRIVE → DRIVE
    2: 0,  # HIGH_PASS → PASS
    3: 0,  # HEADER → PASS
    4: 2,   # BACKGROUND → BACKGROUND
    5: 2,  # OUT → BACKGROUND
    6: 2,  # THROW IN → BACKGROUND
    7: 0,  # CROSS → PASS
    8: 2,  # BALL PLAYER BLOCK → BACKGROUND
    9: 2,  # SHOT → BACKGROUND
    10: 2, # PLAYER SUCCESSFUL TACKLE → BACKGROUND
    11: 2, # FREE KICK → BACKGROUND
    12: 2  # GOAL → BACKGROUND
}

NUM_CLASSES = len(ACTION_CLASSES)  # Now 3 instead of 5

# ============================================
# REDUCED TEMPORAL WINDOW
# ============================================
TEMPORAL_WINDOW_SIZE = 32  # Reduced from 64 to 32
TEMPORAL_STRIDE = 2 # Changed from 1 to 8 to 2 iteratively (post architecture improvements) for better performance
ACTION_CONTEXT_FRAMES = 8  # Reduced from 16

# ============================================
# DENSE TEMPORAL SAMPLING (NEW)
# ============================================
# Toggle between action-centric sampling (current) and dense sampling (new)
USE_DENSE_SAMPLING = False  # OPTIMIZED - now uses fast mode without opening videos during init
DENSE_SAMPLING_STRIDE = 32  # Frames between window starts (32 frame window every 32 frames = no overlap)
BACKGROUND_TO_ACTION_RATIO = 1.0  # How many background samples per action sample (1.0 = 1:1 ratio, balanced)

# ============================================
# CLASS BALANCING (NEW)
# ============================================
# Enable class balancing to address class imbalance
# Current distribution: PASS (52%), DRIVE (34%), BACKGROUND (14%)
# Target: More balanced distribution through oversampling minority classes
USE_CLASS_BALANCING = True

# Class oversampling ratios (multiplier for each class)
# 1.0 = keep original count, 1.5 = add 50% more samples, 3.0 = triple samples
CLASS_BALANCE_RATIOS = {
    0: 1.0,   # PASS - keep as is (already majority)
    1: 1.5,   # DRIVE - increase by 50%
    2: 3.0    # BACKGROUND - triple (200% increase)
}

# ============================================
# REDUCED MODEL ARCHITECTURE
# ============================================
FEATURE_DIM = 512  # Keep YOLO feature dim same
HIDDEN_DIM = 128   # Reduced from 256 to 128
NUM_LAYERS = 2     # Reduced from 4 to 2 (but continuing with transformer)
NUM_HEADS = 4      # Reduced from 8 to 4
DROPOUT = 0.2

# Training Configuration
BATCH_SIZE = 4     # Increased batch size due to smaller model
NUM_EPOCHS = 10    # Much fewer epochs for quick validation
LEARNING_RATE = 5e-6    # Halved from previous runs for stability and gradual learning
WEIGHT_DECAY = 1e-5

#LR scheduler parameters
EARLY_STOPPING_PATIENCE = 7  # Reduced patience as learning is faster on small dataset & LR is reduced
LR_FACTOR = 0.5  # Reduce LR by half when plateaued
MIN_LR = 1e-7    # Don't go below this

# Warmup to ease into training
USE_WARMUP = True
WARMUP_EPOCHS = 2
WARMUP_START_LR = 1e-7


# Data Augmentation - Keep it simple for toy experiment
USE_AUGMENTATION = False  # Disable for faster training

# Loss weights
LOSS_WEIGHTS = {
    "detection": 1.0,
    "classification": 2.0,
    "temporal_consistency": 0.5
}

# Evaluation
EVAL_METRICS = ["accuracy", "precision", "recall", "f1"]
CONFUSION_MATRIX = True

# Logging
LOG_INTERVAL = 1   # Log more frequently
SAVE_INTERVAL = 1  # Save model every epoch
TENSORBOARD = True

# Hardware - Auto-detect MPS for M1 Mac
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# NUM_WORKERS: Force to 0 to avoid deadlock with network-mounted storage
# Multi-process data loading hangs indefinitely with slow network I/O
NUM_WORKERS = 0

# PIN_MEMORY: Auto-adjust based on device
# - False for MPS (not supported)
# - True for CUDA (faster host-to-device transfer)
PIN_MEMORY = False if DEVICE == "mps" else True

# Inference
INFERENCE_BATCH_SIZE = 1
SAVE_VISUALIZATIONS = True
VISUALIZATION_FPS = 10

# Class imbalance handling
USE_CLASS_WEIGHTS = True
FOCAL_LOSS = False  # Disable - causes NaN due to numerical instability
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Team detection
USE_TEAM_DETECTION = False  # Disable for simplicity

# Debug mode
DEBUG = True  # Enable debug mode for toy experiment
VISUALIZE_DETECTIONS = True
SAVE_DEBUG_FRAMES = True

print(f"[TOY CONFIG] Device: {DEVICE}")
print(f"[TOY CONFIG] Temporal window: {TEMPORAL_WINDOW_SIZE}")
print(f"[TOY CONFIG] Hidden dim: {HIDDEN_DIM}")
print(f"[TOY CONFIG] Intelligent filtering: {USE_INTELLIGENT_FILTERING}")
print(f"[TOY CONFIG] Ball-player distance threshold: {BALL_PLAYER_DISTANCE_THRESHOLD}px")
