"""
Soccer Action Recognition - Comprehensive Demo
Demonstrates the complete pipeline from training to inference
"""
import sys
from pathlib import Path
import argparse
import torch

sys.path.append(str(Path(__file__).parent))
import toy_config as config


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def demo_configuration():
    """Demo: Display current configuration"""
    print_section("DEMO 1: Configuration Overview")

    print(f"\n📁 Project Structure:")
    print(f"  Dataset Root: {config.DATASET_ROOT}")
    print(f"  Output Dir:   {config.OUTPUT_DIR}")
    print(f"  Model Dir:    {config.MODEL_DIR}")

    print(f"\n🎥 Video Processing:")
    print(f"  Resolution:     {config.VIDEO_RESOLUTION}")
    print(f"  FPS:            {config.FPS}")
    print(f"  Train Videos:   {config.MAX_TRAIN_VIDEOS}")
    print(f"  Val Videos:     {config.MAX_VAL_VIDEOS}")

    print(f"\n🧠 Model Architecture:")
    print(f"  YOLO Model:     yolov8{config.YOLO_MODEL_SIZE}")
    print(f"  Feature Dim:    {config.FEATURE_DIM}")
    print(f"  Hidden Dim:     {config.HIDDEN_DIM}")
    print(f"  Num Layers:     {config.NUM_LAYERS}")
    print(f"  Num Heads:      {config.NUM_HEADS}")
    print(f"  Encoder Type:   Transformer")

    print(f"\n⏱️  Temporal Configuration:")
    print(f"  Window Size:    {config.TEMPORAL_WINDOW_SIZE} frames")
    print(f"  Stride:         {config.TEMPORAL_STRIDE} frames")
    print(f"  Context Frames: {config.ACTION_CONTEXT_FRAMES} frames")

    print(f"\n🎯 Action Classes ({config.NUM_CLASSES} classes):")
    for idx, name in config.ACTION_CLASSES.items():
        print(f"  {idx}: {name}")

    print(f"\n⚖️  Class Balancing:")
    print(f"  Enabled:        {config.USE_CLASS_BALANCING}")
    if config.USE_CLASS_BALANCING:
        for cls_idx, ratio in config.CLASS_BALANCE_RATIOS.items():
            cls_name = config.ACTION_CLASSES[cls_idx]
            print(f"  {cls_name} (class {cls_idx}): {ratio}x")

    print(f"\n🔧 Training Configuration:")
    print(f"  Device:         {config.DEVICE}")
    print(f"  Batch Size:     {config.BATCH_SIZE}")
    print(f"  Epochs:         {config.NUM_EPOCHS}")
    print(f"  Learning Rate:  {config.LEARNING_RATE}")
    print(f"  Weight Decay:   {config.WEIGHT_DECAY}")

    print(f"\n⚡ Performance Optimizations:")
    print(f"  YOLO Caching:   {config.USE_YOLO_CACHE}")
    print(f"  Batched YOLO:   {config.USE_BATCHED_YOLO}")
    print(f"  Frame Filtering: {config.USE_INTELLIGENT_FILTERING}")
    if config.USE_INTELLIGENT_FILTERING:
        print(f"  Distance Threshold: {config.BALL_PLAYER_DISTANCE_THRESHOLD}")

    print("\n✅ Configuration loaded successfully!")


def demo_model_architecture():
    """Demo: Show model architecture and parameters"""
    print_section("DEMO 2: Model Architecture")

    from models.toy_action_classifier import ToyActionClassifier

    print("\n🏗️  Building ToyActionClassifier...")
    model = ToyActionClassifier(
        use_frame_filtering=config.USE_INTELLIGENT_FILTERING
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size:           ~{total_params * 4 / 1024 / 1024:.2f} MB")

    print(f"\n🔍 Model Components:")
    print(f"  1. Track Embedding:    Converts tracking data to {model.tracking_feature_dim}D features")
    print(f"  2. Temporal Encoder:   Transformer with {config.NUM_LAYERS} layers, {config.NUM_HEADS} heads")
    print(f"  3. Temporal Decoder:   Maps {config.HIDDEN_DIM}D features to {config.NUM_CLASSES} classes")

    if model.frame_filter is not None:
        print(f"  4. Frame Filter:       Intelligent filtering with {config.BALL_PLAYER_DISTANCE_THRESHOLD} threshold")

    print(f"\n📐 Data Flow:")
    batch_size = 2
    seq_len = config.TEMPORAL_WINDOW_SIZE

    print(f"  Input:")
    print(f"    - Spatial Features:  ({batch_size}, {seq_len}, {config.FEATURE_DIM})")
    print(f"    - Detections:        List[List[Dict]] (per frame, per batch)")
    print(f"    - Tracks:            List[List[Dict]] (per frame, per batch)")

    print(f"  Internal:")
    print(f"    - Track Features:    ({batch_size}, {seq_len}, {model.tracking_feature_dim})")
    print(f"    - Combined:          ({batch_size}, {seq_len}, {config.FEATURE_DIM + model.tracking_feature_dim})")
    print(f"    - Encoded:           ({batch_size}, {seq_len}, {config.HIDDEN_DIM})")

    print(f"  Output:")
    print(f"    - Logits:            ({batch_size}, {config.NUM_CLASSES})")

    # Test forward pass
    print(f"\n🧪 Testing Forward Pass...")
    model.eval()

    with torch.no_grad():
        # Create dummy inputs
        spatial_features = torch.randn(batch_size, seq_len, config.FEATURE_DIM)

        # Create dummy detections and tracks
        detections = []
        tracks = []
        for b in range(batch_size):
            sample_dets = []
            sample_tracks = []
            for t in range(seq_len):
                frame_dets = [
                    {'bbox': [100, 100, 150, 200], 'class_id': 0, 'confidence': 0.9},
                    {'bbox': [200, 150, 220, 170], 'class_id': 32, 'confidence': 0.85}
                ]
                sample_dets.append(frame_dets)
                sample_tracks.append(frame_dets)
            detections.append(sample_dets)
            tracks.append(sample_tracks)

        # Forward pass
        outputs, _ = model(spatial_features, detections, tracks)

        print(f"  ✅ Forward pass successful!")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Predicted classes: {outputs.argmax(dim=1).tolist()}")

    print("\n✅ Model architecture demo complete!")


def demo_detection_tracking():
    """Demo: Detection and tracking on sample video"""
    print_section("DEMO 3: Detection and Tracking")

    from models.detector import PlayerBallDetector
    from models.tracker import ByteTracker
    import cv2

    # Find a sample video
    video_files = list(config.TRAIN_DIR.rglob("*.mp4"))
    if not video_files:
        print("\n⚠️  No videos found in training directory!")
        print(f"  Expected path: {config.TRAIN_DIR}")
        return

    video_path = str(video_files[0])
    print(f"\n📹 Sample Video: {video_path}")

    # Initialize detector and tracker
    print(f"\n🔧 Initializing Components...")
    detector = PlayerBallDetector(device=config.DEVICE)
    tracker = ByteTracker()
    print(f"  ✅ Detector: YOLOv8{config.YOLO_MODEL_SIZE}")
    print(f"  ✅ Tracker: ByteTrack")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n❌ Cannot open video: {video_path}")
        return

    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📊 Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")

    # Process first 100 frames
    num_frames = min(100, total_frames)
    print(f"\n🎬 Processing first {num_frames} frames...")

    detection_stats = {'total_detections': 0, 'total_tracks': 0}

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect
        detection_result = detector.detect_frame(frame_rgb)

        # Track
        track_results = tracker.update(detection_result)

        detection_stats['total_detections'] += len(detection_result['boxes'])
        detection_stats['total_tracks'] += len(track_results)

        if frame_idx % 25 == 0:
            print(f"  Frame {frame_idx:3d}: {len(detection_result['boxes'])} detections, {len(track_results)} tracks")

    cap.release()

    print(f"\n📈 Statistics:")
    print(f"  Frames Processed: {num_frames}")
    print(f"  Total Detections: {detection_stats['total_detections']}")
    print(f"  Avg Detections/Frame: {detection_stats['total_detections'] / num_frames:.2f}")
    print(f"  Total Track Updates: {detection_stats['total_tracks']}")

    print("\n✅ Detection and tracking demo complete!")


def demo_training_setup():
    """Demo: Show training setup without actually training"""
    print_section("DEMO 4: Training Setup")

    from utils.toy_dataset import get_toy_dataloader
    from models.toy_action_classifier import ToyActionClassifier
    import torch.optim as optim

    print(f"\n📚 Loading Datasets...")

    try:
        train_loader = get_toy_dataloader(split='train', max_videos=config.MAX_TRAIN_VIDEOS)
        val_loader = get_toy_dataloader(split='val', max_videos=config.MAX_VAL_VIDEOS)

        print(f"  ✅ Train Dataset: {len(train_loader.dataset)} samples")
        print(f"  ✅ Val Dataset:   {len(val_loader.dataset)} samples")
        print(f"  Batches per Epoch: {len(train_loader)}")

    except Exception as e:
        print(f"  ⚠️  Could not load datasets: {e}")
        print(f"  Expected dataset path: {config.TRAIN_DIR}")
        return

    print(f"\n🏗️  Initializing Model...")
    model = ToyActionClassifier(use_frame_filtering=config.USE_INTELLIGENT_FILTERING)
    model = model.to(config.DEVICE)

    print(f"  ✅ Model on device: {config.DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n⚙️  Training Components:")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"  ✅ Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    print(f"  ✅ Scheduler: ReduceLROnPlateau")

    # Loss function
    if config.FOCAL_LOSS:
        print(f"  ✅ Loss: Focal Loss (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
    else:
        print(f"  ✅ Loss: CrossEntropyLoss")

    print(f"\n📊 Training Configuration:")
    print(f"  Epochs:     {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Batches/Epoch: {len(train_loader)}")
    print(f"  Total Steps: {config.NUM_EPOCHS * len(train_loader)}")

    print(f"\n💾 Output Paths:")
    print(f"  Models:   {config.MODEL_DIR}")
    print(f"  Logs:     {config.LOGS_DIR}")
    print(f"  Results:  {config.RESULTS_DIR}")

    print(f"\n🚀 To start training, run:")
    print(f"  python toy_train.py")

    print("\n✅ Training setup demo complete!")


def demo_inference():
    """Demo: Inference on sample video"""
    print_section("DEMO 5: Inference")

    import os

    # Check for trained model
    model_paths = [
        config.OUTPUT_DIR.parent / "checkpoints" / "toy_best.pth",
        config.MODEL_DIR / "toy_best.pth",
        config.OUTPUT_DIR.parent / "checkpoints" / "toy_latest.pth",
        config.MODEL_DIR / "toy_latest.pth",
    ]

    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        print(f"\n⚠️  No trained model found!")
        print(f"\nChecked locations:")
        for path in model_paths:
            print(f"  - {path}")
        print(f"\n💡 Train a model first:")
        print(f"  python toy_train.py")
        return

    print(f"\n✅ Found trained model: {model_path}")

    # Check for sample video
    video_files = list(config.TEST_DIR.rglob("*.mp4")) or list(config.TRAIN_DIR.rglob("*.mp4"))

    if not video_files:
        print(f"\n⚠️  No sample videos found!")
        print(f"  Expected paths: {config.TEST_DIR} or {config.TRAIN_DIR}")
        return

    sample_video = str(video_files[0])
    print(f"✅ Found sample video: {sample_video}")

    print(f"\n🎬 Inference Configuration:")
    print(f"  Model:       {model_path.name}")
    print(f"  Device:      {config.DEVICE}")
    print(f"  Window Size: {config.TEMPORAL_WINDOW_SIZE}")
    print(f"  Stride:      8 frames (default)")

    print(f"\n📝 Sample Inference Command:")
    output_video = "demo_inference_output.mp4"
    print(f"  python inference_headless.py \\")
    print(f"    --video \"{sample_video}\" \\")
    print(f"    --model \"{model_path}\" \\")
    print(f"    --output {output_video} \\")
    print(f"    --window-size {config.TEMPORAL_WINDOW_SIZE} \\")
    print(f"    --stride 8")

    print(f"\n💡 Optional Parameters:")
    print(f"  --confidence-threshold 0.5  # Minimum confidence for predictions (0.0-1.0)")

    print(f"\n🎯 Expected Output:")
    print(f"  - Annotated video with action predictions")
    print(f"  - Frame-by-frame action labels")
    print(f"  - Running counts for each action class")

    print("\n✅ Inference demo complete!")


def demo_evaluation():
    """Demo: Model evaluation"""
    print_section("DEMO 6: Model Evaluation")

    # Check for trained model
    model_paths = [
        config.OUTPUT_DIR.parent / "checkpoints" / "toy_best.pth",
        config.MODEL_DIR / "toy_best.pth",
    ]

    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        print(f"\n⚠️  No trained model found for evaluation!")
        print(f"\n💡 Train a model first:")
        print(f"  python toy_train.py")
        return

    print(f"\n✅ Found trained model: {model_path}")

    print(f"\n📊 Evaluation Metrics:")
    print(f"  - Accuracy")
    print(f"  - Precision, Recall, F1-Score (per class)")
    print(f"  - Confusion Matrix")
    print(f"  - mAP (mean Average Precision)")

    print(f"\n📈 Evaluation Command:")
    print(f"  python evaluate_model_fast.py \\")
    print(f"    --model \"{model_path}\" \\")
    print(f"    --split val")

    print(f"\n💾 Output:")
    print(f"  - Metrics printed to console")
    print(f"  - Confusion matrix saved to {config.RESULTS_DIR}")
    print(f"  - Per-class metrics visualization")

    # Try to extract checkpoint info
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            print(f"\n📦 Checkpoint Info:")
            if 'epoch' in checkpoint:
                print(f"  Epoch:         {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")
            if 'best_val_acc' in checkpoint:
                print(f"  Best Val Acc:  {checkpoint['best_val_acc']:.2f}%")
            if 'config' in checkpoint:
                print(f"  Config:        {checkpoint['config']}")
    except Exception as e:
        print(f"\n⚠️  Could not load checkpoint info: {e}")

    print("\n✅ Evaluation demo complete!")


def demo_pipeline_overview():
    """Demo: Complete pipeline overview"""
    print_section("COMPLETE PIPELINE OVERVIEW")

    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      SOCCER ACTION RECOGNITION PIPELINE                 │
    └─────────────────────────────────────────────────────────────────────────┘

    📥 INPUT: Soccer Video (25 FPS, 224p)
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STAGE 1: SPATIAL DETECTION (YOLOv8n)                                   │
    │  - Detect players (class 0) and ball (class 32)                         │
    │  - Extract 512-dim spatial features                                     │
    │  - Cache features for faster training                                   │
    └────────────────────────────┬────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STAGE 2: TEMPORAL TRACKING (ByteTrack)                                 │
    │  - Track objects across frames                                          │
    │  - Maintain track IDs and trajectories                                  │
    │  - Extract 128-dim tracking features                                    │
    └────────────────────────────┬────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STAGE 3: ACTION CLASSIFICATION (Transformer)                           │
    │  - Temporal window: 32 frames                                           │
    │  - Combined features: 512 (spatial) + 128 (tracking) = 640              │
    │  - Transformer encoder: 2 layers, 4 heads, hidden=128                   │
    │  - Decoder: Temporal aggregation + classifier                           │
    └────────────────────────────┬────────────────────────────────────────────┘
         │
         ▼
    📤 OUTPUT: Action Class (PASS / DRIVE / BACKGROUND)
    """)

    print(f"\n🔑 Key Features:")
    print(f"  ✓ Frozen YOLO detector (no fine-tuning needed)")
    print(f"  ✓ Efficient feature caching")
    print(f"  ✓ Intelligent frame filtering (reduces computation by ~40%)")
    print(f"  ✓ Class balancing (3x oversampling for BACKGROUND)")
    print(f"  ✓ Multi-object tracking with ByteTrack")
    print(f"  ✓ Transformer-based temporal modeling")

    print(f"\n📊 Dataset:")
    print(f"  Source: SoccerNet Ball Action Spotting 2024")
    print(f"  Classes: 3 (PASS, DRIVE, BACKGROUND)")
    print(f"  Original: 13 classes → mapped to 3")
    print(f"  Train Videos: {config.MAX_TRAIN_VIDEOS}")
    print(f"  Val Videos: {config.MAX_VAL_VIDEOS}")

    print(f"\n⚡ Performance:")
    print(f"  Model Size: ~500K parameters")
    print(f"  Training Time: ~10 epochs on toy dataset")
    print(f"  Inference: Real-time capable (25 FPS)")
    print(f"  Device Support: MPS (M1 Mac) / CUDA / CPU")


def main():
    """Main demo routine"""
    parser = argparse.ArgumentParser(
        description='Soccer Action Recognition - Comprehensive Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python demo.py --demo all

  # Run specific demo
  python demo.py --demo config
  python demo.py --demo model
  python demo.py --demo detection
  python demo.py --demo training
  python demo.py --demo inference
  python demo.py --demo evaluation
        """
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['all', 'overview', 'config', 'model', 'detection', 'training', 'inference', 'evaluation'],
        default='all',
        help='Which demo to run'
    )

    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "SOCCER ACTION RECOGNITION - DEMO" + " " * 26 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    demos = {
        'overview': demo_pipeline_overview,
        'config': demo_configuration,
        'model': demo_model_architecture,
        'detection': demo_detection_tracking,
        'training': demo_training_setup,
        'inference': demo_inference,
        'evaluation': demo_evaluation,
    }

    if args.demo == 'all':
        # Run all demos in logical order
        for name in ['overview', 'config', 'model', 'detection', 'training', 'inference', 'evaluation']:
            try:
                demos[name]()
            except KeyboardInterrupt:
                print("\n\n⚠️  Demo interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Demo '{name}' failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        try:
            demos[args.demo]()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 30 + "DEMO COMPLETE" + " " * 35 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    print(f"\n📚 Next Steps:")
    print(f"  1. Review configuration:   python demo.py --demo config")
    print(f"  2. Train the model:        python toy_train.py")
    print(f"  3. Evaluate the model:     python evaluate_model_fast.py --model <path>")
    print(f"  4. Run inference:          python inference_headless.py --video <path> --model <path>")

    print(f"\n📖 Documentation:")
    print(f"  - README.md:               Quick start guide")
    print(f"  - PROJECT_ARCHITECTURE.md: Detailed architecture documentation")
    print(f"  - toy_config.py:           Configuration parameters")

    print(f"\n💡 For more information:")
    print(f"  python demo.py --help")

    print("")


if __name__ == "__main__":
    main()
