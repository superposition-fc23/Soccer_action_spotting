"""
Toy Training Script for Quick Experimentation
Optimized for M1 Mac with intelligent frame filtering
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))
import toy_config as config
from models.toy_action_classifier import ToyActionClassifier
from models.detector import PlayerBallDetector
from models.tracker import ByteTracker
from utils.metrics import MetricsTracker


# Focal loss replaced with standard CrossEntropyLoss due to NaN issues
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs.reshape(-1, self.num_classes),
            targets.reshape(-1),
            reduction='none'
        )
        # Add epsilon for numerical stability
        pt = torch.exp(-ce_loss).clamp(min=1e-8, max=1.0)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
        loss = (focal_weight * ce_loss).mean()
        return loss


class ToyTrainer:
    """
    Simplified trainer for quick experimentation with toy settings
    """

    def __init__(self):
        self.device = config.DEVICE
        print(f"[TOY TRAINER] Using device: {self.device}")
        print(f"[TOY TRAINER] Hidden dim: {config.HIDDEN_DIM}")
        print(f"[TOY TRAINER] Temporal window: {config.TEMPORAL_WINDOW_SIZE}")
        print(f"[TOY TRAINER] Intelligent filtering: {config.USE_INTELLIGENT_FILTERING}")

        # Initialize detector and tracker
        print("[TOY TRAINER] Initializing detector and tracker...")
        # Note: PlayerBallDetector auto-constructs model name from config
        self.detector = PlayerBallDetector(
            device=self.device
        )
        self.tracker = ByteTracker()

        # Initialize toy action classifier with filtering
        print("[TOY TRAINER] Initializing toy action classifier...")
        self.model = ToyActionClassifier(
            use_frame_filtering=config.USE_INTELLIGENT_FILTERING,
            distance_threshold=config.BALL_PLAYER_DISTANCE_THRESHOLD
        ).to(self.device)

        # Feature projection and normalization for YOLO features
        self.feature_proj = None
        self.feature_norm = None

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[TOY TRAINER] Total parameters: {total_params:,}")
        print(f"[TOY TRAINER] Trainable parameters: {trainable_params:,}")

        # Loss function
        if config.FOCAL_LOSS:
            self.criterion = FocalLoss(
                alpha=config.FOCAL_ALPHA,
                gamma=config.FOCAL_GAMMA,
                num_classes=config.NUM_CLASSES
            )
            print("[TOY TRAINER] Using Focal Loss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("[TOY TRAINER] Using Cross Entropy Loss")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # Initialize metrics trackers
        self.train_metrics = MetricsTracker(config.ACTION_CLASSES, device=self.device)
        self.val_metrics = MetricsTracker(config.ACTION_CLASSES, device=self.device)
        print("[TOY TRAINER] Metrics trackers initialized")
        
        # TensorBoard
        if config.TENSORBOARD:
            log_dir = config.LOGS_DIR / f"toy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Ensure log directory exists before creating the writer
            try:
                config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            self.writer = SummaryWriter(log_dir)

            # Hold last epoch metrics for summaries
            self.last_train_results = None
            self.last_val_results = None
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        # Best accuracy and early stopping state
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()

        # Reset metrics tracker
        self.train_metrics.reset()

        total_loss = 0
        correct = 0
        total = 0

        # Show progress bar when in debug mode to see batch-level progress
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", disable=not config.DEBUG)

        for batch_idx, batch in enumerate(pbar):
            try:
                # Extract batch data
                videos = batch['video']  # (B, T, C, H, W)
                labels = batch['label']  # (B,)

                batch_size, seq_len = videos.shape[:2]

                # Process each video through detector and tracker
                spatial_features_list = []
                detections_list = []
                tracks_list = []

                for b in range(batch_size):
                    video = videos[b]  # (T, C, H, W)
                    label_idx = labels[b].item()

                    # Create cache key based on video metadata
                    cache_key = f"sample_{batch_idx}_{b}_label_{label_idx}.pt"
                    cache_path = config.YOLO_CACHE_DIR / cache_key

                    # Try to load from cache (project cached raw features to model feature dim)
                    if config.USE_YOLO_CACHE and cache_path.exists():
                        cached_data = torch.load(cache_path, map_location='cpu')
                        frame_detections = cached_data.get('detections', [])
                        frame_tracks = cached_data.get('tracks', [])
                        raw_cached_feats = cached_data.get('features', [])
                        frame_features = []
                        for raw_f in raw_cached_feats:
                            if raw_f is None:
                                frame_features.append(torch.zeros(config.FEATURE_DIM, device=self.device))
                                continue
                            # raw_f is CPU tensor of pooled channels
                            # Ensure raw_f is at least 1D
                            if not hasattr(raw_f, 'shape') or raw_f.dim() == 0:
                                raw_f = raw_f.view(1) if hasattr(raw_f, 'view') else torch.tensor([raw_f])

                            raw_dim = raw_f.shape[0]
                            if raw_dim != config.FEATURE_DIM:
                                # initialize projection if needed
                                if self.feature_proj is None:
                                    feat_dim = raw_dim
                                    print(f"[INFO] Initializing feature projection from cache: {feat_dim} -> {config.FEATURE_DIM}")
                                    self.feature_proj = nn.Linear(feat_dim, config.FEATURE_DIM).to(self.device)
                                    self.feature_norm = nn.LayerNorm(config.FEATURE_DIM).to(self.device)
                                    self.optimizer.add_param_group({
                                        'params': list(self.feature_proj.parameters()) + list(self.feature_norm.parameters())
                                    })
                                f_proj = self.feature_proj(raw_f.to(self.device))
                                f_proj = self.feature_norm(f_proj)
                                frame_features.append(f_proj)
                            else:
                                frame_features.append(raw_f.to(self.device))
                    else:
                        # Process frames (with optional batching)
                        frame_features = []
                        
                        raw_features_to_save = []
                        frame_detections = []
                        frame_tracks = []

                        if config.USE_BATCHED_YOLO and seq_len > 1:
                            # BATCHED YOLO: Process all frames at once
                            frames_np = video.cpu().numpy().transpose(0, 2, 3, 1)  # (T, H, W, C)
                            frames_np = (frames_np * 255).astype('uint8')

                            # Batch inference through YOLO
                            for t in range(seq_len):
                                frame_np = frames_np[t]
                                detection_result = self.detector.detect_frame(frame_np, return_features=True)

                                # Track
                                track_results = self.tracker.update(detection_result)

                                # Convert detections
                                frame_det_list = []
                                for i in range(len(detection_result['boxes'])):
                                    frame_det_list.append({
                                        'bbox': detection_result['boxes'][i].tolist(),
                                        'class_id': int(detection_result['classes'][i]),
                                        'confidence': float(detection_result['scores'][i])
                                    })

                                frame_detections.append(frame_det_list)
                                frame_tracks.append(track_results)

                                # Extract features
                                if 'features' in detection_result:
                                    raw_features = detection_result['features']
                                    # Pool to (1, 1) spatial dims, keep channel dimension
                                    # Input: (C, H, W) -> unsqueeze(0) -> (1, C, H, W)
                                    # After pool: (1, C, 1, 1) -> squeeze(-1).squeeze(-1) -> (1, C) -> squeeze(0) -> (C,)
                                    feat = F.adaptive_avg_pool2d(raw_features.unsqueeze(0), (1, 1))
                                    feat = feat.squeeze(-1).squeeze(-1).squeeze(0)  # Carefully remove only spatial and batch dims

                                    # Ensure feat is at least 1D (should be shape [C])
                                    if feat.dim() == 0:
                                        feat = feat.unsqueeze(0)

                                    # Keep a CPU copy of the raw pooled vector for caching
                                    raw_feat_vec = feat.detach().cpu()

                                    # Initialize projection on first use
                                    if self.feature_proj is None:
                                        feat_dim = feat.shape[0]
                                        print(f"[INFO] Initializing feature projection: {feat_dim} -> {config.FEATURE_DIM}")
                                        self.feature_proj = nn.Linear(feat_dim, config.FEATURE_DIM).to(self.device)
                                        self.feature_norm = nn.LayerNorm(config.FEATURE_DIM).to(self.device)
                                        self.optimizer.add_param_group({
                                            'params': list(self.feature_proj.parameters()) + list(self.feature_norm.parameters())
                                        })
                                        print(f"[INFO] Added feature projection parameters to optimizer")

                                    # Ensure feature tensor is on the same device as the projection
                                    feat = feat.to(self.device)
                                    feat = self.feature_proj(feat)
                                    feat = self.feature_norm(feat)
                                    raw_features_to_save.append(raw_feat_vec)
                                else:
                                    feat = torch.zeros(config.FEATURE_DIM, device=self.device)
                                    raw_features_to_save.append(None)

                                frame_features.append(feat)
                        else:
                            # Sequential processing (fallback)
                            for t in range(seq_len):
                                frame = video[t]
                                frame_np = frame.cpu().numpy().transpose(1, 2, 0)
                                frame_np = (frame_np * 255).astype('uint8')

                                detection_result = self.detector.detect_frame(frame_np, return_features=True)
                                track_results = self.tracker.update(detection_result)

                                frame_det_list = []
                                for i in range(len(detection_result['boxes'])):
                                    frame_det_list.append({
                                        'bbox': detection_result['boxes'][i].tolist(),
                                        'class_id': int(detection_result['classes'][i]),
                                        'confidence': float(detection_result['scores'][i])
                                    })

                                frame_detections.append(frame_det_list)
                                frame_tracks.append(track_results)

                                # Extract features, handle missing features (which was quite a challenge) to make it more effective for Bytetrack and provide more representations.
                                # Posed challenges with training dropoffs.
                                if 'features' in detection_result:
                                    raw_features = detection_result['features']
                                    # Pool to (1, 1) spatial dims, keep channel dimension
                                    # Input: (C, H, W) -> unsqueeze(0) -> (1, C, H, W)
                                    # After pool: (1, C, 1, 1) -> squeeze(-1).squeeze(-1) -> (1, C) -> squeeze(0) -> (C,)
                                    feat = F.adaptive_avg_pool2d(raw_features.unsqueeze(0), (1, 1))
                                    feat = feat.squeeze(-1).squeeze(-1).squeeze(0)  # Carefully remove only spatial and batch dims

                                    # Ensure feat is at least 1D (should be shape [C])
                                    if feat.dim() == 0:
                                        feat = feat.unsqueeze(0)

                                    # Keep a CPU copy of the raw pooled vector for caching
                                    raw_feat_vec = feat.detach().cpu()

                                    if self.feature_proj is None:
                                        feat_dim = feat.shape[0]
                                        print(f"[INFO] Initializing feature projection: {feat_dim} -> {config.FEATURE_DIM}")
                                        self.feature_proj = nn.Linear(feat_dim, config.FEATURE_DIM).to(self.device)
                                        self.feature_norm = nn.LayerNorm(config.FEATURE_DIM).to(self.device)
                                        self.optimizer.add_param_group({
                                            'params': list(self.feature_proj.parameters()) + list(self.feature_norm.parameters())
                                        })

                                    # Ensure feature tensor is on the same device as the projection
                                    feat = feat.to(self.device)
                                    feat = self.feature_proj(feat)
                                    feat = self.feature_norm(feat)
                                    raw_features_to_save.append(raw_feat_vec)
                                else:
                                    feat = torch.zeros(config.FEATURE_DIM, device=self.device)
                                    raw_features_to_save.append(None)

                                frame_features.append(feat)

                        # For feature caching : Save raw pooled features to cache so future runs can project consistently
                        if config.USE_YOLO_CACHE:
                            try:
                                torch.save({
                                    'features': raw_features_to_save,
                                    'detections': frame_detections,
                                    'tracks': frame_tracks
                                }, cache_path)
                            except Exception:
                                pass

                    spatial_features_list.append(torch.stack(frame_features))
                    detections_list.append(frame_detections)
                    tracks_list.append(frame_tracks)

                # Stack spatial features
                spatial_features = torch.stack(spatial_features_list).to(self.device)

                # Debug: Print feature statistics on first batch
                if batch_idx == 0:
                    print(f"[DEBUG] Spatial features shape: {spatial_features.shape}")
                    print(f"[DEBUG] Spatial features stats: min={spatial_features.min():.4f}, max={spatial_features.max():.4f}, mean={spatial_features.mean():.4f}, std={spatial_features.std():.4f}")

                # Before forward pass, check if we have valid data
                if spatial_features.shape[1] == 0:
                    print(f"[WARNING] Batch {batch_idx}: No frames after filtering, skipping")
                    continue

                # Forward pass
                outputs, stats = self.model(
                    spatial_features,
                    detections_list,
                    tracks_list,
                    track_statistics=True
                )

                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    print(f"[ERROR] NaN detected in model outputs at batch {batch_idx}")
                    print(f"  Outputs: {outputs}")
                    print(f"  Spatial features shape: {spatial_features.shape}")
                    print(f"  Spatial features stats: min={spatial_features.min()}, max={spatial_features.max()}, mean={spatial_features.mean()}")
                    break

                # Compute loss
                labels = labels.to(self.device)
                loss = self.criterion(outputs, labels)

                # Check for NaN in loss
                if torch.isnan(loss):
                    print(f"[ERROR] NaN loss at batch {batch_idx}")
                    print(f"  Outputs: {outputs}")
                    print(f"  Labels: {labels}")
                    print(f"  Outputs stats: min={outputs.min()}, max={outputs.max()}, mean={outputs.mean()}")
                    break

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Update metrics tracker
                self.train_metrics.update(outputs, labels, loss.item())

                # Clip gradients FIRST to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Clip feature projection gradients if they exist
                if self.feature_proj is not None:
                    torch.nn.utils.clip_grad_norm_(self.feature_proj.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.feature_norm.parameters(), max_norm=1.0)

                # Check for NaN gradients AFTER clipping
                nan_grads = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient in {name} (after clipping!)")
                        print(f"  Grad stats: min={param.grad.min()}, max={param.grad.max()}")
                        nan_grads = True
                        break

                if nan_grads:
                    break

                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'filtered': f'{stats["reduction_percentage"]:.1f}%' if stats else 'N/A'
                })

                # Log to TensorBoard
                if self.writer and batch_idx % config.LOG_INTERVAL == 0:
                    step = epoch * len(train_loader) + batch_idx
                    self.writer.add_scalar('Train/Loss', loss.item(), step)
                    self.writer.add_scalar('Train/Accuracy', 100.*correct/total, step)
                    if stats:
                        self.writer.add_scalar('Train/FrameReduction', stats["reduction_percentage"], step)
                    try:
                        self.writer.flush()
                    except Exception:
                        pass

            except Exception as e:
                import traceback
                print(f"[ERROR] Batch {batch_idx} failed: {str(e)}")
                print(f"[ERROR] Traceback:")
                traceback.print_exc()
                if batch_idx < 5:  # Only print detailed traceback for first few errors
                    print(f"[DEBUG] Video shape: {videos.shape if 'videos' in locals() else 'N/A'}")
                    print(f"[DEBUG] Label shape: {labels.shape if 'labels' in locals() else 'N/A'}")
                continue

        # Compute all metrics
        train_results = self.train_metrics.compute()
        # Store last train results for external summary
        self.last_train_results = train_results

        # Log to TensorBoard
        if self.writer:
            # Scalar metrics
            self.writer.add_scalar('Loss/train', train_results['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_results['accuracy'], epoch)
            self.writer.add_scalar('F1/train_macro', train_results['macro_f1'], epoch)
            self.writer.add_scalar('mAP/train_at_1', train_results['map_at_1'], epoch)
            self.writer.add_scalar('mAP/train_tight_avg', train_results['tight_avg_map'], epoch)

            # Per-class metrics
            for i, (cls_name, f1) in enumerate(zip([config.ACTION_CLASSES[j] for j in range(len(config.ACTION_CLASSES))],
                                                    train_results['per_class_f1'])):
                self.writer.add_scalar(f'F1_per_class/train_{cls_name}', f1, epoch)

            # Confusion matrix image
            cm_img = self.train_metrics.plot_confusion_matrix(split_name="train")
            self.writer.add_image('ConfusionMatrix/train', cm_img, epoch, dataformats='HWC')

        # Print summary
        print(f"\n[TRAIN EPOCH {epoch}] Metrics:")
        print(f"  Loss: {train_results['loss']:.4f}")
        print(f"  Accuracy: {train_results['accuracy']:.2f}%")
        print(f"  Macro F1: {train_results['macro_f1']:.4f}")
        print(f"  mAP@1: {train_results['map_at_1']:.4f}")

        return train_results['loss'], train_results['accuracy']

    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()

        # Reset metrics tracker
        self.val_metrics.reset()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                try:
                    videos = batch['video']
                    labels = batch['label']
                    batch_size, seq_len = videos.shape[:2]

                    # Process similar to training
                    spatial_features_list = []
                    detections_list = []
                    tracks_list = []

                    for b in range(batch_size):
                        video = videos[b]
                        frame_features = []
                        frame_detections = []
                        frame_tracks = []

                        for t in range(seq_len):
                            frame = video[t]

                            # Detect (convert from CHW to HWC format for YOLO)
                            frame_np = frame.cpu().numpy().transpose(1, 2, 0)
                            frame_np = (frame_np * 255).astype('uint8')  # Convert back to 0-255 range

                            detection_result = self.detector.detect_frame(frame_np)

                            # Track (tracker returns list of track dicts)
                            track_results = self.tracker.update(detection_result)

                            # Convert detections to unified format for filtering/model
                            frame_det_list = []
                            for i in range(len(detection_result['boxes'])):
                                frame_det_list.append({
                                    'bbox': detection_result['boxes'][i].tolist(),
                                    'class_id': int(detection_result['classes'][i]),
                                    'confidence': float(detection_result['scores'][i])
                                })

                            frame_detections.append(frame_det_list)
                            frame_tracks.append(track_results)  # Tracks already in correct format
                            feat = torch.randn(config.FEATURE_DIM, device=self.device)
                            frame_features.append(feat)

                        spatial_features_list.append(torch.stack(frame_features))
                        detections_list.append(frame_detections)
                        tracks_list.append(frame_tracks)

                        # (metrics updated after outputs/loss computed)

                    spatial_features = torch.stack(spatial_features_list).to(self.device)
                    outputs, _ = self.model(spatial_features, detections_list, tracks_list)

                    labels = labels.to(self.device)
                    loss = self.criterion(outputs, labels)

                    # Update validation metrics after computing outputs and loss
                    self.val_metrics.update(outputs, labels.to(self.device), loss.item())
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                except Exception as e:
                    print(f"[ERROR] Validation batch failed: {str(e)}")
                    continue
        
        
        # Compute all metrics
        val_results = self.val_metrics.compute()
        # Store last validation results for external summary
        self.last_val_results = val_results

        # Log to TensorBoard
        if self.writer:
            # Scalar metrics
            self.writer.add_scalar('Loss/val', val_results['loss'], epoch)
            self.writer.add_scalar('Accuracy/val', val_results['accuracy'], epoch)
            self.writer.add_scalar('F1/val_macro', val_results['macro_f1'], epoch)
            self.writer.add_scalar('mAP/val_at_1', val_results['map_at_1'], epoch)
            self.writer.add_scalar('mAP/val_tight_avg', val_results['tight_avg_map'], epoch)

            # Per-class metrics
            for i, (cls_name, f1) in enumerate(zip([config.ACTION_CLASSES[j] for j in range(len(config.ACTION_CLASSES))],
                                                    val_results['per_class_f1'])):
                self.writer.add_scalar(f'F1_per_class/val_{cls_name}', f1, epoch)

            # Confusion matrix image
            cm_img = self.val_metrics.plot_confusion_matrix(split_name="validation")
            self.writer.add_image('ConfusionMatrix/val', cm_img, epoch, dataformats='HWC')

        # Print summary
        self.val_metrics.print_summary("validation")

        return val_results['loss'], val_results['accuracy']

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': {
                'hidden_dim': config.HIDDEN_DIM,
                'num_layers': config.NUM_LAYERS,
                'num_heads': config.NUM_HEADS,
                'temporal_window': config.TEMPORAL_WINDOW_SIZE,
                'use_filtering': config.USE_INTELLIGENT_FILTERING,
                'distance_threshold': config.BALL_PLAYER_DISTANCE_THRESHOLD
            }
        }

        # Save latest checkpoint
        checkpoint_path = config.MODEL_DIR / 'toy_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = config.MODEL_DIR / 'toy_best.pth'
            torch.save(checkpoint, best_path)
            print(f"[CHECKPOINT] Best model saved: {best_path}")

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*50)
        print("STARTING TOY TRAINING")
        print("="*50)
        print(f"Training videos: {config.MAX_TRAIN_VIDEOS}")
        print(f"Validation videos: {config.MAX_VAL_VIDEOS}")
        print(f"Epochs: {config.NUM_EPOCHS}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print("="*50 + "\n")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            self.current_epoch = epoch

            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # Validation
            val_loss, val_acc = self.validate(val_loader, epoch)
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)

            # Check for improvement
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
                print(f"  ✓ New best validation loss!")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"  ✓ New best validation accuracy!")

            # Save checkpoint
            if epoch % config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Per-epoch summary (concise)
            print(f"\n[Epoch {epoch} Summary]")
            if self.last_train_results is not None:
                tr = self.last_train_results
                print(f"  Train — Loss: {tr['loss']:.4f}, Acc: {tr['accuracy']:.2f}%, Macro F1: {tr['macro_f1']:.4f}")
                # Per-class F1
                per_class = tr.get('per_class_f1', [])
                if len(per_class) > 0:
                    cls_names = [config.ACTION_CLASSES[i] for i in range(len(per_class))]
                    per_class_str = ', '.join([f"{n}:{v:.3f}" for n, v in zip(cls_names, per_class)])
                    print(f"  Train Per-class F1: {per_class_str}")

            if self.last_val_results is not None:
                vr = self.last_val_results
                print(f"  Val   — Loss: {vr['loss']:.4f}, Acc: {vr['accuracy']:.2f}%, Macro F1: {vr['macro_f1']:.4f}")
                per_class = vr.get('per_class_f1', [])
                if len(per_class) > 0:
                    cls_names = [config.ACTION_CLASSES[i] for i in range(len(per_class))]
                    per_class_str = ', '.join([f"{n}:{v:.3f}" for n, v in zip(cls_names, per_class)])
                    print(f"  Val Per-class F1: {per_class_str}")

            # Flush writer to ensure TB picks up epoch-level scalars
            if self.writer:
                try:
                    self.writer.flush()
                except Exception:
                    pass

            # Early stopping
            if not is_best:
                self.patience_counter += 1
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n[EARLY STOPPING] No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
                    break

            # Print filtering statistics
            if config.USE_INTELLIGENT_FILTERING:
                stats = self.model.get_filtering_statistics()
                print(f"  Filtering: {stats.get('reduction_percentage', 0):.1f}% frames removed")

            print()

        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print("="*50 + "\n")

        if self.writer:
            self.writer.close()


def main():
    """Main entry point"""
    print("="*50)
    print("TOY EXPERIMENT - SOCCER ACTION CLASSIFICATION")
    print("="*50)
    print(f"Configuration:")
    print(f"  - Device: {config.DEVICE}")
    print(f"  - YOLO Model: yolov8{config.YOLO_MODEL_SIZE}")
    print(f"  - Hidden Dim: {config.HIDDEN_DIM}")
    print(f"  - Num Layers: {config.NUM_LAYERS}")
    print(f"  - Num Heads: {config.NUM_HEADS}")
    print(f"  - Temporal Window: {config.TEMPORAL_WINDOW_SIZE}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print(f"  - Frame Filtering: {config.USE_INTELLIGENT_FILTERING}")
    if config.USE_INTELLIGENT_FILTERING:
        print(f"  - Distance Threshold: {config.BALL_PLAYER_DISTANCE_THRESHOLD}px")
    print("="*50 + "\n")

    # Placeholder for data loaders
    from utils.toy_dataset import get_toy_dataloader
    train_loader = get_toy_dataloader(split='train', max_videos=config.MAX_TRAIN_VIDEOS)
    val_loader = get_toy_dataloader(split='val', max_videos=config.MAX_VAL_VIDEOS)

    # Initialize trainer
    trainer = ToyTrainer()

    # Start training
    trainer.train(train_loader, val_loader)

    

if __name__ == "__main__":
    main()
