"""
Fast Model Evaluation Script - Uses fewer videos for quick results
Computes metrics for presentation:
1. Loss & Accuracy (Train/Val)
2. Macro F1 Score
3. Per-Class F1 Scores
4. Confusion Matrix
5. mAP@1 (Mean Average Precision at rank 1)
6. Tight Average mAP
7. Per-Class Average Precision
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent))
import toy_config as config
from models.toy_action_classifier import ToyActionClassifier
from utils.toy_dataset import get_toy_dataloader


class FastModelEvaluator:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or config.DEVICE
        self.model_path = model_path

        # Load model
        print(f"[LOADING] Model from {model_path}")
        self.model = ToyActionClassifier(use_frame_filtering=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.checkpoint_metrics = checkpoint
        else:
            self.model.load_state_dict(checkpoint)
            self.checkpoint_metrics = None

        self.model.eval()
        print(f"[LOADED] Model loaded successfully")

        # Action classes
        self.action_classes_dict = config.ACTION_CLASSES
        self.action_classes = [config.ACTION_CLASSES[i] for i in range(len(config.ACTION_CLASSES))]
        self.num_classes = len(self.action_classes)

        # Initialize detector and tracker (needed for processing)
        from models.detector import PlayerBallDetector
        from models.tracker import ByteTracker
        self.detector = PlayerBallDetector(device=self.device)
        self.tracker = ByteTracker()
        print(f"[LOADED] Detector and tracker initialized")

    def _process_video_batch(self, videos):
        """Process videos through detector and tracker (like toy_train.py)"""
        import torch.nn.functional as F

        batch_size, seq_len = videos.shape[:2]
        spatial_features_list = []
        detections_list = []
        tracks_list = []

        for b in range(batch_size):
            video = videos[b]  # (T, C, H, W)

            # Reset tracker for each video
            self.tracker.reset()

            sequence_detections = []
            sequence_tracks = []
            frame_features = []

            for t in range(seq_len):
                frame_tensor = video[t]  # (C, H, W) in [0, 1]

                # Convert to numpy for detector (HWC, BGR, 0-255)
                frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
                frame_np = frame_np.astype(np.uint8)
                frame_np = frame_np[:, :, ::-1]  # RGB -> BGR

                # Detect
                detection_result = self.detector.detect_frame(frame_np, return_features=True)

                # Track
                track_results = self.tracker.update(detection_result)

                # Convert detections to expected format
                frame_det_list = []
                for i in range(len(detection_result['boxes'])):
                    frame_det_list.append({
                        'bbox': detection_result['boxes'][i].tolist(),
                        'class_id': int(detection_result['classes'][i]),
                        'confidence': float(detection_result['scores'][i])
                    })

                sequence_detections.append(frame_det_list)
                sequence_tracks.append(track_results)

                # Extract and process features
                if 'features' in detection_result:
                    raw_features = detection_result['features']
                    feat = F.adaptive_avg_pool2d(raw_features.unsqueeze(0), (1, 1)).squeeze()

                    # Initialize projection on first use
                    if not hasattr(self, 'feature_proj') or self.feature_proj is None:
                        feat_dim = feat.shape[0]
                        self.feature_proj = torch.nn.Linear(feat_dim, config.FEATURE_DIM).to(self.device)
                        self.feature_norm = torch.nn.LayerNorm(config.FEATURE_DIM).to(self.device)

                    feat = self.feature_proj(feat)
                    feat = self.feature_norm(feat)
                else:
                    feat = torch.zeros(config.FEATURE_DIM, device=self.device)

                frame_features.append(feat)

            # Stack features
            spatial_features = torch.stack(frame_features)  # (T, D)
            spatial_features_list.append(spatial_features)
            detections_list.append(sequence_detections)
            tracks_list.append(sequence_tracks)

        # Stack all batch items
        spatial_features_batch = torch.stack(spatial_features_list)  # (B, T, D)

        return spatial_features_batch, detections_list, tracks_list

    def _compute_map_metrics(self, labels, probabilities):
        """
        Compute mAP@1, tight average mAP, and per-class AP

        Args:
            labels: Ground truth labels (N,)
            probabilities: Predicted probabilities (N, num_classes)

        Returns:
            map_at_1: Mean Average Precision at rank 1
            tight_avg_map: Tight average mAP
            per_class_ap: Average Precision per class (list)
        """
        from sklearn.metrics import average_precision_score

        num_samples = len(labels)
        num_classes = probabilities.shape[1]

        # Convert labels to one-hot encoding
        labels_one_hot = np.zeros((num_samples, num_classes))
        labels_one_hot[np.arange(num_samples), labels] = 1

        # Compute per-class Average Precision
        per_class_ap = []
        for class_idx in range(num_classes):
            # Get binary labels and scores for this class
            y_true = labels_one_hot[:, class_idx]
            y_score = probabilities[:, class_idx]

            # Compute Average Precision for this class
            if np.sum(y_true) > 0:  # Only if class has positive samples
                ap = average_precision_score(y_true, y_score)
            else:
                ap = 0.0

            per_class_ap.append(ap)

        # mAP@1: Check if top-1 prediction is correct for each sample
        # This is essentially accuracy, but we compute it using AP framework
        top_1_predictions = np.argmax(probabilities, axis=1)
        map_at_1 = np.mean(top_1_predictions == labels)

        # Tight average mAP: Mean of per-class AP (macro averaging)
        # Only average over classes that have positive samples in the dataset
        classes_with_samples = [i for i in range(num_classes) if np.sum(labels == i) > 0]
        if classes_with_samples:
            tight_avg_map = np.mean([per_class_ap[i] for i in classes_with_samples])
        else:
            tight_avg_map = 0.0

        return map_at_1, tight_avg_map, per_class_ap

    def evaluate_dataloader(self, dataloader, split_name="validation", max_batches=None, time_range_minutes=None):
        """
        Evaluate model on a dataloader

        Args:
            dataloader: PyTorch dataloader
            split_name: Name of split (train/validation)
            max_batches: Maximum number of batches to evaluate
            time_range_minutes: Optional tuple (start_min, end_min) to filter samples by video timestamp
        """
        print(f"\n[EVALUATING] Running evaluation on {split_name} set...")
        if max_batches:
            print(f"[FAST MODE] Evaluating only {max_batches} batches for quick results")
        if time_range_minutes:
            print(f"[TIME FILTER] Only using samples from minutes {time_range_minutes[0]}-{time_range_minutes[1]}")

        all_predictions = []
        all_labels = []
        all_probabilities = []  # For mAP calculation
        total_loss = 0.0
        correct = 0
        total = 0
        skipped = 0

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
                if max_batches and batch_idx >= max_batches:
                    break

                # Get data
                videos = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                # Time-based filtering (optional)
                if time_range_minutes:
                    metadata = batch.get('metadata', [])
                    if metadata:
                        # Filter samples by time range
                        valid_indices = []
                        for idx, meta in enumerate(metadata):
                            # Parse gameTime (format: "MM:SS")
                            game_time = meta.get('gameTime', '0:00')
                            try:
                                time_parts = game_time.split(':')
                                minutes = int(time_parts[0])
                                # Check if within time range
                                if time_range_minutes[0] <= minutes < time_range_minutes[1]:
                                    valid_indices.append(idx)
                            except:
                                # If parsing fails, include this sample (don't filter)
                                valid_indices.append(idx)

                        # If we have valid indices, filter the batch
                        if valid_indices and len(valid_indices) < len(metadata):
                            videos = videos[valid_indices]
                            labels = labels[valid_indices]
                        elif not valid_indices:
                            # No samples in time range, skip this batch
                            skipped += len(metadata)
                            continue

                # Process videos through detector and tracker
                spatial_features, detections, tracks = self._process_video_batch(videos)

                # Forward pass
                outputs, _ = self.model(spatial_features, detections, tracks, track_statistics=True)

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Get probabilities (softmax) for mAP
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                # Predictions
                _, predicted = outputs.max(1)

                # Accumulate
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Compute metrics
        num_batches = min(batch_idx + 1, len(dataloader)) if max_batches else len(dataloader)
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * correct / total

        # F1 scores
        macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.action_classes,
            zero_division=0
        )

        # Mean Average Precision metrics
        map_at_1, tight_avg_map, per_class_ap = self._compute_map_metrics(
            all_labels, all_probabilities
        )

        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'num_samples': total,
            'map_at_1': map_at_1,
            'tight_avg_map': tight_avg_map,
            'per_class_ap': per_class_ap
        }

        return results

    def print_results(self, results, split_name="validation"):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print(f"{split_name.upper()} RESULTS ({results['num_samples']} samples)")
        print(f"{'='*60}")
        print(f"Loss:      {results['loss']:.4f}")
        print(f"Accuracy:  {results['accuracy']:.2f}%")
        print(f"Macro F1:  {results['macro_f1']:.4f}")
        print(f"\nMean Average Precision:")
        print(f"  mAP@1:           {results['map_at_1']:.4f}")
        print(f"  Tight Avg mAP:   {results['tight_avg_map']:.4f}")
        print(f"\nPer-Class F1 Scores:")
        for i, (cls_name, f1) in enumerate(zip(self.action_classes, results['per_class_f1'])):
            print(f"  {cls_name:12s}: {f1:.4f}")
        print(f"\nPer-Class Average Precision:")
        for i, (cls_name, ap) in enumerate(zip(self.action_classes, results['per_class_ap'])):
            print(f"  {cls_name:12s}: {ap:.4f}")

        print(f"\nDetailed Classification Report:")
        print(results['classification_report'])

    def plot_confusion_matrix(self, cm, output_path: str, split_name="validation"):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.action_classes,
            yticklabels=self.action_classes,
            cbar_kws={'label': 'Normalized Count'}
        )

        plt.title(f'Confusion Matrix - {split_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Confusion matrix: {output_path}")
        plt.close()

    def plot_per_class_f1(self, per_class_f1, output_path: str, split_name="validation"):
        """Plot per-class F1 scores"""
        plt.figure(figsize=(10, 6))

        colors = ['#2ecc71', '#3498db', '#95a5a6']  # Green, Blue, Gray
        bars = plt.bar(self.action_classes, per_class_f1, color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.title(f'Per-Class F1 Scores - {split_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.xlabel('Action Class', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Per-class F1 plot: {output_path}")
        plt.close()

    def plot_per_class_ap(self, per_class_ap, output_path: str, split_name="validation"):
        """Plot per-class Average Precision"""
        plt.figure(figsize=(10, 6))

        colors = ['#2ecc71', '#3498db', '#95a5a6']  # Green, Blue, Gray
        bars = plt.bar(self.action_classes, per_class_ap, color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.title(f'Per-Class Average Precision - {split_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.ylabel('Average Precision', fontsize=12)
        plt.xlabel('Action Class', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Per-class AP plot: {output_path}")
        plt.close()

    def plot_metrics_summary(self, train_results, val_results, output_path: str):
        """Plot summary comparison of train vs val metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Loss comparison
        metrics = ['Train', 'Val']
        losses = [train_results['loss'], val_results['loss']]
        axes[0].bar(metrics, losses, color=['#3498db', '#e74c3c'])
        axes[0].set_title('Loss Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(losses):
            axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

        # 2. Accuracy comparison
        accuracies = [train_results['accuracy'], val_results['accuracy']]
        axes[1].bar(metrics, accuracies, color=['#3498db', '#e74c3c'])
        axes[1].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(accuracies):
            axes[1].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Macro F1 comparison
        f1_scores = [train_results['macro_f1'], val_results['macro_f1']]
        axes[2].bar(metrics, f1_scores, color=['#3498db', '#e74c3c'])
        axes[2].set_title('Macro F1 Score Comparison', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Macro F1')
        axes[2].set_ylim(0, 1.0)
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(f1_scores):
            axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Metrics summary: {output_path}")
        plt.close()

    def save_results_json(self, train_results, val_results, output_path: str):
        """Save results to JSON file"""
        results_dict = {
            'model_path': self.model_path,
            'train': {
                'loss': float(train_results['loss']),
                'accuracy': float(train_results['accuracy']),
                'macro_f1': float(train_results['macro_f1']),
                'map_at_1': float(train_results['map_at_1']),
                'tight_avg_map': float(train_results['tight_avg_map']),
                'num_samples': int(train_results['num_samples']),
                'per_class_f1': {
                    cls_name: float(f1)
                    for cls_name, f1 in zip(self.action_classes, train_results['per_class_f1'])
                },
                'per_class_ap': {
                    cls_name: float(ap)
                    for cls_name, ap in zip(self.action_classes, train_results['per_class_ap'])
                }
            },
            'validation': {
                'loss': float(val_results['loss']),
                'accuracy': float(val_results['accuracy']),
                'macro_f1': float(val_results['macro_f1']),
                'map_at_1': float(val_results['map_at_1']),
                'tight_avg_map': float(val_results['tight_avg_map']),
                'num_samples': int(val_results['num_samples']),
                'per_class_f1': {
                    cls_name: float(f1)
                    for cls_name, f1 in zip(self.action_classes, val_results['per_class_f1'])
                },
                'per_class_ap': {
                    cls_name: float(ap)
                    for cls_name, ap in zip(self.action_classes, val_results['per_class_ap'])
                }
            }
        }

        # Add checkpoint info if available
        if self.checkpoint_metrics:
            if 'epoch' in self.checkpoint_metrics:
                results_dict['checkpoint_epoch'] = self.checkpoint_metrics['epoch']
            if 'best_val_loss' in self.checkpoint_metrics:
                results_dict['best_val_loss'] = float(self.checkpoint_metrics['best_val_loss'])

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"[SAVED] Results JSON: {output_path}")

    def run_fast_evaluation(self, output_dir: str = "evaluation_results", train_batches=50, val_batches=25,
                           time_range_minutes=None):
        """
        Run fast evaluation on subset of data

        Args:
            output_dir: Output directory for results
            train_batches: Number of train batches to evaluate
            val_batches: Number of val batches to evaluate
            time_range_minutes: Optional tuple (start_min, end_min) to limit evaluation to specific video time range
                               e.g., (10, 13) evaluates only minutes 10-13 of the video
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"[EVALUATION] Starting FAST evaluation...")
        print(f"[FAST MODE] Train batches: {train_batches}, Val batches: {val_batches}")
        if time_range_minutes:
            print(f"[TIME FILTER] Evaluating only minutes {time_range_minutes[0]}-{time_range_minutes[1]} of videos")
        print(f"[OUTPUT] Results will be saved to: {output_dir}")

        # Load dataloaders
        print(f"\n[LOADING] Loading datasets...")
        train_loader = get_toy_dataloader(split='train', max_videos=5)
        val_loader = get_toy_dataloader(split='val', max_videos=2)

        # Evaluate on train set (subset)
        train_results = self.evaluate_dataloader(train_loader, "train", max_batches=train_batches,
                                                 time_range_minutes=time_range_minutes)
        self.print_results(train_results, "train")

        # Evaluate on validation set (subset)
        val_results = self.evaluate_dataloader(val_loader, "validation", max_batches=val_batches,
                                               time_range_minutes=time_range_minutes)
        self.print_results(val_results, "validation")

        # Generate plots
        print(f"\n[GENERATING] Creating visualizations...")

        # Confusion matrices
        self.plot_confusion_matrix(
            train_results['confusion_matrix'],
            str(output_dir / 'confusion_matrix_train.png'),
            "train"
        )
        self.plot_confusion_matrix(
            val_results['confusion_matrix'],
            str(output_dir / 'confusion_matrix_val.png'),
            "validation"
        )

        # Per-class F1 scores
        self.plot_per_class_f1(
            train_results['per_class_f1'],
            str(output_dir / 'per_class_f1_train.png'),
            "train"
        )
        self.plot_per_class_f1(
            val_results['per_class_f1'],
            str(output_dir / 'per_class_f1_val.png'),
            "validation"
        )

        # Per-class Average Precision
        self.plot_per_class_ap(
            train_results['per_class_ap'],
            str(output_dir / 'per_class_ap_train.png'),
            "train"
        )
        self.plot_per_class_ap(
            val_results['per_class_ap'],
            str(output_dir / 'per_class_ap_val.png'),
            "validation"
        )

        # Metrics summary
        self.plot_metrics_summary(
            train_results,
            val_results,
            str(output_dir / 'metrics_summary.png')
        )

        # Save results to JSON
        self.save_results_json(
            train_results,
            val_results,
            str(output_dir / 'evaluation_results.json')
        )

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE!")
        print(f"{'='*60}")
        print(f"All results saved to: {output_dir}")
        print(f"\nGenerated files:")
        print(f"  - confusion_matrix_train.png")
        print(f"  - confusion_matrix_val.png")
        print(f"  - per_class_f1_train.png")
        print(f"  - per_class_f1_val.png")
        print(f"  - per_class_ap_train.png")
        print(f"  - per_class_ap_val.png")
        print(f"  - metrics_summary.png")
        print(f"  - evaluation_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fast model evaluation on subset')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results_fast', help='Output directory')
    parser.add_argument('--train-batches', type=int, default=50, help='Number of train batches to evaluate')
    parser.add_argument('--val-batches', type=int, default=25, help='Number of val batches to evaluate')
    parser.add_argument('--time-start', type=int, default=None, help='Start time in minutes (e.g., 10)')
    parser.add_argument('--time-end', type=int, default=None, help='End time in minutes (e.g., 13)')

    args = parser.parse_args()

    # Create time range tuple if both start and end are specified
    time_range = None
    if args.time_start is not None and args.time_end is not None:
        time_range = (args.time_start, args.time_end)

    evaluator = FastModelEvaluator(args.model)
    evaluator.run_fast_evaluation(args.output, args.train_batches, args.val_batches, time_range_minutes=time_range)
