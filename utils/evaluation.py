"""
Evaluation metrics and visualization for soccer action detection
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


class ActionEvaluator:
    """
    Evaluator for action classification performance
    """

    def __init__(self, class_names: Dict[int, str] = None):
        """
        Args:
            class_names: Mapping from class index to class name
        """
        self.class_names = class_names or config.ACTION_CLASSES
        self.predictions = []
        self.ground_truth = []

    def add_batch(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Add a batch of predictions and ground truth

        Args:
            predictions: Predicted labels (B,) or (B, T)
            ground_truth: Ground truth labels (B,) or (B, T)
        """
        predictions = predictions.flatten()
        ground_truth = ground_truth.flatten()

        self.predictions.extend(predictions)
        self.ground_truth.extend(ground_truth)

    def compute_metrics(self) -> Dict:
        """
        Compute all evaluation metrics

        Returns:
            Dictionary of metrics
        """
        preds = np.array(self.predictions)
        gt = np.array(self.ground_truth)

        # Overall accuracy
        accuracy = accuracy_score(gt, preds)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            gt, preds, average=None, labels=list(self.class_names.keys())
        )

        # Weighted average
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            gt, preds, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(gt, preds, labels=list(self.class_names.keys()))

        metrics = {
            'accuracy': accuracy,
            'precision_per_class': {self.class_names[i]: p for i, p in enumerate(precision)},
            'recall_per_class': {self.class_names[i]: r for i, r in enumerate(recall)},
            'f1_per_class': {self.class_names[i]: f for i, f in enumerate(f1)},
            'support_per_class': {self.class_names[i]: s for i, s in enumerate(support)},
            'precision_weighted': precision_avg,
            'recall_weighted': recall_avg,
            'f1_weighted': f1_avg,
            'confusion_matrix': cm
        }

        return metrics

    def print_report(self):
        """
        Print detailed classification report
        """
        preds = np.array(self.predictions)
        gt = np.array(self.ground_truth)

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)

        report = classification_report(
            gt, preds,
            target_names=[self.class_names[i] for i in sorted(self.class_names.keys())],
            labels=list(self.class_names.keys())
        )
        print(report)

    def plot_confusion_matrix(self, save_path: str = None, normalize: bool = True):
        """
        Plot confusion matrix

        Args:
            save_path: Path to save the figure
            normalize: Whether to normalize the confusion matrix
        """
        metrics = self.compute_metrics()
        cm = metrics['confusion_matrix']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=[self.class_names[i] for i in sorted(self.class_names.keys())],
            yticklabels=[self.class_names[i] for i in sorted(self.class_names.keys())]
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

    def plot_per_class_metrics(self, save_path: str = None):
        """
        Plot per-class precision, recall, and F1 scores

        Args:
            save_path: Path to save the figure
        """
        metrics = self.compute_metrics()

        classes = [self.class_names[i] for i in sorted(self.class_names.keys())]
        precision = [metrics['precision_per_class'][c] for c in classes]
        recall = [metrics['recall_per_class'][c] for c in classes]
        f1 = [metrics['f1_per_class'][c] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Action Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        else:
            plt.show()

    def reset(self):
        """Reset all stored predictions and ground truth"""
        self.predictions = []
        self.ground_truth = []


class TemporalActionSpotter:
    """
    Evaluate temporal action spotting performance
    """

    def __init__(self, tolerance_frames: int = 12):
        """
        Args:
            tolerance_frames: Tolerance window for matching predictions to ground truth
        """
        self.tolerance = tolerance_frames
        self.predictions = []  # List of (frame, class) tuples
        self.ground_truth = []  # List of (frame, class) tuples

    def add_video(self, pred_actions: List[Dict], gt_actions: List[Dict]):
        """
        Add predictions and ground truth for a video

        Args:
            pred_actions: List of predicted actions, each with 'frame' and 'class'
            gt_actions: List of ground truth actions
        """
        self.predictions.append(pred_actions)
        self.ground_truth.append(gt_actions)

    def compute_spotting_metrics(self) -> Dict:
        """
        Compute action spotting metrics (precision, recall at different tolerances)

        Returns:
            Dictionary of metrics
        """
        all_metrics = []

        for pred_actions, gt_actions in zip(self.predictions, self.ground_truth):
            # Match predictions to ground truth
            tp = 0  # True positives
            fp = 0  # False positives
            fn = len(gt_actions)  # False negatives (start with all GT)

            matched_gt = set()

            for pred in pred_actions:
                pred_frame = pred['frame']
                pred_class = pred['class']

                # Find matching ground truth
                matched = False
                for i, gt in enumerate(gt_actions):
                    if i in matched_gt:
                        continue

                    gt_frame = gt['frame']
                    gt_class = gt['class']

                    # Check if within tolerance and same class
                    if (abs(pred_frame - gt_frame) <= self.tolerance and
                        pred_class == gt_class):
                        tp += 1
                        fn -= 1
                        matched_gt.add(i)
                        matched = True
                        break

                if not matched:
                    fp += 1

            # Compute metrics for this video
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            all_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })

        # Average across all videos
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])

        total_tp = sum([m['tp'] for m in all_metrics])
        total_fp = sum([m['fp'] for m in all_metrics])
        total_fn = sum([m['fn'] for m in all_metrics])

        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }

    def plot_precision_recall_curve(self, save_path: str = None):
        """
        Plot precision-recall curve at different tolerance levels

        Args:
            save_path: Path to save the figure
        """
        tolerances = [6, 12, 24, 36, 48]  # Different tolerance windows
        precisions = []
        recalls = []

        for tol in tolerances:
            original_tol = self.tolerance
            self.tolerance = tol
            metrics = self.compute_spotting_metrics()
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            self.tolerance = original_tol

        plt.figure(figsize=(10, 6))
        plt.plot(tolerances, precisions, marker='o', label='Precision', linewidth=2)
        plt.plot(tolerances, recalls, marker='s', label='Recall', linewidth=2)
        plt.xlabel('Tolerance (frames)')
        plt.ylabel('Score')
        plt.title('Action Spotting Performance vs. Tolerance')
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Test evaluator
    print("Testing ActionEvaluator...")

    evaluator = ActionEvaluator()

    # Generate dummy predictions and ground truth
    np.random.seed(42)
    num_samples = 1000

    gt = np.random.randint(0, 5, num_samples)
    preds = gt.copy()
    # Add some errors
    error_idx = np.random.choice(num_samples, size=200, replace=False)
    preds[error_idx] = np.random.randint(0, 5, 200)

    evaluator.add_batch(preds, gt)

    # Compute and print metrics
    metrics = evaluator.compute_metrics()
    print("\nMetrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")

    evaluator.print_report()

    # Plot confusion matrix
    save_path = config.RESULTS_DIR / "test_confusion_matrix.png"
    evaluator.plot_confusion_matrix(save_path=save_path)

    # Plot per-class metrics
    save_path = config.RESULTS_DIR / "test_per_class_metrics.png"
    evaluator.plot_per_class_metrics(save_path=save_path)

    print("\nEvaluator test completed!")
