"""
Unified Metrics Module
Centralized computation of all evaluation metrics for training and evaluation
Eliminates code duplication across scripts
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
from pathlib import Path


class MetricsTracker:
    """
    Unified metrics tracker for training and evaluation

    Computes all metrics:
    - Loss & Accuracy
    - Per-class F1 scores
    - Macro F1 score
    - Confusion matrix
    - mAP@1
    - Tight Average mAP
    - Per-class Average Precision
    """

    def __init__(self, class_names, device='cpu'):
        """
        Args:
            class_names: List or dict of class names
            device: Torch device
        """
        # Handle dict or list of class names
        if isinstance(class_names, dict):
            self.class_names = [class_names[i] for i in range(len(class_names))]
        else:
            self.class_names = class_names

        self.num_classes = len(self.class_names)
        self.device = device
        self.reset()

    def reset(self):
        """Reset accumulators for new epoch"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, outputs, labels, loss=None):
        """
        Update with batch results

        Args:
            outputs: Model outputs (logits), shape (batch_size, num_classes)
            labels: Ground truth labels, shape (batch_size,)
            loss: Optional loss value for this batch
        """
        # Compute probabilities
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        # Store results
        self.all_predictions.extend(preds.cpu().detach().numpy())
        self.all_labels.extend(labels.cpu().detach().numpy())
        self.all_probabilities.extend(probs.cpu().detach().numpy())

        # Track loss if provided
        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1

    def compute(self):
        """
        Compute all metrics from accumulated data

        Returns:
            dict: All computed metrics
        """
        preds = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probabilities)

        # Basic metrics
        accuracy = np.mean(preds == labels) * 100
        avg_loss = self.total_loss / self.num_batches if self.num_batches > 0 else 0.0

        # F1 scores
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=range(self.num_classes))
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))

        # mAP metrics
        map_at_1, tight_avg_map, per_class_ap = self._compute_map(labels, probs)

        # Classification report (string)
        cls_report = classification_report(
            labels, preds,
            target_names=self.class_names,
            zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'confusion_matrix': cm,
            'map_at_1': map_at_1,
            'tight_avg_map': tight_avg_map,
            'per_class_ap': per_class_ap,
            'classification_report': cls_report,
            'num_samples': len(labels)
        }

    def _compute_map(self, labels, probabilities):
        """
        Compute mAP metrics

        Args:
            labels: Ground truth labels (N,)
            probabilities: Predicted probabilities (N, num_classes)

        Returns:
            map_at_1: Mean Average Precision at rank 1 (essentially accuracy)
            tight_avg_map: Tight average mAP (macro-averaged over classes with samples)
            per_class_ap: Average Precision per class (list)
        """
        num_samples = len(labels)

        # One-hot encoding
        labels_one_hot = np.zeros((num_samples, self.num_classes))
        labels_one_hot[np.arange(num_samples), labels] = 1

        # Per-class Average Precision
        per_class_ap = []
        for class_idx in range(self.num_classes):
            y_true = labels_one_hot[:, class_idx]
            y_score = probabilities[:, class_idx]

            if np.sum(y_true) > 0:  # Only if class has positive samples
                ap = average_precision_score(y_true, y_score)
            else:
                ap = 0.0

            per_class_ap.append(ap)

        # mAP@1: Top-1 prediction accuracy
        top_1_predictions = np.argmax(probabilities, axis=1)
        map_at_1 = np.mean(top_1_predictions == labels)

        # Tight average mAP: Mean of per-class AP (only classes with samples)
        classes_with_samples = [i for i in range(self.num_classes) if np.sum(labels == i) > 0]
        if classes_with_samples:
            tight_avg_map = np.mean([per_class_ap[i] for i in classes_with_samples])
        else:
            tight_avg_map = 0.0

        return map_at_1, tight_avg_map, per_class_ap

    def plot_confusion_matrix(self, save_path=None, split_name="validation"):
        """
        Generate confusion matrix plot

        Args:
            save_path: Optional path to save image
            split_name: Name of data split (for title)

        Returns:
            numpy array: Image as RGB array (for TensorBoard)
        """
        metrics = self.compute()
        cm = metrics['confusion_matrix']

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {split_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Convert to RGB array for TensorBoard
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image.convert('RGB'))
        plt.close()

        return image_array

    def plot_per_class_f1(self, save_path=None, split_name="validation"):
        """
        Generate per-class F1 score bar plot

        Args:
            save_path: Optional path to save image
            split_name: Name of data split (for title)

        Returns:
            numpy array: Image as RGB array (for TensorBoard)
        """
        metrics = self.compute()
        per_class_f1 = metrics['per_class_f1']

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#95a5a6']  # Green, Blue, Gray
        bars = ax.bar(self.class_names, per_class_f1, color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_title(f'Per-Class F1 Scores - {split_name.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_xlabel('Action Class', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Convert to RGB array for TensorBoard
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image.convert('RGB'))
        plt.close()

        return image_array

    def plot_per_class_ap(self, save_path=None, split_name="validation"):
        """
        Generate per-class Average Precision bar plot

        Args:
            save_path: Optional path to save image
            split_name: Name of data split (for title)

        Returns:
            numpy array: Image as RGB array (for TensorBoard)
        """
        metrics = self.compute()
        per_class_ap = metrics['per_class_ap']

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#95a5a6']  # Green, Blue, Gray
        bars = ax.bar(self.class_names, per_class_ap, color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_title(f'Per-Class Average Precision - {split_name.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Precision', fontsize=12)
        ax.set_xlabel('Action Class', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Convert to RGB array for TensorBoard
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image.convert('RGB'))
        plt.close()

        return image_array

    def print_summary(self, split_name="validation"):
        """
        Print formatted summary of all metrics

        Args:
            split_name: Name of data split
        """
        metrics = self.compute()

        print(f"\n{'='*60}")
        print(f"{split_name.upper()} RESULTS ({metrics['num_samples']} samples)")
        print(f"{'='*60}")
        print(f"Loss:      {metrics['loss']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"\nMean Average Precision:")
        print(f"  mAP@1:           {metrics['map_at_1']:.4f}")
        print(f"  Tight Avg mAP:   {metrics['tight_avg_map']:.4f}")
        print(f"\nPer-Class F1 Scores:")
        for cls_name, f1 in zip(self.class_names, metrics['per_class_f1']):
            print(f"  {cls_name:12s}: {f1:.4f}")
        print(f"\nPer-Class Average Precision:")
        for cls_name, ap in zip(self.class_names, metrics['per_class_ap']):
            print(f"  {cls_name:12s}: {ap:.4f}")
        print(f"\nDetailed Classification Report:")
        print(metrics['classification_report'])


def log_metrics_to_tensorboard(writer, metrics, epoch, prefix='train'):
    """
    Log all metrics to TensorBoard

    Args:
        writer: TensorBoard SummaryWriter
        metrics: Metrics dict from MetricsTracker.compute()
        epoch: Current epoch number
        prefix: Metric prefix ('train' or 'val')
    """
    # Scalar metrics
    writer.add_scalar(f'Loss/{prefix}', metrics['loss'], epoch)
    writer.add_scalar(f'Accuracy/{prefix}', metrics['accuracy'], epoch)
    writer.add_scalar(f'F1/{prefix}_macro', metrics['macro_f1'], epoch)
    writer.add_scalar(f'mAP/{prefix}_at_1', metrics['map_at_1'], epoch)
    writer.add_scalar(f'mAP/{prefix}_tight_avg', metrics['tight_avg_map'], epoch)

    # Per-class F1 scores
    class_names = metrics.get('class_names', [f'Class_{i}' for i in range(len(metrics['per_class_f1']))])
    for i, (cls_name, f1) in enumerate(zip(class_names, metrics['per_class_f1'])):
        writer.add_scalar(f'F1_per_class/{prefix}_{cls_name}', f1, epoch)

    # Per-class AP scores
    for i, (cls_name, ap) in enumerate(zip(class_names, metrics['per_class_ap'])):
        writer.add_scalar(f'AP_per_class/{prefix}_{cls_name}', ap, epoch)


if __name__ == "__main__":
    # Example usage
    print("Testing MetricsTracker...")

    # Simulate some data
    class_names = {0: "PASS", 1: "DRIVE", 2: "BACKGROUND"}
    tracker = MetricsTracker(class_names)

    # Simulate batch updates
    for _ in range(10):
        outputs = torch.randn(8, 3)  # Batch of 8, 3 classes
        labels = torch.randint(0, 3, (8,))
        loss = torch.tensor(0.5)

        tracker.update(outputs, labels, loss)

    # Compute and print metrics
    metrics = tracker.compute()
    tracker.print_summary("test")

    # Generate plots
    cm_img = tracker.plot_confusion_matrix()
    f1_img = tracker.plot_per_class_f1()
    ap_img = tracker.plot_per_class_ap()

    print(f"\n[TEST] Generated images with shapes:")
    print(f"  Confusion matrix: {cm_img.shape}")
    print(f"  Per-class F1: {f1_img.shape}")
    print(f"  Per-class AP: {ap_img.shape}")
    print("\n[TEST] MetricsTracker test complete!")
