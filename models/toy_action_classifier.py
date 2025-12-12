"""
Toy Action Classifier - Optimized version for quick experimentation
Uses reduced dimensions and intelligent frame filtering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import toy_config as config
from models.action_classifier import TemporalEncoder, TemporalDecoder, TrackFeatureExtractor
from utils.frame_filter import IntelligentFrameFilter


class ToyActionClassifier(nn.Module):
    """
    Optimized action classifier for toy experiments.
    Includes intelligent frame filtering to reduce computational load.
    """

    def __init__(
        self,
        spatial_feature_dim: int = None,
        tracking_feature_dim: int = 128,
        hidden_dim: int = None,
        num_layers: int = None,
        num_heads: int = None,
        num_classes: int = None,
        dropout: float = None,
        encoder_type: str = "transformer",
        use_frame_filtering: bool = True,
        distance_threshold: float = None
    ):
        """
        Args:
            spatial_feature_dim: Dimension of spatial features from YOLO
            tracking_feature_dim: Dimension of tracking features
            hidden_dim: Hidden dimension (default from toy_config: 128)
            num_layers: Number of encoder layers (default from toy_config: 2)
            num_heads: Number of attention heads (default from toy_config: 4)
            num_classes: Number of action classes
            dropout: Dropout rate
            encoder_type: 'transformer'
            use_frame_filtering: Whether to use intelligent frame filtering
            distance_threshold: Ball-player distance threshold for filtering
        """
        super().__init__()

        # Use toy_config defaults
        self.spatial_feature_dim = spatial_feature_dim or config.FEATURE_DIM
        self.tracking_feature_dim = tracking_feature_dim
        self.hidden_dim = hidden_dim or config.HIDDEN_DIM  # 128
        self.num_classes = num_classes or config.NUM_CLASSES
        self.use_frame_filtering = use_frame_filtering
        dropout = dropout or config.DROPOUT

        # Frame filter
        if self.use_frame_filtering and config.USE_INTELLIGENT_FILTERING:
            self.frame_filter = IntelligentFrameFilter(
                distance_threshold=distance_threshold or config.BALL_PLAYER_DISTANCE_THRESHOLD,
                min_frames=config.MIN_FRAMES_PER_WINDOW,
                strategy=config.FILTER_STRATEGY
            )
        else:
            self.frame_filter = None

        # Feature extraction for tracking
        self.track_embedding = TrackFeatureExtractor(
            output_dim=tracking_feature_dim,
            dropout=dropout
        )

        # Combined feature dimension
        combined_dim = self.spatial_feature_dim + tracking_feature_dim

        # Temporal encoder (Transformer as requested)
        self.encoder = TemporalEncoder(
            input_dim=combined_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers or config.NUM_LAYERS,  # 2
            num_heads=num_heads or config.NUM_HEADS,      # 4
            dropout=dropout,
            encoder_type=encoder_type  # "transformer"
        )

        # Temporal decoder
        self.decoder = TemporalDecoder(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=1,
            dropout=dropout
        )

        # Statistics tracking
        self.filtering_stats = {
            "total_frames_before": 0,
            "total_frames_after": 0,
            "num_sequences": 0
        }

    def forward(
        self,
        spatial_features: torch.Tensor,
        detections: List[List[Dict]],
        tracks: List[List[Dict]],
        return_sequence: bool = False,
        track_statistics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with intelligent frame filtering.

        Args:
            spatial_features: Spatial features from YOLO (B, T, D_spatial)
            detections: List of detections for each frame in each batch
            tracks: List of tracks for each sample in batch
            return_sequence: If True, return per-frame predictions
            track_statistics: If True, return filtering statistics

        Returns:
            Class logits (B, C) if return_sequence=False, else (B, T, C)
            Optional statistics dictionary
        """
        batch_size, seq_len = spatial_features.shape[:2]
        device = spatial_features.device

        # Apply frame filtering if enabled
        if self.frame_filter is not None and self.training:
            filtered_spatial = []
            filtered_detections = []
            filtered_tracks = []

            for b in range(batch_size):
                # Filter frames for this sample
                sample_spatial = spatial_features[b]  # (T, D)
                sample_detections = detections[b] if b < len(detections) else []
                sample_tracks = tracks[b] if b < len(tracks) else []

                f_spatial, f_detections, kept_indices = self.frame_filter.filter_sequence(
                    sample_spatial,
                    sample_detections,
                    sample_tracks
                )

                # Update statistics
                if track_statistics:
                    self.filtering_stats["total_frames_before"] += seq_len
                    self.filtering_stats["total_frames_after"] += len(kept_indices)
                    self.filtering_stats["num_sequences"] += 1

                filtered_spatial.append(f_spatial)
                filtered_detections.append(f_detections)

                # Filter tracks based on kept indices
                if len(kept_indices) > 0 and len(sample_tracks) > 0:
                    f_tracks = [sample_tracks[i] for i in kept_indices if i < len(sample_tracks)]
                else:
                    f_tracks = []
                filtered_tracks.append(f_tracks)

            # Pad sequences to same length
            max_len = max(len(f) for f in filtered_spatial)
            if max_len == 0:
                # No frames kept, return zeros
                if return_sequence:
                    return torch.zeros(batch_size, 1, self.num_classes).to(device), None
                else:
                    return torch.zeros(batch_size, self.num_classes).to(device), None

            padded_spatial = []
            for f_spatial in filtered_spatial:
                if len(f_spatial) < max_len:
                    padding = torch.zeros(max_len - len(f_spatial), f_spatial.shape[-1]).to(device)
                    f_spatial = torch.cat([f_spatial, padding], dim=0)
                padded_spatial.append(f_spatial)

            spatial_features = torch.stack(padded_spatial)  # (B, T', D)
            detections = filtered_detections
            tracks = filtered_tracks
            seq_len = max_len

        # Extract tracking features
        track_features = []
        for b, sample_tracks in enumerate(tracks):
            sample_track_feats = self.track_embedding(sample_tracks, seq_len)
            track_features.append(sample_track_feats)

        track_features = torch.stack(track_features).to(device)  # (B, T, D_track)

        # Combine spatial and tracking features
        combined_features = torch.cat([spatial_features, track_features], dim=-1)

        # Encode temporal sequence
        encoded = self.encoder(combined_features)  # (B, T, H)

        # Decode to action logits
        logits = self.decoder(encoded)  # (B, T, C)

        # Prepare statistics
        stats = None
        if track_statistics and self.filtering_stats["num_sequences"] > 0:
            total_before = self.filtering_stats["total_frames_before"]
            total_after = self.filtering_stats["total_frames_after"]
            stats = {
                "frames_before": total_before,
                "frames_after": total_after,
                "reduction_percentage": ((total_before - total_after) / total_before * 100) if total_before > 0 else 0,
                "avg_frames_per_seq": total_after / self.filtering_stats["num_sequences"]
            }

        if return_sequence:
            return logits, stats
        else:
            # Global temporal pooling
            logits = logits.mean(dim=1)  # (B, C)
            return logits, stats

    def get_filtering_statistics(self) -> Dict:
        """Get accumulated filtering statistics."""
        if self.filtering_stats["num_sequences"] == 0:
            return {"message": "No sequences processed yet"}

        total_before = self.filtering_stats["total_frames_before"]
        total_after = self.filtering_stats["total_frames_after"]

        return {
            "total_sequences": self.filtering_stats["num_sequences"],
            "total_frames_before": total_before,
            "total_frames_after": total_after,
            "frames_removed": total_before - total_after,
            "reduction_percentage": ((total_before - total_after) / total_before * 100) if total_before > 0 else 0,
            "avg_frames_per_seq_before": total_before / self.filtering_stats["num_sequences"],
            "avg_frames_per_seq_after": total_after / self.filtering_stats["num_sequences"],
        }

    def reset_statistics(self):
        """Reset filtering statistics."""
        self.filtering_stats = {
            "total_frames_before": 0,
            "total_frames_after": 0,
            "num_sequences": 0
        }


if __name__ == "__main__":
    print("Testing Toy Action Classifier...")
    print(f"Device: {config.DEVICE}")
    print(f"Hidden dim: {config.HIDDEN_DIM}")
    print(f"Num layers: {config.NUM_LAYERS}")
    print(f"Num heads: {config.NUM_HEADS}")

    # Create model
    model = ToyActionClassifier(
        spatial_feature_dim=512,
        use_frame_filtering=True
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    batch_size = 4
    seq_len = 32  # Reduced temporal window
    spatial_features = torch.randn(batch_size, seq_len, 512)

    # Create dummy detections and tracks
    detections = []
    tracks = []

    for b in range(batch_size):
        sample_detections = []
        sample_tracks = []

        for t in range(seq_len):
            # Vary ball-player distance
            # First half: close (should be kept)
            # Second half: far (should be filtered)
            if t < seq_len // 2:
                ball_pos = 200 + t * 2
                player_pos = 200 + t * 2
            else:
                ball_pos = 200
                player_pos = 500  # Far away

            frame_dets = [
                {'bbox': [player_pos, 150, player_pos + 50, 250], 'class_id': 0},
                {'bbox': [ball_pos, 200, ball_pos + 20, 220], 'class_id': 32}
            ]
            sample_detections.append(frame_dets)
            sample_tracks.append(frame_dets)

        detections.append(sample_detections)
        tracks.append(sample_tracks)

    # Forward pass
    print("\nTesting forward pass with frame filtering...")
    model.train()  # Set to training mode to enable filtering
    with torch.no_grad():
        output, stats = model(
            spatial_features,
            detections,
            tracks,
            return_sequence=False,
            track_statistics=True
        )
        print(f"Output shape: {output.shape}")
        if stats:
            print(f"Filtering stats: {stats}")

        # Test sequence output
        output_seq, _ = model(
            spatial_features,
            detections,
            tracks,
            return_sequence=True
        )
        print(f"Output shape (sequence): {output_seq.shape}")

    # Get accumulated statistics
    print("\nAccumulated filtering statistics:")
    accum_stats = model.get_filtering_statistics()
    for key, value in accum_stats.items():
        print(f"  {key}: {value}")

    print("\nToy action classifier test completed!")
