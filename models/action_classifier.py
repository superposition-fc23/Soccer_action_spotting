"""
Temporal Encoder-Decoder for Soccer Action Classification
Classifies actions (PASS, DRIVE) from temporal sequences of detections and features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


class TemporalEncoder(nn.Module):
    """
    Temporal encoder using Transformer or LSTM to encode temporal sequences
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        encoder_type: str = "transformer"
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads (for transformer)
            dropout: Dropout rate
            encoder_type: 'transformer' or 'lstm'
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if encoder_type == "transformer":
            # Positional encoding
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        elif encoder_type == "lstm":
            # Bidirectional LSTM
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
                batch_first=True
            )

        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, D)
            mask: Optional attention mask (B, T)

        Returns:
            Encoded features (B, T, H)
        """
        # Project input
        x = self.input_proj(x)  # (B, T, H)

        if self.encoder_type == "transformer":
            # Add positional encoding
            x = self.pos_encoder(x)

            # Apply transformer encoder
            if mask is not None:
                x = self.encoder(x, src_key_padding_mask=mask)
            else:
                x = self.encoder(x)

        else:  # LSTM
            # Apply LSTM
            x, _ = self.encoder(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalDecoder(nn.Module):
    """
    Temporal decoder for action classification
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_classes: Number of action classes
            num_layers: Number of decoder layers
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim // 2
            out_dim = hidden_dim // 2 if i < num_layers - 1 else num_classes
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features (B, T, H) or (B, H)

        Returns:
            Class logits (B, T, C) or (B, C)
        """
        return self.decoder(x)


class SoccerActionClassifier(nn.Module):
    """
    Complete model for soccer action classification
    Combines spatial features from YOLO with temporal modeling
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
        encoder_type: str = "transformer"
    ):
        """
        Args:
            spatial_feature_dim: Dimension of spatial features from YOLO
            tracking_feature_dim: Dimension of tracking features
            hidden_dim: Hidden dimension
            num_layers: Number of encoder/decoder layers
            num_heads: Number of attention heads
            num_classes: Number of action classes
            dropout: Dropout rate
            encoder_type: 'transformer' or 'lstm'
        """
        super().__init__()

        # Use config defaults if not specified
        self.spatial_feature_dim = spatial_feature_dim or config.FEATURE_DIM
        self.tracking_feature_dim = tracking_feature_dim
        self.hidden_dim = hidden_dim or config.HIDDEN_DIM
        self.num_classes = num_classes or config.NUM_CLASSES
        dropout = dropout or config.DROPOUT

        # Feature extraction for tracking information
        self.track_embedding = TrackFeatureExtractor(
            output_dim=tracking_feature_dim,
            dropout=dropout
        )

        # Combined feature dimension
        combined_dim = self.spatial_feature_dim + tracking_feature_dim

        # Temporal encoder
        self.encoder = TemporalEncoder(
            input_dim=combined_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers or config.NUM_LAYERS,
            num_heads=num_heads or config.NUM_HEADS,
            dropout=dropout,
            encoder_type=encoder_type
        )

        # Temporal decoder
        self.decoder = TemporalDecoder(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=2,
            dropout=dropout
        )

        # Temporal pooling for global action classification
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        spatial_features: torch.Tensor,
        tracks: List[List[Dict]],
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: Spatial features from YOLO (B, T, D_spatial)
            tracks: List of tracks for each sample in batch
                    Each track contains detection info per frame
            return_sequence: If True, return per-frame predictions

        Returns:
            Class logits (B, C) if return_sequence=False, else (B, T, C)
        """
        batch_size, seq_len = spatial_features.shape[:2]

        # Extract tracking features
        track_features = []
        for sample_tracks in tracks:
            sample_track_feats = self.track_embedding(sample_tracks, seq_len)
            track_features.append(sample_track_feats)

        track_features = torch.stack(track_features)  # (B, T, D_track)

        # Combine spatial and tracking features
        combined_features = torch.cat([spatial_features, track_features], dim=-1)

        # Encode temporal sequence
        encoded = self.encoder(combined_features)  # (B, T, H)

        # Decode to action logits
        logits = self.decoder(encoded)  # (B, T, C)

        if return_sequence:
            return logits
        else:
            # Global temporal pooling for sequence-level classification
            # Average over time dimension
            logits = logits.mean(dim=1)  # (B, C)
            return logits


class TrackFeatureExtractor(nn.Module):
    """
    Extract features from tracking information
    """

    def __init__(self, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.output_dim = output_dim

        # Features to extract:
        # - Bounding box position and size (4)
        # - Velocity (2)
        # - Acceleration (2)
        # - Distance to ball (1)
        # - Interaction with nearby players (variable)

        input_dim = 16  # Total feature dimension per frame

        # Add LayerNorm for numerical stability
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),  # Normalize after first layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),  # Normalize output
            nn.ReLU()
        )

        # He initialization for ReLU layers (prevents gradient explosion)
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Store device for tensor creation
        self._device = None

    def _get_device(self):
        """Get the device this module is on"""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def forward(self, tracks: List[Dict], seq_len: int) -> torch.Tensor:
        """
        Extract features from tracks for a sequence

        Args:
            tracks: List of track dictionaries per frame
            seq_len: Sequence length

        Returns:
            Track features (T, D)
        """
        device = self._get_device()
        features = []

        for frame_idx in range(seq_len):
            if frame_idx < len(tracks):
                frame_tracks = tracks[frame_idx]
                feat = self._extract_frame_features(frame_tracks, device)
            else:
                # Pad with zeros if no tracks (float32)
                feat = torch.zeros(16, device=device, dtype=torch.float32)

            features.append(feat)

        # Convert to tensor and pass through FC
        features = torch.stack(features)  # (T, input_dim)

        # Check for NaN/Inf in input features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"[ERROR] NaN/Inf in track input features before FC!")
            print(f"  Shape: {features.shape}")
            print(f"  Stats: min={features.min()}, max={features.max()}, mean={features.mean()}")
            print(f"  NaN count: {torch.isnan(features).sum()}")
            print(f"  Inf count: {torch.isinf(features).sum()}")
            # Replace NaN/Inf with zeros to prevent propagation
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        features = self.fc(features)  # (T, output_dim)

        # Check for NaN/Inf in output features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"[ERROR] NaN/Inf in track output features after FC!")
            print(f"  Shape: {features.shape}")
            print(f"  Stats: min={features.min()}, max={features.max()}, mean={features.mean()}")
            # Replace NaN/Inf with zeros
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _extract_frame_features(self, tracks: List[Dict], device) -> torch.Tensor:
        """
        Extract features from tracks in a single frame
        """
        if len(tracks) == 0:
            return torch.zeros(16, device=device, dtype=torch.float32)

        # Assumed frame dimensions for normalization (standard soccer video)
        # TODO: Pass actual frame dimensions from config
        FRAME_WIDTH = 1920.0
        FRAME_HEIGHT = 1080.0

        # Find ball and players
        ball_tracks = [t for t in tracks if t.get('class_id') == 32]
        player_tracks = [t for t in tracks if t.get('class_id') == 0]

        # Ball position (center) - NORMALIZE to [0, 1]
        if ball_tracks:
            ball_bbox = ball_tracks[0]['bbox']
            ball_pos = torch.tensor([
                ((ball_bbox[0] + ball_bbox[2]) / 2) / FRAME_WIDTH,
                ((ball_bbox[1] + ball_bbox[3]) / 2) / FRAME_HEIGHT
            ], device=device, dtype=torch.float32)
        else:
            ball_pos = torch.zeros(2, device=device, dtype=torch.float32)

        # Player statistics
        if player_tracks:
            import numpy as np
            player_bboxes_np = np.array([t['bbox'] for t in player_tracks], dtype=np.float32)
            player_bboxes = torch.from_numpy(player_bboxes_np).to(device)

            # Normalize player centers to [0, 1]
            player_centers = torch.stack([
                ((player_bboxes[:, 0] + player_bboxes[:, 2]) / 2) / FRAME_WIDTH,
                ((player_bboxes[:, 1] + player_bboxes[:, 3]) / 2) / FRAME_HEIGHT
            ], dim=1)

            # Average player position
            avg_player_pos = player_centers.mean(dim=0)

            # Distances to ball (add epsilon for numerical stability)
            if ball_tracks:
                distances = torch.norm(player_centers - ball_pos, dim=1) + 1e-8
                min_dist = distances.min()
                avg_dist = distances.mean()
                # Clamp to reasonable range
                min_dist = torch.clamp(min_dist, 0.0, 2.0)
                avg_dist = torch.clamp(avg_dist, 0.0, 2.0)
            else:
                min_dist = torch.tensor(0.0, device=device, dtype=torch.float32)
                avg_dist = torch.tensor(0.0, device=device, dtype=torch.float32)

            # Player spread (std of positions)
            # Add epsilon to prevent NaN when only 1 player or all at same position
            if len(player_tracks) > 1:
                player_spread = player_centers.std(dim=0, unbiased=False)  # Use population std
                # Replace any NaN with 0
                player_spread = torch.nan_to_num(player_spread, nan=0.0)
            else:
                player_spread = torch.zeros(2, device=device, dtype=torch.float32)

            # Number of players
            num_players = len(player_tracks)
        else:
            avg_player_pos = torch.zeros(2, device=device, dtype=torch.float32)
            min_dist = torch.tensor(0.0, device=device, dtype=torch.float32)
            avg_dist = torch.tensor(0.0, device=device, dtype=torch.float32)
            player_spread = torch.zeros(2, device=device, dtype=torch.float32)
            num_players = 0

        # Concatenate all features - ensure all are float32 and normalized
        features = torch.cat([
            ball_pos,  # 2 (normalized [0, 1])
            avg_player_pos,  # 2 (normalized [0, 1])
            torch.tensor([min_dist.item(), avg_dist.item()], device=device, dtype=torch.float32),  # 2 (normalized distances [0, 1])
            player_spread,  # 2 (normalized [0, 1])
            torch.tensor([num_players / 22.0], device=device, dtype=torch.float32),  # 1 (normalized)
            torch.zeros(7, device=device, dtype=torch.float32)  # Padding to 16
        ])

        # Clamp features to prevent extreme values
        features = torch.clamp(features, min=-10.0, max=10.0)

        return features


if __name__ == "__main__":
    # Test the action classifier
    print("Testing Soccer Action Classifier...")

    # Create model
    model = SoccerActionClassifier(
        spatial_feature_dim=512,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_classes=5
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    batch_size = 2
    seq_len = 64
    spatial_features = torch.randn(batch_size, seq_len, 512)

    # Create dummy tracks
    tracks = []
    for b in range(batch_size):
        sample_tracks = []
        for t in range(seq_len):
            frame_tracks = [
                {'bbox': [100, 100, 150, 200], 'class_id': 0},  # Player
                {'bbox': [200, 150, 250, 250], 'class_id': 0},  # Player
                {'bbox': [300, 200, 320, 220], 'class_id': 32}  # Ball
            ]
            sample_tracks.append(frame_tracks)
        tracks.append(sample_tracks)

    # Forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        # Sequence classification
        output = model(spatial_features, tracks, return_sequence=False)
        print(f"Output shape (sequence): {output.shape}")

        # Frame-level classification
        output_frames = model(spatial_features, tracks, return_sequence=True)
        print(f"Output shape (frames): {output_frames.shape}")

    print("\nAction classifier test completed!")
