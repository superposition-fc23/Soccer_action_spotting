"""
Intelligent Frame Filtering Module
Filters frames based on ball-player proximity to reduce computational load
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


class IntelligentFrameFilter:
    """
    Filters frames based on ball-player proximity.
    Only keeps frames where the ball is close to at least one player.
    """

    def __init__(
        self,
        distance_threshold: float = 150.0,
        min_frames: int = 8,
        strategy: str = "ball_player_proximity"
    ):
        """
        Args:
            distance_threshold: Maximum distance (in pixels) between ball and closest player
            min_frames: Minimum number of frames to keep even if all are far
            strategy: Filtering strategy to use
        """
        self.distance_threshold = distance_threshold
        self.min_frames = min_frames
        self.strategy = strategy

    def filter_sequence(
        self,
        frames: torch.Tensor,
        detections: List[List[Dict]],
        tracks: Optional[List[List[Dict]]] = None
    ) -> Tuple[torch.Tensor, List[List[Dict]], List[int]]:
        """
        Filter frames based on ball-player proximity.

        Args:
            frames: Tensor of frames (T, C, H, W) or spatial features (T, D)
            detections: List of detections for each frame
            tracks: Optional list of tracks for each frame

        Returns:
            filtered_frames: Filtered frames
            filtered_detections: Filtered detections
            kept_indices: Indices of frames that were kept
        """
        if self.strategy == "ball_player_proximity":
            return self._filter_by_ball_player_proximity(frames, detections, tracks)
        elif self.strategy == "none":
            return frames, detections, list(range(len(detections)))
        else:
            raise ValueError(f"Unknown filtering strategy: {self.strategy}")

    def _filter_by_ball_player_proximity(
        self,
        frames: torch.Tensor,
        detections: List[List[Dict]],
        tracks: Optional[List[List[Dict]]] = None
    ) -> Tuple[torch.Tensor, List[List[Dict]], List[int]]:
        """
        Filter frames where ball is far from all players.
        """
        kept_indices = []

        for frame_idx, frame_detections in enumerate(detections):
            # Find ball and players in this frame
            ball_detections = [d for d in frame_detections if d.get('class_id') == 32 or d.get('class') == 'ball']
            player_detections = [d for d in frame_detections if d.get('class_id') == 0 or d.get('class') == 'person']

            # If no ball or no players, skip this frame
            if not ball_detections or not player_detections:
                continue

            # Get ball centroid
            ball_bbox = ball_detections[0]['bbox']  # Assume single ball
            ball_centroid = np.array([
                (ball_bbox[0] + ball_bbox[2]) / 2,
                (ball_bbox[1] + ball_bbox[3]) / 2
            ])

            # Calculate distance to closest player
            min_distance = float('inf')
            for player in player_detections:
                player_bbox = player['bbox']
                player_centroid = np.array([
                    (player_bbox[0] + player_bbox[2]) / 2,
                    (player_bbox[1] + player_bbox[3]) / 2
                ])

                # Euclidean distance
                distance = np.linalg.norm(ball_centroid - player_centroid)
                min_distance = min(min_distance, distance)

            # Keep frame if ball is close to at least one player
            if min_distance <= self.distance_threshold:
                kept_indices.append(frame_idx)

        # Ensure we keep at least min_frames
        if len(kept_indices) < self.min_frames and len(detections) > 0:
            # Keep frames with smallest distances
            all_distances = []
            for frame_idx, frame_detections in enumerate(detections):
                ball_detections = [d for d in frame_detections if d.get('class_id') == 32 or d.get('class') == 'ball']
                player_detections = [d for d in frame_detections if d.get('class_id') == 0 or d.get('class') == 'person']

                if not ball_detections or not player_detections:
                    all_distances.append((frame_idx, float('inf')))
                    continue

                ball_bbox = ball_detections[0]['bbox']
                ball_centroid = np.array([
                    (ball_bbox[0] + ball_bbox[2]) / 2,
                    (ball_bbox[1] + ball_bbox[3]) / 2
                ])

                min_dist = float('inf')
                for player in player_detections:
                    player_bbox = player['bbox']
                    player_centroid = np.array([
                        (player_bbox[0] + player_bbox[2]) / 2,
                        (player_bbox[1] + player_bbox[3]) / 2
                    ])
                    dist = np.linalg.norm(ball_centroid - player_centroid)
                    min_dist = min(min_dist, dist)

                all_distances.append((frame_idx, min_dist))

            # Sort by distance and take top min_frames
            all_distances.sort(key=lambda x: x[1])
            kept_indices = [idx for idx, _ in all_distances[:self.min_frames]]
            kept_indices.sort()  # Keep temporal order

        # Filter frames and detections
        if len(kept_indices) == 0:
            # Return empty tensors if no frames kept
            if frames.dim() == 4:  # (T, C, H, W)
                filtered_frames = torch.zeros((0, *frames.shape[1:]))
            else:  # (T, D)
                filtered_frames = torch.zeros((0, frames.shape[-1]))
            filtered_detections = []
        else:
            filtered_frames = frames[kept_indices]
            filtered_detections = [detections[i] for i in kept_indices]

        # Also filter tracks if provided
        filtered_tracks = None
        if tracks is not None and len(kept_indices) > 0:
            filtered_tracks = [tracks[i] for i in kept_indices]

        return filtered_frames, filtered_detections, kept_indices

    def get_statistics(self, original_count: int, filtered_count: int) -> Dict[str, float]:
        """
        Get filtering statistics.

        Args:
            original_count: Number of frames before filtering
            filtered_count: Number of frames after filtering

        Returns:
            Dictionary with statistics
        """
        reduction_ratio = 1.0 - (filtered_count / max(original_count, 1))

        return {
            "original_frames": original_count,
            "filtered_frames": filtered_count,
            "frames_removed": original_count - filtered_count,
            "reduction_ratio": reduction_ratio,
            "reduction_percentage": reduction_ratio * 100
        }


class AdaptiveFrameFilter(IntelligentFrameFilter):
    """
    Adaptive frame filter that adjusts threshold based on action labels.
    For action frames, use a larger threshold to capture context.
    """

    def __init__(
        self,
        distance_threshold: float = 150.0,
        action_distance_threshold: float = 250.0,
        min_frames: int = 8,
        strategy: str = "ball_player_proximity"
    ):
        """
        Args:
            distance_threshold: Default distance threshold
            action_distance_threshold: Larger threshold for action frames
            min_frames: Minimum frames to keep
            strategy: Filtering strategy
        """
        super().__init__(distance_threshold, min_frames, strategy)
        self.action_distance_threshold = action_distance_threshold

    def filter_with_labels(
        self,
        frames: torch.Tensor,
        detections: List[List[Dict]],
        labels: List[int],
        tracks: Optional[List[List[Dict]]] = None
    ) -> Tuple[torch.Tensor, List[List[Dict]], List[int], List[int]]:
        """
        Filter frames with adaptive thresholding based on labels.

        Args:
            frames: Tensor of frames
            detections: List of detections
            labels: List of action labels for each frame
            tracks: Optional tracks

        Returns:
            Tuple of (filtered_frames, filtered_detections, kept_indices, filtered_labels)
        """
        kept_indices = []

        for frame_idx, (frame_detections, label) in enumerate(zip(detections, labels)):
            # Use larger threshold for action frames (non-background)
            threshold = self.action_distance_threshold if label != 4 else self.distance_threshold

            ball_detections = [d for d in frame_detections if d.get('class_id') == 32 or d.get('class') == 'ball']
            player_detections = [d for d in frame_detections if d.get('class_id') == 0 or d.get('class') == 'person']

            if not ball_detections or not player_detections:
                # Keep action frames even without detections
                if label != 4:
                    kept_indices.append(frame_idx)
                continue

            ball_bbox = ball_detections[0]['bbox']
            ball_centroid = np.array([
                (ball_bbox[0] + ball_bbox[2]) / 2,
                (ball_bbox[1] + ball_bbox[3]) / 2
            ])

            min_distance = float('inf')
            for player in player_detections:
                player_bbox = player['bbox']
                player_centroid = np.array([
                    (player_bbox[0] + player_bbox[2]) / 2,
                    (player_bbox[1] + player_bbox[3]) / 2
                ])
                distance = np.linalg.norm(ball_centroid - player_centroid)
                min_distance = min(min_distance, distance)

            if min_distance <= threshold:
                kept_indices.append(frame_idx)

        # Ensure minimum frames
        if len(kept_indices) < self.min_frames and len(detections) > 0:
            remaining = self.min_frames - len(kept_indices)
            all_indices = list(range(len(detections)))
            missing_indices = [i for i in all_indices if i not in kept_indices]
            kept_indices.extend(missing_indices[:remaining])
            kept_indices.sort()

        # Filter everything
        if len(kept_indices) == 0:
            if frames.dim() == 4:
                filtered_frames = torch.zeros((0, *frames.shape[1:]))
            else:
                filtered_frames = torch.zeros((0, frames.shape[-1]))
            filtered_detections = []
            filtered_labels = []
        else:
            filtered_frames = frames[kept_indices]
            filtered_detections = [detections[i] for i in kept_indices]
            filtered_labels = [labels[i] for i in kept_indices]

        return filtered_frames, filtered_detections, kept_indices, filtered_labels


if __name__ == "__main__":
    print("Testing Intelligent Frame Filter...")

    # Create dummy data
    num_frames = 100
    frames = torch.randn(num_frames, 512)  # Dummy spatial features

    # Create dummy detections
    detections = []
    for i in range(num_frames):
        frame_dets = []
        # Ball at random position
        frame_dets.append({
            'bbox': [200 + i, 200, 220 + i, 220],
            'class_id': 32,
            'class': 'ball'
        })
        # Player close to ball (first 50 frames)
        if i < 50:
            frame_dets.append({
                'bbox': [190 + i, 190, 240 + i, 280],
                'class_id': 0,
                'class': 'person'
            })
        else:
            # Player far from ball (last 50 frames)
            frame_dets.append({
                'bbox': [500, 500, 550, 600],
                'class_id': 0,
                'class': 'person'
            })
        detections.append(frame_dets)

    # Test filtering
    filter = IntelligentFrameFilter(distance_threshold=150, min_frames=8)
    filtered_frames, filtered_dets, kept_indices = filter.filter_sequence(
        frames, detections
    )

    print(f"\nOriginal frames: {num_frames}")
    print(f"Filtered frames: {len(filtered_frames)}")
    print(f"Frames removed: {num_frames - len(filtered_frames)}")
    print(f"Reduction: {(1 - len(filtered_frames)/num_frames)*100:.1f}%")
    print(f"Kept indices: {kept_indices[:10]}... (showing first 10)")

    # Test statistics
    stats = filter.get_statistics(num_frames, len(filtered_frames))
    print(f"\nStatistics: {stats}")

    print("\nFrame filter test completed!")
