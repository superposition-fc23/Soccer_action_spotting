"""
Multi-object tracking using ByteTrack algorithm
Tracks players and ball across video frames
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


class Track:
    """
    Single object track
    """
    _id_counter = 0

    def __init__(self, bbox: np.ndarray, score: float, class_id: int, frame_id: int):
        """
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            score: Detection confidence
            class_id: Object class
            frame_id: Frame number
        """
        self.id = Track._id_counter
        Track._id_counter += 1

        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.start_frame = frame_id
        self.last_frame = frame_id

        self.history = {
            'bboxes': [bbox],
            'scores': [score],
            'frames': [frame_id]
        }

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        # Kalman filter for position prediction
        self.mean = self._bbox_to_state(bbox)
        self.covariance = np.eye(7) * 10

    def _bbox_to_state(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert bbox to state vector [cx, cy, w, h, vx, vy, vw]
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h, 0, 0, 0])

    def _state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state vector to bbox [x1, y1, x2, y2]
        """
        cx, cy, w, h = state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

    def predict(self):
        """
        Predict next position using Kalman filter
        """
        # Simple constant velocity model
        F = np.eye(7)
        F[0, 4] = 1  # cx += vx
        F[1, 5] = 1  # cy += vy
        F[2, 6] = 1  # w += vw

        Q = np.eye(7) * 0.1  # Process noise

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q

        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: np.ndarray, score: float, frame_id: int):
        """
        Update track with new detection
        """
        # Measurement update
        z = self._bbox_to_state(bbox)
        H = np.eye(7)
        R = np.eye(7) * 1.0  # Measurement noise

        y = z - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.mean = self.mean + K @ y
        self.covariance = (np.eye(7) - K @ H) @ self.covariance

        # Update track properties
        self.bbox = self._state_to_bbox(self.mean)
        self.score = score
        self.last_frame = frame_id
        self.time_since_update = 0
        self.hits += 1

        # Update history
        self.history['bboxes'].append(self.bbox)
        self.history['scores'].append(score)
        self.history['frames'].append(frame_id)

    def get_state(self) -> Dict:
        """
        Get current track state
        """
        return {
            'id': self.id,
            'bbox': self.bbox,
            'score': self.score,
            'class_id': self.class_id,
            'age': self.age,
            'hits': self.hits,
            'predicted': self._state_to_bbox(self.mean)
        }


class ByteTracker:
    """
    ByteTrack multi-object tracker
    """

    def __init__(
        self,
        max_age: int = None,
        min_hits: int = None,
        iou_threshold: float = None
    ):
        """
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum detections before confirming track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age or config.TRACKER_MAX_AGE
        self.min_hits = min_hits or config.TRACKER_MIN_HITS
        self.iou_threshold = iou_threshold or config.TRACKER_IOU_THRESHOLD

        self.tracks = []
        self.frame_count = 0

        # Separate thresholds for high and low confidence detections
        self.high_thresh = 0.6
        self.low_thresh = 0.3

    def update(self, detections: Dict) -> List[Dict]:
        """
        Update tracker with new detections

        Args:
            detections: Dictionary with 'boxes', 'scores', 'classes'

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Separate high and low confidence detections
        boxes = detections['boxes']
        scores = detections['scores']
        classes = detections['classes']

        if len(boxes) == 0:
            # No detections, just predict existing tracks
            for track in self.tracks:
                track.predict()
            self._remove_dead_tracks()
            return self._get_active_tracks()

        high_idx = scores >= self.high_thresh
        low_idx = (scores >= self.low_thresh) & (scores < self.high_thresh)

        high_dets = {
            'boxes': boxes[high_idx],
            'scores': scores[high_idx],
            'classes': classes[high_idx]
        }

        low_dets = {
            'boxes': boxes[low_idx],
            'scores': scores[low_idx],
            'classes': classes[low_idx]
        }

        # First association with high confidence detections
        unmatched_tracks, unmatched_dets = self._associate_detections(
            high_dets, self.iou_threshold
        )

        # Second association with low confidence detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        _, _ = self._associate_detections(
            low_dets, self.iou_threshold - 0.1, track_pool=remaining_tracks
        )

        # Remove dead tracks
        self._remove_dead_tracks()

        return self._get_active_tracks()

    def _associate_detections(
        self,
        detections: Dict,
        iou_thresh: float,
        track_pool: List[Track] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Associate detections to tracks using IoU matching

        Returns:
            (unmatched_tracks, unmatched_detections)
        """
        if track_pool is None:
            track_pool = self.tracks

        if len(track_pool) == 0 or len(detections['boxes']) == 0:
            # Create new tracks for all detections
            for i in range(len(detections['boxes'])):
                track = Track(
                    detections['boxes'][i],
                    detections['scores'][i],
                    int(detections['classes'][i]),
                    self.frame_count
                )
                self.tracks.append(track)
            return list(range(len(track_pool))), []

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(
            [t.bbox for t in track_pool],
            detections['boxes']
        )

        # Hungarian matching
        matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
            iou_matrix, iou_thresh
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            track_pool[track_idx].update(
                detections['boxes'][det_idx],
                detections['scores'][det_idx],
                self.frame_count
            )

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track = Track(
                detections['boxes'][det_idx],
                detections['scores'][det_idx],
                int(detections['classes'][det_idx]),
                self.frame_count
            )
            self.tracks.append(track)

        # Predict unmatched tracks
        for track_idx in unmatched_tracks:
            track_pool[track_idx].predict()

        return unmatched_tracks, unmatched_dets

    def _compute_iou_matrix(
        self,
        bboxes1: List[np.ndarray],
        bboxes2: np.ndarray
    ) -> np.ndarray:
        """
        Compute IoU matrix between two sets of bboxes
        """
        bboxes1 = np.array(bboxes1)
        iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))

        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                iou_matrix[i, j] = self._compute_iou(bbox1, bbox2)

        return iou_matrix

    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute IoU between two bboxes
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _linear_assignment(
        self,
        iou_matrix: np.ndarray,
        iou_thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Perform linear assignment with Hungarian algorithm
        """
        from scipy.optimize import linear_sum_assignment

        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix

        # Perform assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter by threshold
        matches = []
        unmatched_tracks = list(range(iou_matrix.shape[0]))
        unmatched_dets = list(range(iou_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_thresh:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    def _remove_dead_tracks(self):
        """
        Remove tracks that have not been updated for too long
        """
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

    def _get_active_tracks(self) -> List[Dict]:
        """
        Get all tracks that have sufficient hits
        """
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                active_tracks.append(track.get_state())

        return active_tracks

    def reset(self):
        """
        Reset tracker
        """
        self.tracks = []
        self.frame_count = 0
        Track._id_counter = 0


if __name__ == "__main__":
    # Test the tracker
    print("Testing ByteTracker...")

    tracker = ByteTracker()

    # Simulate detections over multiple frames
    for frame_id in range(10):
        # Create dummy detections
        num_dets = np.random.randint(3, 8)
        boxes = np.random.rand(num_dets, 4) * 100
        boxes[:, 2:] += boxes[:, :2]  # Convert to x2, y2
        scores = np.random.rand(num_dets) * 0.5 + 0.5
        classes = np.random.randint(0, 2, num_dets)

        detections = {
            'boxes': boxes,
            'scores': scores,
            'classes': classes
        }

        # Update tracker
        tracks = tracker.update(detections)
        print(f"\nFrame {frame_id}: {len(tracks)} active tracks")
        for track in tracks:
            print(f"  Track {track['id']}: bbox={track['bbox']}, hits={track['hits']}")

    print("\nTracker test completed!")
