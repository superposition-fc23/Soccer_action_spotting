"""
YOLO-based player and ball detection module
"""
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


class PlayerBallDetector:
    """
    Wrapper class for YOLO model to detect players and ball in soccer videos
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        conf_threshold: float = None,
        iou_threshold: float = None,
        pretrained: bool = True
    ):
        """
        Args:
            model_name: YOLO model name (e.g., 'yolov8x', 'yolov11x')
            device: Device to run model on ('cuda', 'mps', 'cpu')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            pretrained: Use pretrained weights
        """
        self.device = device or self._get_device()
        self.conf_threshold = conf_threshold or config.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.YOLO_IOU_THRESHOLD

        # Determine model name
        if model_name is None:
            version = config.YOLO_VERSION
            size = config.YOLO_MODEL_SIZE
            model_name = f"{version}{size}"

        # Load YOLO model
        if pretrained:
            model_path = f"{model_name}.pt"
        else:
            model_path = model_name

        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Classes of interest: person (0) and sports ball (32)
        self.classes_of_interest = [0, 32]  # COCO classes
        self.class_names = {0: "person", 32: "ball"}

    def _get_device(self) -> str:
        """
        Automatically detect best available device
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def detect_frame(
        self,
        frame: np.ndarray,
        return_features: bool = False
    ) -> Dict:
        """
        Detect players and ball in a single frame

        Args:
            frame: Input frame (H, W, C) in RGB format
            return_features: If True, return feature maps from backbone

        Returns:
            Dictionary containing:
                - boxes: Bounding boxes (N, 4) [x1, y1, x2, y2]
                - scores: Confidence scores (N,)
                - classes: Class labels (N,)
                - features: Feature maps (optional)
        """
        # Run detection
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes_of_interest,
            verbose=False,
            device=self.device
        )[0]

        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.array([])
        scores = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.array([])
        classes = results.boxes.cls.cpu().numpy() if len(results.boxes) > 0 else np.array([])

        output = {
            'boxes': boxes,
            'scores': scores,
            'classes': classes
        }

        # Extract features if requested
        if return_features:
            # Get features from the model's backbone
            # This will be used for action classification
            features = self._extract_features(frame)
            output['features'] = features

        return output

    def detect_batch(
        self,
        frames: np.ndarray,
        return_features: bool = False
    ) -> List[Dict]:
        """
        Detect players and ball in a batch of frames

        Args:
            frames: Batch of frames (B, H, W, C) or (T, H, W, C)
            return_features: If True, return feature maps

        Returns:
            List of detection dictionaries, one per frame
        """
        detections = []

        for frame in frames:
            detection = self.detect_frame(frame, return_features=return_features)
            detections.append(detection)

        return detections

    def _extract_features(self, frame: np.ndarray) -> torch.Tensor:
        """
        Extract feature maps from YOLO backbone

        Args:
            frame: Input frame (H, W, C)

        Returns:
            features: Feature tensor (C, H', W')
        """
        # Preprocess frame
        img = cv2.resize(frame, (640, 640))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        # Extract features from backbone
        with torch.no_grad():
            # Access the model's backbone
            # Different versions might have different architectures
            try:
                features = self.model.model.model[:10](img)  # First 10 layers
            except:
                # Fallback: use full forward pass
                _ = self.model.model(img)
                features = img

        return features.squeeze(0)

    def classify_players_by_team(
        self,
        frame: np.ndarray,
        player_boxes: np.ndarray
    ) -> np.ndarray:
        """
        Classify players into teams based on jersey color

        Args:
            frame: Input frame (H, W, C)
            player_boxes: Player bounding boxes (N, 4)

        Returns:
            team_labels: Team assignment for each player (N,)
                        0 = team 1, 1 = team 2, 2 = referee/other
        """
        if len(player_boxes) == 0:
            return np.array([])

        # Extract jersey regions and compute dominant colors
        colors = []
        for box in player_boxes:
            x1, y1, x2, y2 = box.astype(int)

            # Crop player region
            player_crop = frame[y1:y2, x1:x2]

            # Focus on upper body (jersey region)
            h = y2 - y1
            jersey_crop = player_crop[:int(h * 0.6), :]

            if jersey_crop.size == 0:
                colors.append([0, 0, 0])
                continue

            # Get dominant color using k-means
            pixels = jersey_crop.reshape(-1, 3)
            dominant_color = np.median(pixels, axis=0)
            colors.append(dominant_color)

        colors = np.array(colors)

        # Cluster into 3 teams using k-means
        from sklearn.cluster import KMeans

        if len(colors) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            team_labels = kmeans.fit_predict(colors)
        elif len(colors) == 2:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            team_labels = kmeans.fit_predict(colors)
            team_labels = np.append(team_labels, 2)  # Add referee class
        else:
            team_labels = np.zeros(len(colors), dtype=int)

        return team_labels

    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: Dict,
        team_labels: np.ndarray = None
    ) -> np.ndarray:
        """
        Visualize detections on frame

        Args:
            frame: Input frame (H, W, C)
            detections: Detection dictionary
            team_labels: Optional team labels for players

        Returns:
            Annotated frame
        """
        frame_vis = frame.copy()
        boxes = detections['boxes']
        scores = detections['scores']
        classes = detections['classes']

        # Color mapping
        colors = {
            0: [(0, 255, 0), (0, 0, 255), (255, 255, 0)],  # Teams: Green, Blue, Yellow
            32: (255, 0, 0)  # Ball: Red
        }

        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box.astype(int)
            cls = int(cls)

            # Determine color
            if cls == 0 and team_labels is not None and i < len(team_labels):
                color = colors[0][team_labels[i]]
                label = f"Player-T{team_labels[i]}: {score:.2f}"
            elif cls == 0:
                color = (0, 255, 0)
                label = f"Player: {score:.2f}"
            else:
                color = colors.get(cls, (255, 255, 255))
                label = f"Ball: {score:.2f}"

            # Draw box
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(
                frame_vis, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return frame_vis


if __name__ == "__main__":
    # Test the detector
    print("Testing Player and Ball Detector...")

    # Initialize detector
    detector = PlayerBallDetector()

    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Test single frame detection
    print("\nTesting single frame detection...")
    detection = detector.detect_frame(dummy_frame, return_features=True)
    print(f"Detections:")
    print(f"  Boxes: {detection['boxes'].shape}")
    print(f"  Scores: {detection['scores'].shape}")
    print(f"  Classes: {detection['classes'].shape}")
    if 'features' in detection:
        print(f"  Features: {detection['features'].shape}")

    # Test batch detection
    print("\nTesting batch detection...")
    dummy_batch = np.random.randint(0, 255, (4, 720, 1280, 3), dtype=np.uint8)
    detections = detector.detect_batch(dummy_batch)
    print(f"Batch detections: {len(detections)} frames")

    print("\nDetector test completed!")
