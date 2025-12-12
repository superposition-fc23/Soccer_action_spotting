"""
Headless inference - saves video output instead of displaying
Works on Linux GPU servers without display
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import sys
import argparse

sys.path.append(str(Path(__file__).parent))
import toy_config as config
from models.toy_action_classifier import ToyActionClassifier
from models.detector import PlayerBallDetector
from models.tracker import ByteTracker


class HeadlessInference:
    def __init__(self, model_checkpoint: str, device: str = None, confidence_threshold: float = 0.5):
        self.device = device or config.DEVICE
        
        # Action classes
        self.action_classes = config.ACTION_CLASSES
        
        self.confidence_threshold = confidence_threshold

        # Load model
        print(f"[LOADING] Model from {model_checkpoint}")
        self.model = ToyActionClassifier(use_frame_filtering=False).to(self.device)
        checkpoint = torch.load(model_checkpoint, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()

        # Initialize detector and tracker
        self.detector = PlayerBallDetector(device=self.device)
        self.tracker = ByteTracker()

        # Class counting
        self.class_counts = Counter()
        
    def run_inference(self, video_path: str, output_path: str, window_size: int = 32, stride: int = 8):
        """Run inference and save output video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"[VIDEO] {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"[INFERENCE] Window size: {window_size}, Stride: {stride}")
        
        frame_buffer = []
        frame_idx = 0
        current_prediction = None
        current_class_name = "Buffering..."

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(frame.copy())

            # Process when we have enough frames
            if len(frame_buffer) >= window_size:
                # Classify window
                prediction = self._classify_window(frame_buffer[-window_size:])
                current_prediction = prediction
                current_class_name = self.action_classes[prediction]
                self.class_counts[prediction] += 1

                # Slide window by stride
                frame_buffer = frame_buffer[stride:]

            # Annotate and write EVERY frame
            annotated_frame = self._draw_prediction(frame.copy(), current_class_name, current_prediction)
            out.write(annotated_frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"[PROGRESS] {frame_idx}/{total_frames} frames ({progress:.1f}%) - Current: {current_class_name}")
        
        cap.release()
        out.release()
        
        print(f"\n[COMPLETE] Output saved to: {output_path}")
        print(f"\n[RESULTS] Class counts:")
        for cls_idx, count in self.class_counts.items():
            print(f"  {self.action_classes[cls_idx]}: {count}")
    
    def _classify_window(self, frames):
        """Classify temporal window"""
        with torch.no_grad():
            sequence_detections = []
            sequence_tracks = []
            frame_features = []

            for frame in frames:
                # Detect players and ball
                detection_result = self.detector.detect_frame(frame, return_features=True)

                # Track objects
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
                    import torch.nn.functional as F
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

            # Stack features: (T, D) -> (1, T, D)
            spatial_features = torch.stack(frame_features).unsqueeze(0)

            # Wrap sequences in batch dimension
            detections_list = [sequence_detections]
            tracks_list = [sequence_tracks]

            # Forward pass
            outputs, _ = self.model(spatial_features, detections_list, tracks_list, track_statistics=True)

            # Get probabilities and prediction
            probs = torch.softmax(outputs, dim=1)
            max_prob, predicted = probs.max(dim=1)

            # Apply confidence threshold
            if max_prob.item() < self.confidence_threshold:
                return 2  # Return BACKGROUND if below threshold

            return predicted.item()
    
    def _draw_prediction(self, frame, predicted_class, class_idx):
        """Draw prediction and counts on frame"""
        # Use BLACK for all text
        color = (0, 0, 0)  # BGR black

        # Draw current prediction
        cv2.putText(frame, f"Action: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw counts for ALL classes (even if zero)
        y_offset = 70
        for i in range(len(self.action_classes)):
            class_name = self.action_classes[i]
            count = self.class_counts.get(i, 0)  # Default to 0 if not in dict
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35

        return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Headless action recognition inference')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='output_inference.mp4', help='Output video path')
    parser.add_argument('--window-size', type=int, default=32, help='Temporal window size')
    parser.add_argument('--stride', type=int, default=8, help='Window stride')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold for predictions (0.0-1.0)')
    args = parser.parse_args()
    
    # Pass confidence threshold to HeadlessInference
    inference = HeadlessInference(args.model, confidence_threshold=args.confidence_threshold)
    inference.run_inference(args.video, args.output, args.window_size, args.stride)