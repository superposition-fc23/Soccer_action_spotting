"""
Toy dataset loader - loads only first N videos for quick experimentation
"""
from utils.augmentations import VideoAugmentation, ConservativeVideoAugmentation
import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
import toy_config as config


class ToyDataset(Dataset):
    """
    Simplified dataset that loads only a few videos for quick testing
    """

    def __init__(
        self,
        split: str = 'train',
        max_videos: int = 3,
        temporal_window: int = None
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            max_videos: Maximum number of videos to load
            temporal_window: Number of frames per clip
        """
        self.split = split
        self.max_videos = max_videos
        self.temporal_window = temporal_window or config.TEMPORAL_WINDOW_SIZE
        # Data augmentation
        self.augmentation = ConservativeVideoAugmentation(mode=split)
        # OR for more aggressive augmentation:
        # self.augmentation = VideoAugmentation(mode=split)

        # Get data directory
        if split == 'train':
            self.data_dir = config.TRAIN_DIR
        elif split == 'val':
            self.data_dir = config.VALID_DIR
        else:
            self.data_dir = config.TEST_DIR

        # Action label mapping
        self.action_to_idx = {
            "PASS": 0,
            "DRIVE": 1,
            "HIGH PASS": 2,
            "HEADER": 3,
            "BACKGROUND": 4,
            "OUT": 5,
            "THROW IN": 6,
            "CROSS": 7,
            "BALL PLAYER BLOCK": 8,
            "SHOT": 9,
            "PLAYER SUCCESSFUL TACKLE": 10,
            "FREE KICK": 11,
            "GOAL": 12

        }

        # Load samples
        self.samples = self._load_dataset()

        # Apply class balancing if enabled (only for training)
        if config.USE_CLASS_BALANCING and split == 'train':
            self.samples = self._apply_class_balancing(self.samples)

        num_videos = len(set([s['video_path'] for s in self.samples]))
        print(f"[TOY DATASET] Loaded {len(self.samples)} clips from {num_videos} videos ({split})")

    def _load_dataset(self) -> List[Dict]:
        """Load video samples from dataset directory"""
        # Check if dataset directory exists
        if not self.data_dir.exists():
            print(f"[WARNING] Dataset directory not found: {self.data_dir}")
            print(f"[INFO] Creating synthetic dummy data for testing...")
            return self._create_synthetic_samples()

        # Choose sampling strategy based on config
        if config.USE_DENSE_SAMPLING:
            print(f"[DENSE SAMPLING] Using dense temporal sampling across entire video")
            return self._load_dataset_dense()
        else:
            print(f"[ACTION-CENTRIC] Using action-centric sampling (current method)")
            return self._load_dataset_action_centric()

    def _load_dataset_action_centric(self) -> List[Dict]:
        """Original action-centric sampling - one sample per labeled action"""
        samples = []
        video_count = 0

        # Iterate through season/match directories
        for season_dir in sorted(self.data_dir.glob("*")):
            if not season_dir.is_dir():
                continue

            for match_dir in sorted(season_dir.glob("*")):
                if not match_dir.is_dir():
                    continue

                if video_count >= self.max_videos:
                    break

                # Look for video and label files
                video_path = match_dir / f"{config.VIDEO_RESOLUTION}.mp4"
                label_path = match_dir / "Labels-ball.json"

                if not video_path.exists() or not label_path.exists():
                    print(f"[WARNING] Missing files in {match_dir}")
                    continue

                # Load annotations
                try:
                    with open(label_path, 'r') as f:
                        annotations = json.load(f)

                    actions = self._parse_annotations(annotations.get('annotations', []))

                    # Create one sample per action (clip-based loading)
                    for action in actions:
                        samples.append({
                            'video_path': str(video_path),
                            'match_name': match_dir.name,
                            'action': action,
                        })

                    video_count += 1

                except Exception as e:
                    print(f"[WARNING] Failed to load {match_dir}: {str(e)}")
                    continue

            if video_count >= self.max_videos:
                break

        if len(samples) == 0:
            print(f"[WARNING] No valid samples found. Creating synthetic data...")
            return self._create_synthetic_samples()

        return samples

    def _load_dataset_dense(self) -> List[Dict]:
        """Dense sampling - slide window across entire video with stride"""
        import time
        start_time = time.time()

        samples = []
        video_count = 0

        print(f"[DENSE SAMPLING] Starting to scan videos from {self.data_dir}...")

        # Iterate through season/match directories
        for season_dir in sorted(self.data_dir.glob("*")):
            if not season_dir.is_dir():
                continue

            for match_dir in sorted(season_dir.glob("*")):
                if not match_dir.is_dir():
                    continue

                if video_count >= self.max_videos:
                    break

                # Look for video and label files
                video_path = match_dir / f"{config.VIDEO_RESOLUTION}.mp4"
                label_path = match_dir / "Labels-ball.json"

                if not video_path.exists() or not label_path.exists():
                    print(f"[WARNING] Missing files in {match_dir}")
                    continue

                # Load annotations
                try:
                    print(f"[DENSE SAMPLING] Processing video {video_count + 1}/{self.max_videos}: {match_dir.name}...")

                    with open(label_path, 'r') as f:
                        annotations = json.load(f)

                    actions = self._parse_annotations(annotations.get('annotations', []))
                    print(f"  - Found {len(actions)} actions in annotations")

                    # SKIP video opening - just use action-centric with extra background samples
                    # This avoids slow network I/O during initialization
                    print(f"  - Using action frames to estimate background samples (no video opening needed)...")

                    # Generate dense samples WITHOUT video_length (faster initialization)
                    print(f"  - Generating dense samples (stride={config.DENSE_SAMPLING_STRIDE}, ratio={config.BACKGROUND_TO_ACTION_RATIO})...")
                    video_samples = self._generate_dense_samples_fast(
                        str(video_path),
                        match_dir.name,
                        actions
                    )

                    samples.extend(video_samples)
                    video_count += 1

                    elapsed = time.time() - start_time
                    print(f"  - Total samples so far: {len(samples)} (took {elapsed:.1f}s)")

                except Exception as e:
                    print(f"[WARNING] Failed to load {match_dir}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            if video_count >= self.max_videos:
                break

        total_time = time.time() - start_time
        print(f"[DENSE SAMPLING] Finished! Generated {len(samples)} total samples in {total_time:.1f}s")

        if len(samples) == 0:
            print(f"[WARNING] No valid samples found. Creating synthetic data...")
            return self._create_synthetic_samples()

        return samples

    def _parse_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Parse annotation list"""
        actions = []

        for ann in annotations:
            # Normalize label strings: uppercase, replace underscores, trim
            raw_label = str(ann.get('label', 'PASS'))
            label_norm = raw_label.strip().upper().replace('_', ' ')

            action = {
                'label': label_norm,
                'label_idx': self.action_to_idx.get(label_norm, 0),
                'frame': int(ann.get('position', 0)),
                'gameTime': ann.get('gameTime', '0:00'),
                'team': ann.get('team', 'home'),
                'visibility': ann.get('visibility', 'visible')
            }
            actions.append(action)

        return actions

    def _apply_class_balancing(self, samples: List[Dict]) -> List[Dict]:
        """
        Apply class balancing by oversampling minority classes

        Uses config.CLASS_BALANCE_RATIOS to determine how much to oversample each class
        """
        import random
        from collections import Counter

        print(f"\n[CLASS BALANCING] Applying class balancing to training data...")

        # Group samples by their MAPPED class (0: PASS, 1: DRIVE, 2: BACKGROUND)
        class_samples = {0: [], 1: [], 2: []}

        for sample in samples:
            original_label_idx = sample['action']['label_idx']
            # Map from 5-class to 3-class system
            mapped_label = config.LABEL_MAPPING_5_TO_3[original_label_idx]
            class_samples[mapped_label].append(sample)

        # Print original distribution
        original_counts = {k: len(v) for k, v in class_samples.items()}
        total_original = sum(original_counts.values())
        print(f"[CLASS BALANCING] Original distribution:")
        for class_idx in range(3):
            class_name = config.ACTION_CLASSES[class_idx]
            count = original_counts[class_idx]
            pct = (count / total_original * 100) if total_original > 0 else 0
            print(f"  {class_name:12s}: {count:4d} ({pct:5.1f}%)")

        # Apply oversampling
        balanced_samples = []

        for class_idx in range(3):
            class_name = config.ACTION_CLASSES[class_idx]
            ratio = config.CLASS_BALANCE_RATIOS.get(class_idx, 1.0)

            original_class_samples = class_samples[class_idx]
            num_original = len(original_class_samples)
            num_target = int(num_original * ratio)

            # Add original samples
            balanced_samples.extend(original_class_samples)

            # Add duplicates to reach target
            num_to_add = num_target - num_original
            if num_to_add > 0:
                # Randomly sample with replacement
                duplicates = random.choices(original_class_samples, k=num_to_add)
                balanced_samples.extend(duplicates)
                print(f"  {class_name:12s}: Added {num_to_add} duplicate samples (ratio: {ratio}x)")

        # Shuffle to mix classes
        random.shuffle(balanced_samples)

        # Print new distribution
        new_class_counts = Counter()
        for sample in balanced_samples:
            original_label_idx = sample['action']['label_idx']
            mapped_label = config.LABEL_MAPPING_5_TO_3[original_label_idx]
            new_class_counts[mapped_label] += 1

        total_new = len(balanced_samples)
        print(f"\n[CLASS BALANCING] New distribution after balancing:")
        for class_idx in range(3):
            class_name = config.ACTION_CLASSES[class_idx]
            count = new_class_counts[class_idx]
            pct = (count / total_new * 100) if total_new > 0 else 0
            print(f"  {class_name:12s}: {count:4d} ({pct:5.1f}%)")

        print(f"[CLASS BALANCING] Total samples: {total_original} → {total_new} (+{total_new - total_original})\n")

        return balanced_samples

    def _generate_dense_samples_fast(
        self,
        video_path: str,
        match_name: str,
        actions: List[Dict]
    ) -> List[Dict]:
        """
        Fast dense sampling - generates background samples between actions
        WITHOUT opening video file (avoids slow network I/O)
        """
        import random
        samples = []

        # Collect action samples
        action_samples = []
        for action in actions:
            action_samples.append({
                'video_path': video_path,
                'match_name': match_name,
                'action': action,
            })

        # Generate background samples BETWEEN consecutive actions
        # Randomize ratio per video to prevent model from learning the ratio itself
        ratio_variance = 0.3  # ±30% variance
        min_ratio = max(0.2, config.BACKGROUND_TO_ACTION_RATIO - ratio_variance)
        max_ratio = config.BACKGROUND_TO_ACTION_RATIO + ratio_variance
        random_ratio = random.uniform(min_ratio, max_ratio)
        num_background_needed = int(len(action_samples) * random_ratio)
        background_samples = []

        print(f"    Using random background ratio: {random_ratio:.2f} (base: {config.BACKGROUND_TO_ACTION_RATIO})")

        # Sort actions by frame number
        sorted_actions = sorted(actions, key=lambda x: x['frame'])

        # Generate background samples in gaps between actions
        for i in range(len(sorted_actions) - 1):
            current_action_frame = sorted_actions[i]['frame']
            next_action_frame = sorted_actions[i + 1]['frame']

            # Gap between actions must be large enough for a window + buffer
            min_gap = self.temporal_window * 2  # Need space for non-overlapping window
            gap_size = next_action_frame - current_action_frame

            if gap_size > min_gap:
                # Place background sample in middle of gap
                background_frame = (current_action_frame + next_action_frame) // 2

                background_action = {
                    'label': 'BACKGROUND',
                    'label_idx': 4,
                    'frame': background_frame,
                    'gameTime': f'{int(background_frame / config.FPS / 60)}:{int((background_frame / config.FPS) % 60):02d}',
                    'team': 'none',
                    'visibility': 'visible'
                }

                background_samples.append({
                    'video_path': video_path,
                    'match_name': match_name,
                    'action': background_action,
                })

        # If we need more background samples, add them before first action and after last action
        if len(background_samples) < num_background_needed and len(sorted_actions) > 0:
            # Add before first action (if there's space)
            first_frame = sorted_actions[0]['frame']
            if first_frame > self.temporal_window:
                bg_frame = first_frame - self.temporal_window
                background_samples.append({
                    'video_path': video_path,
                    'match_name': match_name,
                    'action': {
                        'label': 'BACKGROUND',
                        'label_idx': 4,
                        'frame': bg_frame,
                        'gameTime': f'{int(bg_frame / config.FPS / 60)}:{int((bg_frame / config.FPS) % 60):02d}',
                        'team': 'none',
                        'visibility': 'visible'
                    }
                })

            # Add after last action (assume ~5 minutes of video = 4500 frames)
            last_frame = sorted_actions[-1]['frame']
            estimated_video_end = last_frame + 3000  # Conservative estimate
            if len(background_samples) < num_background_needed:
                bg_frame = last_frame + self.temporal_window
                if bg_frame < estimated_video_end:
                    background_samples.append({
                        'video_path': video_path,
                        'match_name': match_name,
                        'action': {
                            'label': 'BACKGROUND',
                            'label_idx': 4,
                            'frame': bg_frame,
                            'gameTime': f'{int(bg_frame / config.FPS / 60)}:{int((bg_frame / config.FPS) % 60):02d}',
                            'team': 'none',
                            'visibility': 'visible'
                        }
                    })

        # Limit to requested ratio
        if len(background_samples) > num_background_needed:
            background_samples = random.sample(background_samples, num_background_needed)

        # Combine and shuffle
        samples = action_samples + background_samples
        random.shuffle(samples)

        print(f"  [{match_name}] Generated {len(action_samples)} action + {len(background_samples)} background = {len(samples)} total samples")

        return samples

# For future work : Intuition says this is conceptually better but could not implement due to compute and time constraints. 
# The idea is to provide more representations of non-actions in game which helps the model generalize better.
    def _generate_dense_samples(
        self,
        video_path: str,
        match_name: str,
        video_length: int,
        actions: List[Dict]
    ) -> List[Dict]:
        """
        Generate dense samples by sliding window across entire video.
        Intelligently balances action and background samples.
        """
        samples = []

        # First, collect all action-centered windows
        action_samples = []
        for action in actions:
            action_samples.append({
                'video_path': video_path,
                'match_name': match_name,
                'action': action,
            })

        # Calculate how many background samples to add
        num_action_samples = len(action_samples)
        num_background_samples = int(num_action_samples * config.BACKGROUND_TO_ACTION_RATIO)

        # Generate candidate background windows by sliding across video
        background_candidates = []
        for frame_idx in range(0, video_length - self.temporal_window, config.DENSE_SAMPLING_STRIDE):
            window_center = frame_idx + self.temporal_window // 2

            # Check if this window overlaps with any action
            is_background = True
            for action in actions:
                action_frame = action['frame']
                # If action frame is within this window, skip it (not background)
                if abs(window_center - action_frame) < self.temporal_window // 2:
                    is_background = False
                    break

            if is_background:
                # Create a BACKGROUND action entry
                background_action = {
                    'label': 'BACKGROUND',
                    'label_idx': 4,  # Original index before mapping
                    'frame': window_center,
                    'gameTime': f'{int(window_center / config.FPS / 60)}:{int((window_center / config.FPS) % 60):02d}',
                    'team': 'none',
                    'visibility': 'visible'
                }

                background_candidates.append({
                    'video_path': video_path,
                    'match_name': match_name,
                    'action': background_action,
                })

        # Randomly sample background windows to match desired ratio
        import random
        if len(background_candidates) > num_background_samples:
            background_samples = random.sample(background_candidates, num_background_samples)
        else:
            background_samples = background_candidates

        # Combine action and background samples
        samples = action_samples + background_samples

        # Shuffle to mix action and background
        random.shuffle(samples)

        print(f"  [{match_name}] Generated {len(action_samples)} action + {len(background_samples)} background = {len(samples)} total samples")

        return samples

# not required anymore
    def _create_synthetic_samples(self) -> List[Dict]:
        """Create synthetic dummy samples for testing when no real data available"""
        print(f"[TOY DATASET] Creating {self.max_videos * 10} synthetic samples")

        samples = []
        for video_idx in range(self.max_videos):
            for action_idx in range(10):  # 10 actions per video
                # Cycle through action types
                label_idx = action_idx % 4  # 0-3 (exclude BACKGROUND for now)
                label = list(self.action_to_idx.keys())[label_idx]

                samples.append({
                    'video_path': f'synthetic_video_{video_idx}.mp4',
                    'match_name': f'synthetic_match_{video_idx}',
                    'action': {
                        'label': label,
                        'label_idx': label_idx,
                        'frame': action_idx * 100,  # Spread actions out
                        'gameTime': f'{action_idx}:00',
                        'team': 'home' if action_idx % 2 == 0 else 'away',
                        'visibility': 'visible'
                    },
                    'synthetic': True
                })

        return samples

# Load video clip centered around action frame.
    def _load_video_clip(
        self,
        video_path: str,
        center_frame: int,
        window_size: int
    ) -> np.ndarray:
        """Load a clip of frames centered around the action"""

        # Check if synthetic
        if video_path.startswith('synthetic_'):
            # Return random frames
            return np.random.randint(0, 255, (window_size, 224, 398, 3), dtype=np.uint8)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            # Return blank frames
            return np.zeros((window_size, 224, 398, 3), dtype=np.uint8)

        # Calculate start and end frames
        half_window = window_size // 2
        start_frame = max(0, center_frame - half_window)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(window_size):
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame or blank
                if len(frames) > 0:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((224, 398, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap.release()
        return np.array(frames)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset

        Returns:
            Dictionary with:
                - video: Tensor of shape (T, C, H, W)
                - label: Action label (single integer)
                - metadata: Additional info
        """
        sample = self.samples[idx]
        action = sample['action']

        # Load video clip
        frames = self._load_video_clip(
            sample['video_path'],
            action['frame'],
            self.temporal_window
        )
        
        # Apply augmentation
        frames = self.augmentation(frames)

        # Convert to tensor (T, H, W, C) -> (T, C, H, W)
        # Augmentation returns list of numpy arrays, stack them
        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)  # List -> (T, H, W, C)

        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

        # Get label and map from 5 classes to 3 classes
        original_label = action['label_idx']
        mapped_label = config.LABEL_MAPPING_5_TO_3[original_label]
        label = torch.tensor(mapped_label, dtype=torch.long)

        return {
            'video': frames,
            'label': label,
            'metadata': {
                'match_name': sample['match_name'],
                'action_type': action['label'],
                'frame': action['frame'],
                'gameTime': action.get('gameTime', '0:00'),
                'synthetic': sample.get('synthetic', False)
            }
        }


def custom_collate_fn(batch):
    """Custom collate function to properly handle metadata dicts"""
    videos = torch.stack([item['video'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    metadata = [item['metadata'] for item in batch]  # Keep as list of dicts

    return {
        'video': videos,
        'label': labels,
        'metadata': metadata
    }


def get_toy_dataloader(split: str = 'train', max_videos: int = 3) -> DataLoader:
    """
    Get a DataLoader for toy experiments

    Args:
        split: 'train', 'val', or 'test'
        max_videos: Maximum number of videos to load

    Returns:
        DataLoader
    """
    dataset = ToyDataset(split, max_videos)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == 'train'),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )


if __name__ == "__main__":
    """Test the toy dataset loader"""
    print("Testing Toy Dataset Loader...")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Temporal window: {config.TEMPORAL_WINDOW_SIZE}")
    print()

    # Test train loader
    print("Creating train dataloader...")
    train_loader = get_toy_dataloader(split='train', max_videos=2)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print()

    # Test a batch
    print("Loading first batch...")
    for batch in train_loader:
        print(f"Video shape: {batch['video'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Label values: {batch['label']}")
        print(f"Metadata: {batch['metadata']}")
        break

    print("\nToy dataset loader test completed!")
