
"""
Video data augmentation for action recognition
Maintains temporal consistency and spatial relationships
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np


class VideoAugmentation:
    """
    Augmentation for video clips that maintains temporal consistency.
    All frames in a clip get the SAME transformation.
    """
    
    def __init__(self, mode='train'):
        """
        Args:
            mode: 'train' or 'val' - augmentation only applied during training
        """
        self.mode = mode
        
        # Augmentation probabilities
        self.p_hflip = 0.5          # 50% horizontal flip
        self.p_rotation = 0.3       # 30% rotation
        self.p_translate = 0.3      # 30% translation
        self.p_color = 0.4          # 40% color jitter
        
        # Augmentation parameters
        self.rotation_range = (-10, 10)      # degrees
        self.translate_range = (0.05, 0.05)  # 5% of image size (h, w)
        
    def __call__(self, frames):
        """
        Apply augmentation to video clip.
        
        Args:
            frames: List of PIL Images or numpy arrays (T, H, W, C)
        
        Returns:
            Augmented frames in same format
        """
        if self.mode != 'train':
            return frames
        
        # Sample augmentation parameters ONCE for all frames
        apply_hflip = random.random() < self.p_hflip
        apply_rotation = random.random() < self.p_rotation
        apply_translate = random.random() < self.p_translate
        apply_color = random.random() < self.p_color
        
        # Sample parameters
        if apply_rotation:
            angle = random.uniform(*self.rotation_range)
        
        if apply_translate:
            h_translate = random.uniform(-self.translate_range[0], self.translate_range[0])
            w_translate = random.uniform(-self.translate_range[1], self.translate_range[1])
        
        if apply_color:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
        
        # Apply same transforms to ALL frames
        augmented_frames = []
        for frame in frames:
            # Convert to PIL if needed
            if isinstance(frame, np.ndarray):
                frame = TF.to_pil_image(frame)
            
            # Apply transformations
            if apply_hflip:
                frame = TF.hflip(frame)
            
            if apply_rotation:
                frame = TF.rotate(frame, angle, fill=0)
            
            if apply_translate:
                # Get image size
                w, h = frame.size
                frame = TF.affine(
                    frame,
                    angle=0,
                    translate=(int(w * w_translate), int(h * h_translate)),
                    scale=1.0,
                    shear=0,
                    fill=0
                )
            
            if apply_color:
                frame = TF.adjust_brightness(frame, brightness)
                frame = TF.adjust_contrast(frame, contrast)
                frame = TF.adjust_saturation(frame, saturation)
            
            # Convert back to numpy
            frame = np.array(frame)
            augmented_frames.append(frame)
        
        return augmented_frames


class ConservativeVideoAugmentation:
    """
    More conservative augmentation that preserves ball-player distances better.
    Recommended for action recognition with spatial relationships.
    """
    
    def __init__(self, mode='train'):
        self.mode = mode
        self.p_hflip = 0.5
        self.p_color = 0.5
        
    def __call__(self, frames):
        if self.mode != 'train':
            return frames
        
        apply_hflip = random.random() < self.p_hflip
        apply_color = random.random() < self.p_color
        
        if apply_color:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
        
        augmented_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame = TF.to_pil_image(frame)
            
            if apply_hflip:
                frame = TF.hflip(frame)
            
            if apply_color:
                frame = TF.adjust_brightness(frame, brightness)
                frame = TF.adjust_contrast(frame, contrast)
            
            frame = np.array(frame)
            augmented_frames.append(frame)
        
        return augmented_frames