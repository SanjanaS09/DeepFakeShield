"""
Video Augmentation Pipeline
Augmentation techniques for video deepfake detection
"""

import torch
import numpy as np
import torchvision.transforms as transforms
import random


class VideoAugmentationPipeline:
    """
    Video augmentation pipeline
    Applies consistent augmentation across all frames
    """
    
    def __init__(self, image_size=(224, 224), augmentation_config=None):
        """
        Args:
            image_size: Target image size
            augmentation_config: Dict with augmentation parameters
        """
        self.image_size = image_size
        self.config = augmentation_config or {}
        
        self.transforms_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.transforms_val = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, video_frames, is_training=True):
        """
        Apply augmentation to video frames
        Args:
            video_frames: List or tensor of frames
            is_training: Whether to apply training augmentations
        """
        transforms_to_use = self.transforms_train if is_training else self.transforms_val
        
        if isinstance(video_frames, list):
            augmented_frames = [transforms_to_use(frame) for frame in video_frames]
            return torch.stack(augmented_frames)
        else:
            # Assume tensor of shape [T, C, H, W] or [T, H, W, C]
            if video_frames.dim() == 4:
                augmented_frames = [transforms_to_use(frame) for frame in video_frames]
                return torch.stack(augmented_frames)
            else:
                return transforms_to_use(video_frames)
    
    def apply_temporal_dropout(self, video_frames, dropout_rate=0.1):
        """Randomly drop frames"""
        if random.random() < dropout_rate:
            idx = random.randint(0, len(video_frames) - 1)
            video_frames[idx] = torch.zeros_like(video_frames[idx])
        return video_frames
    
    def apply_frame_shuffle(self, video_frames, shuffle_rate=0.0):
        """Shuffle frame order (with low probability)"""
        if random.random() < shuffle_rate:
            indices = list(range(len(video_frames)))
            random.shuffle(indices)
            video_frames = [video_frames[i] for i in indices]
        return video_frames


if __name__ == '__main__':
    pipeline = VideoAugmentationPipeline()
    print("VideoAugmentationPipeline initialized successfully")
