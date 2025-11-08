"""
Multimodal Dataset Loader
Combines image, video, and audio data for fusion-based deepfake detection
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from data_loaders.image_dataloader import DeepfakeImageDataset, get_image_transforms
from data_loaders.video_dataloader import DeepfakeVideoDataset, get_video_transforms
from data_loaders.audio_dataloader import DeepfakeAudioDataset, get_audio_transforms


class MultimodalDataset(Dataset):
    """
    Multimodal Dataset combining Image, Video, and Audio
    """
    
    def __init__(self, image_root, video_root, audio_root, split='train',
                 image_size=(224, 224), frames_per_clip=16, sample_rate=16000,
                 augment=True):
        """
        Args:
            image_root: Path to image dataset root
            video_root: Path to video dataset root
            audio_root: Path to audio dataset root
            split: 'train', 'validation', or 'test'
            image_size: Target image size
            frames_per_clip: Number of frames per video clip
            sample_rate: Audio sample rate
            augment: Whether to apply augmentation
        """
        self.split = split
        
        # Initialize individual datasets
        self.image_dataset = DeepfakeImageDataset(
            root_dir=image_root,
            split=split,
            img_size=image_size,
            augment=augment
        )
        
        self.video_dataset = DeepfakeVideoDataset(
            root_dir=video_root,
            split=split,
            frames_per_clip=frames_per_clip,
            img_size=image_size,
            augment=augment
        )
        
        self.audio_dataset = DeepfakeAudioDataset(
            root_dir=audio_root,
            split=split,
            sample_rate=sample_rate,
            augment=augment
        )
        
        # Ensure all datasets have same length (use minimum)
        self.length = min(
            len(self.image_dataset),
            len(self.video_dataset),
            len(self.audio_dataset)
        )
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Get multimodal sample
        Returns:
            dict with 'image', 'video', 'audio', and 'label'
        """
        try:
            # Get samples from each modality
            image, image_label = self.image_dataset[idx % len(self.image_dataset)]
            video, video_label = self.video_dataset[idx % len(self.video_dataset)]
            audio, audio_label = self.audio_dataset[idx % len(self.audio_dataset)]
            
            # All labels should be the same (they come from same class folder)
            label = image_label
            
            return {
                'image': image,
                'video': video,
                'audio': audio,
                'label': label
            }
        
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return {
                'image': torch.zeros(3, 224, 224),
                'video': torch.zeros(16, 3, 224, 224),
                'audio': torch.zeros(128, 500),
                'label': 0
            }


if __name__ == '__main__':
    dataset = MultimodalDataset(
        image_root='../dataset/image/train',
        video_root='../dataset/video/train',
        audio_root='../dataset/audio/train',
        split='train',
        augment=True
    )
    print(f"Multimodal dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Label: {sample['label']}")
