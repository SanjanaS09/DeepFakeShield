"""
Multimodal Augmentation Pipeline
Coordinated augmentation across image, video, and audio modalities
"""

import torch
import random
from augmentations.image_augmentations import ImageAugmentationPipeline
from augmentations.video_augmentations import VideoAugmentationPipeline
from augmentations.audio_augmentations import AudioAugmentationPipeline


class MultimodalAugmentationPipeline:
    """
    Multimodal augmentation pipeline that applies coordinated augmentations
    across all modalities
    """
    
    def __init__(self, image_size=(224, 224), sample_rate=16000, 
                 augmentation_config=None):
        """
        Args:
            image_size: Target image size
            sample_rate: Audio sample rate
            augmentation_config: Dict with augmentation parameters
        """
        self.image_size = image_size
        self.sample_rate = sample_rate
        self.config = augmentation_config or {}
        
        # Initialize individual pipelines
        self.image_aug = ImageAugmentationPipeline(image_size, augmentation_config)
        self.video_aug = VideoAugmentationPipeline(image_size, augmentation_config)
        self.audio_aug = AudioAugmentationPipeline(sample_rate, augmentation_config)
    
    def __call__(self, multimodal_sample, is_training=True):
        """
        Apply coordinated augmentations to multimodal sample
        Args:
            multimodal_sample: Dict with 'image', 'video', 'audio' keys
            is_training: Whether to apply training augmentations
        Returns:
            Augmented multimodal sample dict
        """
        augmented = {}
        
        # Apply augmentations (they're coordinated through random seed if needed)
        if 'image' in multimodal_sample:
            augmented['image'] = self.image_aug(
                multimodal_sample['image'],
                is_training=is_training
            )
        
        if 'video' in multimodal_sample:
            augmented['video'] = self.video_aug(
                multimodal_sample['video'],
                is_training=is_training
            )
        
        if 'audio' in multimodal_sample:
            augmented['audio'] = self.audio_aug(
                multimodal_sample['audio'],
                is_training=is_training
            )
        
        # Keep label unchanged
        if 'label' in multimodal_sample:
            augmented['label'] = multimodal_sample['label']
        
        return augmented
    
    def apply_modality_dropout(self, multimodal_sample, dropout_prob=0.1):
        """
        Randomly drop a modality to train robust fusion models
        Args:
            multimodal_sample: Multimodal dict
            dropout_prob: Probability of dropping each modality
        """
        if random.random() < dropout_prob and 'image' in multimodal_sample:
            multimodal_sample['image'] = torch.zeros_like(multimodal_sample['image'])
        
        if random.random() < dropout_prob and 'video' in multimodal_sample:
            multimodal_sample['video'] = torch.zeros_like(multimodal_sample['video'])
        
        if random.random() < dropout_prob and 'audio' in multimodal_sample:
            multimodal_sample['audio'] = torch.zeros_like(multimodal_sample['audio'])
        
        return multimodal_sample
    
    def apply_mixup(self, sample1, sample2, alpha=1.0):
        """
        Apply mixup augmentation across modalities
        Args:
            sample1: First multimodal sample
            sample2: Second multimodal sample
            alpha: Beta distribution parameter
        """
        lam = random.betavariate(alpha, alpha)
        
        mixed = {}
        
        if 'image' in sample1 and 'image' in sample2:
            mixed['image'] = lam * sample1['image'] + (1 - lam) * sample2['image']
        
        if 'video' in sample1 and 'video' in sample2:
            mixed['video'] = lam * sample1['video'] + (1 - lam) * sample2['video']
        
        if 'audio' in sample1 and 'audio' in sample2:
            mixed['audio'] = lam * sample1['audio'] + (1 - lam) * sample2['audio']
        
        # Mix labels (soft labels)
        if 'label' in sample1 and 'label' in sample2:
            mixed['label'] = lam * sample1['label'] + (1 - lam) * sample2['label']
        
        return mixed, lam


if __name__ == '__main__':
    pipeline = MultimodalAugmentationPipeline()
    print("MultimodalAugmentationPipeline initialized successfully")
