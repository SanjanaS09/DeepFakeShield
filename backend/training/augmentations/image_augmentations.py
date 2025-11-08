"""
Image Augmentation Pipeline
Advanced augmentation techniques for image deepfake detection
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random


class ImageAugmentationPipeline:
    """
    Image augmentation pipeline with multiple advanced techniques
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
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
    
    def __call__(self, image, is_training=True):
        """Apply augmentation"""
        if is_training:
            return self.transforms_train(image)
        else:
            return self.transforms_val(image)
    
    def apply_random_erasing(self, image, p=0.5):
        """Apply random erasing augmentation"""
        if random.random() < p:
            return F.to_tensor(transforms.RandomErasing(p=1.0)(image))
        return F.to_tensor(image)
    
    def apply_mixup(self, image1, image2, alpha=1.0):
        """Apply mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        return lam * image1 + (1 - lam) * image2


if __name__ == '__main__':
    pipeline = ImageAugmentationPipeline()
    print("ImageAugmentationPipeline initialized successfully")
