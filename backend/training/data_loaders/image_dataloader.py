"""
Image Dataset Loader
Handles loading and preprocessing image data for deepfake detection
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def get_image_transforms(augment=True, img_size=(224, 224)):
    """
    Get image transforms for training/validation
    Args:
        augment: Whether to apply augmentation
        img_size: Target image size
    Returns:
        torchvision.transforms.Compose
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class DeepfakeImageDataset(Dataset):
    """
    Image Dataset for Deepfake Detection
    Expected directory structure:
        root/
        ├── REAL/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        └── FAKE/
            ├── img1.jpg
            ├── img2.jpg
            └── ...
    """
    
    def __init__(self, root_dir, split='train', img_size=(224, 224), augment=True):
        """
        Args:
            root_dir: Path to dataset root
            split: 'train', 'validation', or 'test'
            img_size: Target image size
            augment: Whether to apply augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = get_image_transforms(augment, img_size)
        self.images = []
        self.labels = []
        
        # Class names
        self.class_names = ['REAL', 'FAKE']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load images
        self._load_images()
    
    def _load_images(self):
        """Load image paths and labels"""
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            label = self.class_to_idx[class_name]
            
            for img_file in class_dir.glob('*.[jp][pn]g'):
                self.images.append(str(img_file))
                self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images from {self.root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get sample"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data on error
            return torch.zeros(3, *self.img_size), 0


if __name__ == '__main__':
    # Test usage
    dataset = DeepfakeImageDataset(
        root_dir='../dataset/image/train',
        split='train',
        augment=True
    )
    print(f"Dataset size: {len(dataset)}")
