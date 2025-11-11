"""
Video Dataset Loader
Handles loading and preprocessing video data for deepfake detection
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def get_video_transforms(augment=True, img_size=(224, 224)):
    """
    Get video frame transforms
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
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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


class DeepfakeVideoDataset(Dataset):
    """
    Video Dataset for Deepfake Detection
    Extracts frames from videos for processing
    """
    
    def __init__(self, root_dir, split='train', frames_per_clip=16, 
                 img_size=(224, 224), augment=True, sample_rate=2):
        """
        Args:
            root_dir: Path to dataset root containing REAL/ and FAKE/ folders
            split: 'train', 'validation', or 'test'
            frames_per_clip: Number of frames to extract per video
            img_size: Target image size
            augment: Whether to apply augmentation
            sample_rate: Frame sampling rate (1 = every frame, 2 = every 2nd frame)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.transform = get_video_transforms(augment, img_size)
        
        self.videos = []
        self.labels = []
        self.class_names = ['REAL', 'FAKE']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self._load_videos()
    
    def _load_videos(self):
        """Load video paths and labels"""
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            label = self.class_to_idx[class_name]
            video_extensions = ['.mp4', '.avi', '.mov', '.flv', '.mkv']
            
            for video_file in class_dir.iterdir():
                if video_file.suffix.lower() in video_extensions:
                    self.videos.append(str(video_file))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.videos)} videos from {self.root_dir}")
    
    def _extract_frames(self, video_path):
        """Extract frames from video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < self.frames_per_clip:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Pad with black frames if not enough frames
        while len(frames) < self.frames_per_clip:
            frames.append(Image.new('RGB', self.img_size, color=(0, 0, 0)))
        
        return frames[:self.frames_per_clip]
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        """Get sample"""
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        try:
            frames = self._extract_frames(video_path)
            
            # Transform frames
            if self.transform:
                frames = torch.stack([self.transform(frame) for frame in frames])
            else:
                frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])
            
            return frames, label
        
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return dummy data on error
            return torch.zeros(self.frames_per_clip, 3, *self.img_size), 0


if __name__ == '__main__':
    dataset = DeepfakeVideoDataset(
        root_dir='../dataset/video/train',
        split='train',
        augment=True
    )
    print(f"Dataset size: {len(dataset)}")
