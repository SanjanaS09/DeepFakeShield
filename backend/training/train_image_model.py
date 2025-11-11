#!/usr/bin/env python3
"""
FIXED IMAGE MODEL TRAINING SCRIPT
Place in: backend/training/train_image_model_fixed.py

This script properly integrates with your project structure:
- imports from models/
- imports from preprocessing/
- imports from features/
- imports from utils/
- uses data from dataset/image/
"""

import os
import sys
from pathlib import Path

# FIX: Add backend root to Python path (critical!)
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
from PIL import Image

# ============================================
# NOW THESE IMPORTS WILL WORK
# ============================================
try:
    from models.image_detector import ImageDetector
    from preprocessing.image_preprocessor import ImagePreprocessor
    from features.visual_features import VisualFeatureExtractor
    from utils.logger import get_logger
    from utils.validators import validate_input
    from utils.face_detection import FaceDetector
except ImportError as e:
    print(f"⚠️ Warning: Some imports failed: {e}")
    print("Make sure models/, preprocessing/, features/, and utils/ exist in backend/")

<<<<<<< HEAD
# from data_loaders.image_dataloader import DeepfakeImageDataset, get_image_transforms
#from preprocessing.image_augmentations import ImageAugmentationPipeline

from preprocessing.image_augmentations import ImageAugmentationPipeline


from preprocessing.image_dataloader import DeepfakeImageDataset, get_image_transforms

#from augmentations.image_augmentations import ImageAugmentationPipeline
#from models.image_models import XceptionModel, EfficientNetModel, ResNetModel, VisionTransformerModel

from models.image_detector import XceptionModel, EfficientNetModel, ResNetModel, VisionTransformerModel

from utils.logger import get_logger
from utils.metrics import MetricsCalculator
from utils.checkpoint_manager import CheckpointManager
from utils.visualization import plot_training_curves, visualize_predictions
=======
logger = logging.getLogger(__name__)
>>>>>>> 7f57cfa9997653d4de1ece0912f910ae544a86ef

# ============================================
# SIMPLE DATASET THAT WORKS WITH YOUR STRUCTURE
# ============================================
class DeepfakeImageDataset(Dataset):
    """Dataset for image-based deepfake detection"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        # Load REAL images
        real_dir = self.root_dir / split / 'REAL'
        if real_dir.exists():
            for img_path in sorted(real_dir.glob('*')):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(img_path)
                    self.labels.append(0)  # 0 = REAL
        
        # Load FAKE images
        fake_dir = self.root_dir / split / 'FAKE'
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob('*')):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(img_path)
                    self.labels.append(1)  # 1 = FAKE
        
        logger.info(f"Loaded {len(self.images)} images from {split} split")
        
        if len(self.images) == 0:
            logger.warning(f"⚠️ No images found in {real_dir} or {fake_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return dummy image
            return torch.randn(3, 224, 224), label

# ============================================
# SIMPLE MODEL ARCHITECTURE
# ============================================
class SimpleImageModel(nn.Module):
    """Simple ResNet18-based model for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        import torchvision.models as models
        
        # Use ResNet18 (lightweight)
        self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# ============================================
# TRAINER CLASS
# ============================================
class ImageTrainer:
    """Trainer for image deepfake detection"""
    
    def __init__(self, config_path, dataset_root, checkpoint_dir='checkpoints'):
        """Initialize trainer"""
        self.config_path = config_path
        self.dataset_root = Path(dataset_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load config if exists
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        
        # Create model
        self.model = SimpleImageModel(num_classes=2, pretrained=True)
        self.model = self.model.to(self.device)
        
        # Create data loaders
        self._setup_data_loaders()
        
        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
    
    def _setup_data_loaders(self):
        """Setup training and validation data loaders"""
        # Image transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create datasets
        logger.info(f"Loading datasets from {self.dataset_root}")
        train_dataset = DeepfakeImageDataset(
            self.dataset_root,
            split='train',
            transform=train_transform
        )
        
        val_dataset = DeepfakeImageDataset(
            self.dataset_root,
            split='validation',
            transform=val_transform
        )
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.error(f"ERROR: No datasets found!")
            logger.error(f"Expected: {self.dataset_root}/train/REAL/ and FAKE/")
            logger.error(f"Expected: {self.dataset_root}/validation/REAL/ and FAKE/")
            raise FileNotFoundError(f"Datasets not found in {self.dataset_root}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Windows compatibility
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def train(self, num_epochs=10):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"✓ Saved best model to {checkpoint_path}")
            
            # Step scheduler
            self.scheduler.step()
        
        # Save final model
        final_path = self.checkpoint_dir / 'final_model.pth'
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"\n✓ Saved final model to {final_path}")
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*60)

# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description='Train image deepfake detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python training/train_image_model_fixed.py --dataset dataset/image --epochs 10
    python training/train_image_model_fixed.py --dataset dataset/image --config training/configs/image_config.yaml
        """
    )
    
    parser.add_argument('--dataset', type=str, default='dataset/image',
                       help='Path to image dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("IMAGE DEEPFAKE DETECTION TRAINING")
    logger.info("="*60)
    logger.info(f"Backend root: {BACKEND_ROOT}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create trainer
    trainer = ImageTrainer(
        config_path=args.config,
        dataset_root=args.dataset,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)

if __name__ == '__main__':
    main()
