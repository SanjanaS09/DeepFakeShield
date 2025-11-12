# #!/usr/bin/env python3
# """
# FIXED IMAGE MODEL TRAINING SCRIPT
# Place in: backend/training/train_image_model_fixed.py

# This script properly integrates with your project structure:
# - imports from models/
# - imports from preprocessing/
# - imports from features/
# - imports from utils/
# - uses data from dataset/image/
# """

# import os
# import sys
# from pathlib import Path

# # FIX: Add backend root to Python path (critical!)
# BACKEND_ROOT = Path(__file__).parent.parent
# sys.path.insert(0, str(BACKEND_ROOT))

# import argparse
# import logging
# import yaml
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from datetime import datetime
# from tqdm import tqdm
# import json
# from PIL import Image

# # ============================================
# # NOW THESE IMPORTS WILL WORK
# # ============================================
# try:
#     from models.image_detector import ImageDetector
#     from preprocessing.image_preprocessor import ImagePreprocessor
#     from features.visual_features import VisualFeatureExtractor
#     from utils.logger import get_logger
#     from utils.validators import validate_input
#     from utils.face_detection import FaceDetector
# except ImportError as e:
#     print(f"⚠️ Warning: Some imports failed: {e}")
#     print("Make sure models/, preprocessing/, features/, and utils/ exist in backend/")

# logger = logging.getLogger(__name__)

# # ============================================
# # SIMPLE DATASET THAT WORKS WITH YOUR STRUCTURE
# # ============================================
# class DeepfakeImageDataset(Dataset):
#     """Dataset for image-based deepfake detection"""
    
#     def __init__(self, root_dir, split='train', transform=None):
#         self.root_dir = Path(root_dir)
#         self.split = split
#         self.transform = transform
        
#         self.images = []
#         self.labels = []
        
#         # Load REAL images
#         real_dir = self.root_dir / split / 'REAL'
#         if real_dir.exists():
#             for img_path in sorted(real_dir.glob('*')):
#                 if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
#                     self.images.append(img_path)
#                     self.labels.append(0)  # 0 = REAL
        
#         # Load FAKE images
#         fake_dir = self.root_dir / split / 'FAKE'
#         if fake_dir.exists():
#             for img_path in sorted(fake_dir.glob('*')):
#                 if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
#                     self.images.append(img_path)
#                     self.labels.append(1)  # 1 = FAKE
        
#         logger.info(f"Loaded {len(self.images)} images from {split} split")
        
#         if len(self.images) == 0:
#             logger.warning(f"⚠️ No images found in {real_dir} or {fake_dir}")
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         img_path = self.images[idx]
#         label = self.labels[idx]
        
#         try:
#             image = Image.open(img_path).convert('RGB')
            
#             if self.transform:
#                 image = self.transform(image)
            
#             return image, label
#         except Exception as e:
#             logger.warning(f"Error loading {img_path}: {e}")
#             # Return dummy image
#             return torch.randn(3, 224, 224), label

# # ============================================
# # SIMPLE MODEL ARCHITECTURE
# # ============================================
# class SimpleImageModel(nn.Module):
#     """Simple ResNet18-based model for deepfake detection"""
    
#     def __init__(self, num_classes=2, pretrained=True):
#         super().__init__()
#         import torchvision.models as models
        
#         # Use ResNet18 (lightweight)
#         self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        
#         # Replace final layer
#         in_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Linear(in_features, num_classes)
    
#     def forward(self, x):
#         return self.backbone(x)

# # ============================================
# # TRAINER CLASS
# # ============================================
# class ImageTrainer:
#     """Trainer for image deepfake detection"""
    
#     def __init__(self, config_path, dataset_root, checkpoint_dir='checkpoints'):
#         """Initialize trainer"""
#         self.config_path = config_path
#         self.dataset_root = Path(dataset_root)
#         self.checkpoint_dir = Path(checkpoint_dir)
#         self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
#         # Setup device
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         logger.info(f"Using device: {self.device}")
        
#         # Load config if exists
#         self.config = {}
#         if config_path and Path(config_path).exists():
#             with open(config_path, 'r') as f:
#                 self.config = yaml.safe_load(f) or {}
        
#         # Create model
#         self.model = SimpleImageModel(num_classes=2, pretrained=True)
#         self.model = self.model.to(self.device)
        
#         # Create data loaders
#         self._setup_data_loaders()
        
#         # Loss, optimizer, scheduler
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
    
#     def _setup_data_loaders(self):
#         """Setup training and validation data loaders"""
#         # Image transforms
#         train_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(degrees=10),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
        
#         val_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
        
#         # Create datasets
#         logger.info(f"Loading datasets from {self.dataset_root}")
#         train_dataset = DeepfakeImageDataset(
#             self.dataset_root,
#             split='train',
#             transform=train_transform
#         )
        
#         val_dataset = DeepfakeImageDataset(
#             self.dataset_root,
#             split='validation',
#             transform=val_transform
#         )
        
#         if len(train_dataset) == 0 or len(val_dataset) == 0:
#             logger.error(f"ERROR: No datasets found!")
#             logger.error(f"Expected: {self.dataset_root}/train/REAL/ and FAKE/")
#             logger.error(f"Expected: {self.dataset_root}/validation/REAL/ and FAKE/")
#             raise FileNotFoundError(f"Datasets not found in {self.dataset_root}")
        
#         # Create dataloaders
#         self.train_loader = DataLoader(
#             train_dataset,
#             batch_size=8,
#             shuffle=True,
#             num_workers=0  # Windows compatibility
#         )
        
#         self.val_loader = DataLoader(
#             val_dataset,
#             batch_size=8,
#             shuffle=False,
#             num_workers=0
#         )
        
#         logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
#     def train_epoch(self, epoch):
#         """Train for one epoch"""
#         self.model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        
#         for images, labels in pbar:
#             images = images.to(self.device)
#             labels = labels.to(self.device)
            
#             # Forward pass
#             outputs = self.model(images)
#             loss = self.criterion(outputs, labels)
            
#             # Backward pass
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
#             # Statistics
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)
            
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{100.*correct/total:.2f}%'
#             })
        
#         avg_loss = total_loss / len(self.train_loader)
#         avg_acc = 100. * correct / total
        
#         return avg_loss, avg_acc
    
#     def validate(self, epoch):
#         """Validate model"""
#         self.model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
        
#         with torch.no_grad():
#             for images, labels in pbar:
#                 images = images.to(self.device)
#                 labels = labels.to(self.device)
                
#                 outputs = self.model(images)
#                 loss = self.criterion(outputs, labels)
                
#                 total_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 correct += predicted.eq(labels).sum().item()
#                 total += labels.size(0)
                
#                 pbar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{100.*correct/total:.2f}%'
#                 })
        
#         avg_loss = total_loss / len(self.val_loader)
#         avg_acc = 100. * correct / total
        
#         return avg_loss, avg_acc
    
#     def train(self, num_epochs=10):
#         """Main training loop"""
#         logger.info(f"Starting training for {num_epochs} epochs...")
        
#         best_val_loss = float('inf')
        
#         for epoch in range(1, num_epochs + 1):
#             logger.info(f"\n{'='*60}")
#             logger.info(f"Epoch {epoch}/{num_epochs}")
#             logger.info(f"{'='*60}")
            
#             # Train
#             train_loss, train_acc = self.train_epoch(epoch)
#             logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
#             # Validate
#             val_loss, val_acc = self.validate(epoch)
#             logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
#             # Save checkpoint
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 checkpoint_path = self.checkpoint_dir / 'best_model.pth'
#                 torch.save(self.model.state_dict(), checkpoint_path)
#                 logger.info(f"✓ Saved best model to {checkpoint_path}")
            
#             # Step scheduler
#             self.scheduler.step()
        
#         # Save final model
#         final_path = self.checkpoint_dir / 'final_model.pth'
#         torch.save(self.model.state_dict(), final_path)
#         logger.info(f"\n✓ Saved final model to {final_path}")
#         logger.info("\n" + "="*60)
#         logger.info("TRAINING COMPLETE!")
#         logger.info("="*60)

# # ============================================
# # MAIN
# # ============================================
# def main():
#     parser = argparse.ArgumentParser(
#         description='Train image deepfake detection model',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#     python training/train_image_model_fixed.py --dataset dataset/image --epochs 10
#     python training/train_image_model_fixed.py --dataset dataset/image --config training/configs/image_config.yaml
#         """
#     )
    
#     parser.add_argument('--dataset', type=str, default='dataset/image',
#                        help='Path to image dataset')
#     parser.add_argument('--config', type=str, default=None,
#                        help='Path to config file')
#     parser.add_argument('--epochs', type=int, default=10,
#                        help='Number of epochs')
#     parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
#                        help='Directory to save checkpoints')
    
#     args = parser.parse_args()
    
#     # Setup logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     logger.info("="*60)
#     logger.info("IMAGE DEEPFAKE DETECTION TRAINING")
#     logger.info("="*60)
#     logger.info(f"Backend root: {BACKEND_ROOT}")
#     logger.info(f"Dataset: {args.dataset}")
#     logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
#     # Create trainer
#     trainer = ImageTrainer(
#         config_path=args.config,
#         dataset_root=args.dataset,
#         checkpoint_dir=args.checkpoint_dir
#     )
    
#     # Train
#     trainer.train(num_epochs=args.epochs)

# if __name__ == '__main__':
#     main()


"""
FIXED ADVANCED IMAGE MODEL TRAINING
Place in: backend/training/train_image_model.py
All errors fixed - ready to use!
"""
import os
import sys
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import timm
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ============================================
# ADVANCED DATASET
# ============================================
class AdvancedDeepfakeImageDataset(Dataset):
    """Advanced dataset with proper preprocessing"""
    
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
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(img_path)
                    self.labels.append(0)  # 0 = REAL
        
        # Load FAKE images
        fake_dir = self.root_dir / split / 'FAKE'
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob('*')):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(img_path)
                    self.labels.append(1)  # 1 = FAKE
        
        logger.info(f"Loaded {len(self.images)} images from {split} split "
                   f"({sum(1 for l in self.labels if l==0)} REAL, "
                   f"{sum(1 for l in self.labels if l==1)} FAKE)")
    
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
            return torch.randn(3, 224, 224), label


# ============================================
# ADVANCED MODEL ARCHITECTURES
# ============================================
class XceptionModel(nn.Module):
    """Xception architecture for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        self.feature_dim = 2048
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class VisionTransformerModel(nn.Module):
    """Vision Transformer for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.feature_dim = 768
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetModel(nn.Module):
    """EfficientNet architecture"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        self.feature_dim = 1280
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============================================
# TRAINER CLASS - ✅ FIXED SIGNATURE
# ============================================
class ImageTrainer:
    """Advanced trainer with visualization and metrics"""
    
    def __init__(self, dataset_root, architecture='xception', epochs=20, batch_size=16,
                 learning_rate=0.0001, weight_decay=0.0001, num_workers=4,
                 checkpoint_dir='checkpoints/image', experiment_name='image_deepfake'):
        """
        ✅ FIXED: Accept parameters that match main() function call
        """
        self.dataset_root = Path(dataset_root)
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
        
        # Build model
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader, self.val_loader = self.setup_dataloaders()
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def build_model(self):
        """Build model based on architecture choice"""
        arch = self.architecture.lower()
        
        if arch == 'xception':
            model = XceptionModel(num_classes=2, pretrained=True)
            logger.info("Built Xception model")
        elif arch == 'vit':
            model = VisionTransformerModel(num_classes=2, pretrained=True)
            logger.info("Built Vision Transformer model")
        elif arch == 'efficientnet':
            model = EfficientNetModel(num_classes=2, pretrained=True)
            logger.info("Built EfficientNet model")
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        return model
    
    def setup_dataloaders(self):
        """Setup training and validation data loaders"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = AdvancedDeepfakeImageDataset(
            self.dataset_root,
            split='train',
            transform=train_transform
        )
        
        val_dataset = AdvancedDeepfakeImageDataset(
            self.dataset_root,
            split='validation',
            transform=val_transform
        )
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(f"No datasets found in {self.dataset_root}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
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
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
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
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pth')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            logger.info(f"✓ Saved best model with val_acc={self.best_val_acc:.2f}%")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.epochs} epochs...")
        logger.info("="*60)
        
        for epoch in range(1, self.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.epochs}")
            logger.info("-"*60)
            
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            val_loss, val_acc = self.validate(epoch)
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best)
            
            self.scheduler.step()
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        logger.info("="*60)
        
        self.plot_training_curves()
        self.writer.close()


# ============================================
# MAIN - ✅ FIXED
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Train advanced image deepfake detection model')
    parser.add_argument('--dataset', type=str, default='dataset/image', help='Dataset root path')
    parser.add_argument('--architecture', type=str, default='xception',
                       choices=['xception', 'vit', 'efficientnet'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/image', help='Checkpoint dir')
    parser.add_argument('--experiment-name', type=str, default='image_deepfake', help='Experiment name')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("="*60)
    logger.info("IMAGE DEEPFAKE DETECTION TRAINING")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*60)
    
    # ✅ FIXED: Pass all parameters correctly
    trainer = ImageTrainer(
        dataset_root=args.dataset,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name
    )
    
    trainer.train()


if __name__ == '__main__':
    main()