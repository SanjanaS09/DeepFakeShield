# #!/usr/bin/env python3
# """
# FIXED VIDEO MODEL TRAINING SCRIPT  
# Place in: backend/training/train_video_model_fixed.py

# Properly integrates with your project structure.
# """

# import os
# import sys
# from pathlib import Path

# # FIX: Add backend root to path
# BACKEND_ROOT = Path(__file__).parent.parent
# sys.path.insert(0, str(BACKEND_ROOT))

# import argparse
# import logging
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import cv2
# import numpy as np
# from tqdm import tqdm

# logger = logging.getLogger(__name__)

# # ============================================
# # VIDEO DATASET
# # ============================================
# class DeepfakeVideoDataset(Dataset):
#     """Dataset for video-based deepfake detection"""
    
#     def __init__(self, root_dir, split='train', num_frames=16):
#         self.root_dir = Path(root_dir)
#         self.split = split
#         self.num_frames = num_frames
        
#         self.videos = []
#         self.labels = []
        
#         # Load REAL videos
#         real_dir = self.root_dir / split / 'REAL'
#         if real_dir.exists():
#             for vid_path in sorted(real_dir.glob('*')):
#                 if vid_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
#                     self.videos.append(vid_path)
#                     self.labels.append(0)  # REAL
        
#         # Load FAKE videos
#         fake_dir = self.root_dir / split / 'FAKE'
#         if fake_dir.exists():
#             for vid_path in sorted(fake_dir.glob('*')):
#                 if vid_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
#                     self.videos.append(vid_path)
#                     self.labels.append(1)  # FAKE
        
#         logger.info(f"Loaded {len(self.videos)} videos from {split} split")
    
#     def __len__(self):
#         return len(self.videos)
    
#     def _extract_frames(self, video_path):
#         """Extract evenly spaced frames from video"""
#         try:
#             cap = cv2.VideoCapture(str(video_path))
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             if total_frames == 0:
#                 return None
            
#             frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
#             frames = []
            
#             for idx in frame_indices:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#                 ret, frame = cap.read()
                
#                 if ret:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     frame = cv2.resize(frame, (224, 224))
#                     frames.append(frame)
            
#             cap.release()
            
#             if len(frames) == self.num_frames:
#                 return np.array(frames)
#         except Exception as e:
#             logger.warning(f"Error processing {video_path}: {e}")
        
#         return None
    
#     def __getitem__(self, idx):
#         video_path = self.videos[idx]
#         label = self.labels[idx]
        
#         frames = self._extract_frames(video_path)
        
#         if frames is None:
#             frames = np.random.randint(0, 255, (self.num_frames, 224, 224, 3), dtype=np.uint8)
        
#         # Convert to tensor
#         frames = torch.from_numpy(frames).float() / 255.0
#         frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
#         return frames, label

# # ============================================
# # VIDEO MODEL (3D CNN)
# # ============================================
# class Simple3DCNN(nn.Module):
#     """Simple 3D CNN for video classification"""
    
#     def __init__(self, num_classes=2):
#         super().__init__()
        
#         self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
#         self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(128)
#         self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.fc = nn.Linear(128, num_classes)
    
#     def forward(self, x):
#         # x: (B, T, C, H, W) -> (B, C, T, H, W)
#         x = x.permute(0, 2, 1, 3, 4)
        
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.maxpool2(x)
        
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
        
#         return x

# # ============================================
# # TRAINER
# # ============================================
# class VideoTrainer:
#     """Trainer for video deepfake detection"""
    
#     def __init__(self, dataset_root, checkpoint_dir='checkpoints'):
#         self.dataset_root = Path(dataset_root)
#         self.checkpoint_dir = Path(checkpoint_dir)
#         self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         logger.info(f"Using device: {self.device}")
        
#         # Model
#         self.model = Simple3DCNN(num_classes=2)
#         self.model = self.model.to(self.device)
        
#         # Data loaders
#         self._setup_data_loaders()
        
#         # Loss, optimizer
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
    
#     def _setup_data_loaders(self):
#         """Setup data loaders"""
#         logger.info(f"Loading from {self.dataset_root}")
        
#         train_dataset = DeepfakeVideoDataset(
#             self.dataset_root,
#             split='train',
#             num_frames=16
#         )
        
#         val_dataset = DeepfakeVideoDataset(
#             self.dataset_root,
#             split='validation',
#             num_frames=16
#         )
        
#         if len(train_dataset) == 0 or len(val_dataset) == 0:
#             logger.error(f"No videos found in {self.dataset_root}")
#             raise FileNotFoundError(f"Datasets not found")
        
#         self.train_loader = DataLoader(
#             train_dataset,
#             batch_size=4,  # Smaller for videos
#             shuffle=True,
#             num_workers=0
#         )
        
#         self.val_loader = DataLoader(
#             val_dataset,
#             batch_size=4,
#             shuffle=False,
#             num_workers=0
#         )
        
#         logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
#     def train_epoch(self, epoch):
#         """Train one epoch"""
#         self.model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        
#         for frames, labels in pbar:
#             frames = frames.to(self.device)
#             labels = labels.to(self.device)
            
#             outputs = self.model(frames)
#             loss = self.criterion(outputs, labels)
            
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)
            
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{100.*correct/total:.2f}%'
#             })
        
#         return total_loss / len(self.train_loader), 100. * correct / total
    
#     def validate(self, epoch):
#         """Validate"""
#         self.model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
        
#         with torch.no_grad():
#             for frames, labels in pbar:
#                 frames = frames.to(self.device)
#                 labels = labels.to(self.device)
                
#                 outputs = self.model(frames)
#                 loss = self.criterion(outputs, labels)
                
#                 total_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 correct += predicted.eq(labels).sum().item()
#                 total += labels.size(0)
                
#                 pbar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{100.*correct/total:.2f}%'
#                 })
        
#         return total_loss / len(self.val_loader), 100. * correct / total
    
#     def train(self, num_epochs=5):
#         """Main training loop"""
#         logger.info(f"Starting video training for {num_epochs} epochs...")
        
#         best_val_loss = float('inf')
        
#         for epoch in range(1, num_epochs + 1):
#             logger.info(f"\n{'='*60}")
#             logger.info(f"Epoch {epoch}/{num_epochs}")
#             logger.info(f"{'='*60}")
            
#             train_loss, train_acc = self.train_epoch(epoch)
#             logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
#             val_loss, val_acc = self.validate(epoch)
#             logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 checkpoint_path = self.checkpoint_dir / 'best_video_model.pth'
#                 torch.save(self.model.state_dict(), checkpoint_path)
#                 logger.info(f"✓ Saved best model")
            
#             self.scheduler.step()
        
#         final_path = self.checkpoint_dir / 'final_video_model.pth'
#         torch.save(self.model.state_dict(), final_path)
#         logger.info(f"\n✓ Training complete! Model saved to {final_path}")

# # ============================================
# # MAIN
# # ============================================
# def main():
#     parser = argparse.ArgumentParser(description='Train video deepfake detection model')
#     parser.add_argument('--dataset', type=str, default='dataset/video', help='Dataset path')
#     parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
#     parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint dir')
    
#     args = parser.parse_args()
    
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
#     logger.info("="*60)
#     logger.info("VIDEO DEEPFAKE DETECTION TRAINING")
#     logger.info("="*60)
    
#     trainer = VideoTrainer(
#         dataset_root=args.dataset,
#         checkpoint_dir=args.checkpoint_dir
#     )
    
#     trainer.train(num_epochs=args.epochs)

# if __name__ == '__main__':
#     main()


"""
FIXED ADVANCED VIDEO MODEL TRAINING
Place in: backend/training/train_video_model.py
"""
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ============================================
# VIDEO DATASET
# ============================================
class AdvancedDeepfakeVideoDataset(Dataset):
    """Advanced dataset for video-based deepfake detection"""
    
    def __init__(self, root_dir, split='train', num_frames=16):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.videos = []
        self.labels = []
        
        # Load REAL videos
        real_dir = self.root_dir / split / 'REAL'
        if real_dir.exists():
            for vid_path in sorted(real_dir.glob('*')):
                if vid_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    self.videos.append(vid_path)
                    self.labels.append(0)
        
        # Load FAKE videos
        fake_dir = self.root_dir / split / 'FAKE'
        if fake_dir.exists():
            for vid_path in sorted(fake_dir.glob('*')):
                if vid_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    self.videos.append(vid_path)
                    self.labels.append(1)
        
        logger.info(f"Loaded {len(self.videos)} videos from {split} split "
                   f"({sum(1 for l in self.labels if l==0)} REAL, "
                   f"{sum(1 for l in self.labels if l==1)} FAKE)")
    
    def __len__(self):
        return len(self.videos)
    
    def _extract_frames(self, video_path):
        """Extract evenly spaced frames from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return None
            
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) == self.num_frames:
                return np.array(frames)
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
        
        return None
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        frames = self._extract_frames(video_path)
        
        if frames is None:
            frames = np.random.randint(0, 255, (self.num_frames, 224, 224, 3), dtype=np.uint8)
        
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # T, H, W, C -> T, C, H, W
        
        return frames, label


# ============================================
# 3D CNN MODEL
# ============================================
class Simple3DCNN(nn.Module):
    """Simple 3D CNN for video classification"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x: (batch, T, C, H, W) -> (batch, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# ============================================
# TRAINER CLASS - ✅ FIXED
# ============================================
class VideoTrainer:
    """Trainer for video deepfake detection"""
    
    def __init__(self, dataset_root, epochs=15, batch_size=4, learning_rate=0.001,
                 weight_decay=0.0001, num_workers=4, checkpoint_dir='checkpoints/video',
                 experiment_name='video_deepfake', num_frames=16):
        """✅ FIXED: Accept all parameters"""
        self.dataset_root = Path(dataset_root)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.num_frames = num_frames
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
        
        self.model = Simple3DCNN(num_classes=2).to(self.device)
        self.train_loader, self.val_loader = self.setup_dataloaders()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def setup_dataloaders(self):
        """Setup data loaders"""
        logger.info(f"Loading videos from {self.dataset_root}")
        
        train_dataset = AdvancedDeepfakeVideoDataset(
            self.dataset_root,
            split='train',
            num_frames=self.num_frames
        )
        
        val_dataset = AdvancedDeepfakeVideoDataset(
            self.dataset_root,
            split='validation',
            num_frames=self.num_frames
        )
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(f"No videos found in {self.dataset_root}")
        
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
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for frames, labels in pbar:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(frames)
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
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for frames, labels in pbar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
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
        """Plot training curves"""
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


def main():
    parser = argparse.ArgumentParser(description='Train advanced video deepfake detection model')
    parser.add_argument('--dataset', type=str, default='dataset/video', help='Dataset root')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/video', help='Checkpoint dir')
    parser.add_argument('--experiment-name', type=str, default='video_deepfake', help='Experiment name')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to extract')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("="*60)
    logger.info("VIDEO DEEPFAKE DETECTION TRAINING")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*60)
    
    trainer = VideoTrainer(
        dataset_root=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        num_frames=args.num_frames
    )
    
    trainer.train()


if __name__ == '__main__':
    main()