#!/usr/bin/env python3
"""
FIXED AUDIO MODEL TRAINING SCRIPT
Place in: backend/training/train_audio_model_fixed.py

Properly integrates with your project structure.
"""

import os
import sys
from pathlib import Path

# FIX: Add backend root to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================
# AUDIO DATASET
# ============================================
class DeepfakeAudioDataset(Dataset):
    """Dataset for audio-based deepfake detection"""
    
    def __init__(self, root_dir, split='train', sr=16000, duration=3):
        self.root_dir = Path(root_dir)
        self.split = split
        self.sr = sr
        self.duration = duration
        self.n_mfcc = 40
        
        self.audio_files = []
        self.labels = []
        
        # Load REAL audio
        real_dir = self.root_dir / split / 'REAL'
        if real_dir.exists():
            for audio_path in sorted(real_dir.glob('*')):
                if audio_path.suffix.lower() in ['.wav', '.mp3', '.ogg', '.flac']:
                    self.audio_files.append(audio_path)
                    self.labels.append(0)  # REAL
        
        # Load FAKE audio
        fake_dir = self.root_dir / split / 'FAKE'
        if fake_dir.exists():
            for audio_path in sorted(fake_dir.glob('*')):
                if audio_path.suffix.lower() in ['.wav', '.mp3', '.ogg', '.flac']:
                    self.audio_files.append(audio_path)
                    self.labels.append(1)  # FAKE
        
        logger.info(f"Loaded {len(self.audio_files)} audio files from {split} split")
    
    def __len__(self):
        return len(self.audio_files)
    
    def _extract_mfcc(self, audio_path):
        """Extract MFCC features from audio"""
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr, duration=self.duration)
            
            # Ensure fixed length
            max_len = int(self.sr * self.duration)
            if len(y) < max_len:
                y = np.pad(y, (0, max_len - len(y)), mode='constant')
            else:
                y = y[:max_len]
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
            
            # Normalize
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            
            return mfcc
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            return np.zeros((self.n_mfcc, int(self.sr * self.duration / 512)))
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        mfcc = self._extract_mfcc(audio_path)
        mfcc = torch.from_numpy(mfcc).float()
        
        return mfcc, label

# ============================================
# AUDIO MODEL (1D CNN)
# ============================================
class SimpleAudioCNN(nn.Module):
    """Simple 1D CNN for audio classification"""
    
    def __init__(self, num_classes=2, n_mfcc=40):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# ============================================
# TRAINER
# ============================================
class AudioTrainer:
    """Trainer for audio deepfake detection"""
    
    def __init__(self, dataset_root, checkpoint_dir='checkpoints'):
        self.dataset_root = Path(dataset_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = SimpleAudioCNN(num_classes=2, n_mfcc=40)
        self.model = self.model.to(self.device)
        
        # Data loaders
        self._setup_data_loaders()
        
        # Loss, optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        logger.info(f"Loading from {self.dataset_root}")
        
        train_dataset = DeepfakeAudioDataset(
            self.dataset_root,
            split='train',
            sr=16000,
            duration=3
        )
        
        val_dataset = DeepfakeAudioDataset(
            self.dataset_root,
            split='validation',
            sr=16000,
            duration=3
        )
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.error(f"No audio files found in {self.dataset_root}")
            raise FileNotFoundError(f"Datasets not found")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        
        for mfcc, labels in pbar:
            mfcc = mfcc.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(mfcc)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
        
        with torch.no_grad():
            for mfcc, labels in pbar:
                mfcc = mfcc.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(mfcc)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, num_epochs=10):
        """Main training loop"""
        logger.info(f"Starting audio training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            val_loss, val_acc = self.validate(epoch)
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir / 'best_audio_model.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"✓ Saved best model")
            
            self.scheduler.step()
        
        final_path = self.checkpoint_dir / 'final_audio_model.pth'
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"\n✓ Training complete! Model saved to {final_path}")

# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Train audio deepfake detection model')
    parser.add_argument('--dataset', type=str, default='dataset/audio', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint dir')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("="*60)
    logger.info("AUDIO DEEPFAKE DETECTION TRAINING")
    logger.info("="*60)
    
    trainer = AudioTrainer(
        dataset_root=args.dataset,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer.train(num_epochs=args.epochs)

if __name__ == '__main__':
    main()
