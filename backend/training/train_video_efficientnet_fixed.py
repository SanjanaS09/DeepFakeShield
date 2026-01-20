"""
✅ FRAME-BASED VIDEO TRAINING - EFFICIENTNET-B0
Extracts frames from videos, trains frame classifier
Aggregates predictions → video-level confidence
Target: 85%+ accuracy, trains FAST on CPU/GPU
"""
from pathlib import Path
import sys

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
import random
from torchvision import transforms
from PIL import Image
import timm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# ✅ FRAME-BASED VIDEO DATASET
# ============================================
class FrameBasedVideoDataset(Dataset):
    """Extract frames from videos for frame-level training"""
    
    def __init__(self, root_dir, split='train', num_frames=8, frame_size=(224, 224)):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.videos = []
        self.labels = []
        
        # Transform for frames
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load videos
        real_dir = self.root_dir / split / 'REAL'
        if real_dir.exists():
            for vid in sorted(real_dir.glob('*')):
                if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.videos.append(vid)
                    self.labels.append(0)
        
        fake_dir = self.root_dir / split / 'FAKE'
        if fake_dir.exists():
            for vid in sorted(fake_dir.glob('*')):
                if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.videos.append(vid)
                    self.labels.append(1)
        
        # Shuffle
        combined = list(zip(self.videos, self.labels))
        random.shuffle(combined)
        self.videos, self.labels = zip(*combined) if combined else ([], [])
        self.videos = list(self.videos)
        self.labels = list(self.labels)
        
        logger.info(f"✓ Loaded {len(self.videos)} videos ({sum(self.labels)} FAKE, {len(self.videos) - sum(self.labels)} REAL)")
    
    def extract_frames(self, video_path):
        """Extract uniformly sampled frames"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return None
            
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tensor = self.transform(frame_pil)
                    frames.append(frame_tensor)
            
            cap.release()
            
            if len(frames) == self.num_frames:
                return torch.stack(frames)  # [T, C, H, W]
        
        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")
        
        return None
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        frames = self.extract_frames(video_path)
        
        if frames is None:
            # Return dummy frames if extraction fails
            frames = torch.randn(self.num_frames, 3, *self.frame_size)
        
        return frames, label


# ============================================
# ✅ EFFICIENTNET FRAME CLASSIFIER
# ============================================
class EfficientNetFrameClassifier(nn.Module):
    """Frame-level classifier using EfficientNet-B0"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension (1280 for EfficientNet-B0)
        feat_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass for frame batch
        x: [B, C, H, W] - batch of frames
        """
        features = self.backbone(x)  # [B, feat_dim]
        logits = self.classifier(features)  # [B, num_classes]
        return logits


# ============================================
# ✅ VIDEO CLASSIFIER (aggregates frames)
# ============================================
class VideoClassifier(nn.Module):
    """Aggregate frame predictions to video prediction"""
    
    def __init__(self, frame_classifier):
        super().__init__()
        self.frame_classifier = frame_classifier
    
    def forward(self, video_frames):
        """
        Forward pass for video frames
        video_frames: [B, T, C, H, W] - batch of frame sequences
        """
        B, T, C, H, W = video_frames.shape
        
        # Classify each frame: [B*T, C, H, W] → [B*T, 2]
        frames_flat = video_frames.view(B * T, C, H, W)
        frame_logits = self.frame_classifier(frames_flat)
        frame_probs = torch.softmax(frame_logits, dim=1)  # [B*T, 2]
        
        # Aggregate frame predictions to video: [B*T, 2] → [B, 2]
        frame_probs = frame_probs.view(B, T, 2)
        
        # ✅ METHOD: Average probability across frames
        video_probs = torch.mean(frame_probs, dim=1)  # [B, 2]
        
        return video_probs


# ============================================
# ✅ TRAINER
# ============================================
class VideoFrameTrainer:
    def __init__(self, dataset_root, epochs=30, batch_size=16, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        # Datasets
        self.train_ds = FrameBasedVideoDataset(dataset_root, 'train', num_frames=8)
        self.val_ds = FrameBasedVideoDataset(dataset_root, 'validation', num_frames=8)
        
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True, num_workers=2 if str(self.device) == 'cuda' else 0
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if str(self.device) == 'cuda' else 0
        )
        
        # Frame classifier
        frame_classifier = EfficientNetFrameClassifier(num_classes=2, pretrained=True)
        self.model = VideoClassifier(frame_classifier).to(self.device)
        
        # Class weights
        real_count = sum(1 for l in self.train_ds.labels if l == 0)
        fake_count = sum(1 for l in self.train_ds.labels if l == 1)
        
        weight_real = fake_count / (real_count + fake_count)
        weight_fake = real_count / (real_count + fake_count)
        
        logger.info(f"Class distribution - REAL: {real_count:4d} | FAKE: {fake_count:4d}")
        logger.info(f"Class weights - REAL: {weight_real:.3f}, FAKE: {weight_fake:.3f}")
        
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([weight_real, weight_fake]).to(self.device)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        self.best_acc = 0.0
        self.best_epoch = 0
        self.epochs = epochs
        
        Path('checkpoints/video').mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [TRAIN]")
        
        for video_frames, labels in pbar:
            # video_frames: [B, T, C, H, W]
            video_frames = video_frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            video_probs = self.model(video_frames)
            loss = self.criterion(video_probs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, pred = video_probs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.1f}%'})
        
        return total_loss / len(self.train_loader), 100.0 * correct / total
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for video_frames, labels in tqdm(self.val_loader, desc=f"Epoch {epoch} [VAL]"):
                video_frames = video_frames.to(self.device)
                labels = labels.to(self.device)
                
                video_probs = self.model(video_frames)
                loss = self.criterion(video_probs, labels)
                
                total_loss += loss.item()
                _, pred = video_probs.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(self.val_loader), 100.0 * correct / total
    
    def save_checkpoint(self, epoch, train_acc, val_acc, is_best=False):
        """Save checkpoint with metadata"""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'model_type': 'VideoFrameClassifier',
            'architecture': 'EfficientNet-B0',
            'num_frames': 8,
            'frame_size': (224, 224)
        }
        
        torch.save(ckpt, 'checkpoints/video/latest.pth')
        
        if is_best:
            torch.save(ckpt, 'checkpoints/video/best_model.pth')
            logger.info(f"✅ BEST MODEL SAVED: {val_acc:.2f}% (Epoch {epoch})")
    
    def train(self):
        logger.info("\n" + "="*70)
        logger.info("🎬 FRAME-BASED VIDEO TRAINING (EFFICIENTNET-B0)")
        logger.info("="*70)
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            logger.info(f"E{epoch:2d}: Train={train_acc:6.2f}% | Val={val_acc:6.2f}% | Loss={val_loss:.4f}")
            
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.best_epoch = epoch
                logger.info(f"     ⭐ NEW BEST: {self.best_acc:.2f}% @ Epoch {epoch}")
            
            self.save_checkpoint(epoch, train_acc, val_acc, is_best)
            self.scheduler.step()
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Best accuracy: {self.best_acc:.2f}% @ Epoch {self.best_epoch}")
        logger.info(f"   Model saved: checkpoints/video/best_model.pth")
        logger.info("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset/video')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    trainer = VideoFrameTrainer(
        dataset_root=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
