"""
✅ OPTIMIZED VIDEO MODEL TRAINING - FAST & EFFICIENT
Pre-extracts features once, then trains on cached data
Target: 85%+ accuracy, trains in MINUTES
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
import cv2
import numpy as np
from tqdm import tqdm
import random
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# ✅ FEATURE EXTRACTION (CACHED)
# ============================================
class FastFeatureExtractor:
    """Extract features ONCE and cache them"""
    
    def __init__(self, num_frames=8):
        self.num_frames = num_frames
    
    def extract_optical_flow(self, video_path):
        """Extract optical flow magnitude - FAST"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total < 2:
                return None
            
            frame_indices = np.linspace(0, total-1, self.num_frames, dtype=int)
            
            prev_gray = None
            flow_magnitudes = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Fast optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 
                        0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_magnitudes.append(np.mean(mag) / 50.0)
                else:
                    flow_magnitudes.append(0.0)
                
                prev_gray = gray
            
            cap.release()
            
            while len(flow_magnitudes) < self.num_frames:
                flow_magnitudes.append(0.0)
            
            return np.array(flow_magnitudes[:self.num_frames], dtype=np.float32)
        
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def extract_batch(self, video_paths):
        """Extract features from multiple videos"""
        features_list = []
        
        for video_path in tqdm(video_paths, desc="Extracting features"):
            features = self.extract_optical_flow(video_path)
            
            if features is None:
                features = np.random.randn(self.num_frames)
            
            features_list.append(features)
        
        return np.array(features_list, dtype=np.float32)


# ============================================
# ✅ CACHED DATASET
# ============================================
class CachedVideoDataset(Dataset):
    """Uses pre-extracted features"""
    
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(np.array(labels)).long()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(1), self.labels[idx]  # [T] -> [T, 1]


# ============================================
# ✅ SIMPLE LSTM MODEL
# ============================================
class SimpleLSTM(nn.Module):
    """Fast LSTM for optical flow sequences"""
    
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        # x: [B, T, 1]
        lstm_out, _ = self.lstm(x)  # [B, T, 128]
        
        # Attention
        attn_w = self.attention(lstm_out)  # [B, T, 1]
        weighted = torch.sum(lstm_out * attn_w, dim=1)  # [B, 128]
        
        # FC
        logits = self.fc(weighted)  # [B, 2]
        return logits


# ============================================
# ✅ TRAINER WITH CACHING
# ============================================
class OptimizedTrainer:
    def __init__(self, dataset_root, epochs=30, batch_size=32, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        self.dataset_root = Path(dataset_root)
        self.cache_dir = Path('feature_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load or extract features
        logger.info("Loading training data...")
        self.train_features, self.train_labels = self._load_or_extract('train')
        self.val_features, self.val_labels = self._load_or_extract('validation')
        
        # Create datasets
        self.train_ds = CachedVideoDataset(self.train_features, self.train_labels)
        self.val_ds = CachedVideoDataset(self.val_features, self.val_labels)
        
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Model
        self.model = SimpleLSTM(input_dim=1, hidden_dim=64, num_layers=2).to(self.device)
        
        # Loss with class weights
        real_count = sum(1 for l in self.train_labels if l == 0)
        fake_count = sum(1 for l in self.train_labels if l == 1)
        
        weight_real = fake_count / (real_count + fake_count)
        weight_fake = real_count / (real_count + fake_count)
        
        logger.info(f"Class weights - REAL: {weight_real:.3f}, FAKE: {weight_fake:.3f}")
        
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([weight_real, weight_fake]).to(self.device)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        self.best_acc = 0
        self.epochs = epochs
        
        Path('checkpoints/video').mkdir(parents=True, exist_ok=True)
    
    def _load_or_extract(self, split):
        """Load cached features or extract them"""
        cache_file = self.cache_dir / f'{split}_features.pkl'
        
        if cache_file.exists():
            logger.info(f"Loading cached features from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['features'], data['labels']
        
        # Extract features
        logger.info(f"Extracting features for {split}...")
        videos = []
        labels = []
        
        real_dir = self.dataset_root / split / 'REAL'
        if real_dir.exists():
            for vid in sorted(real_dir.glob('*')):
                if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    videos.append(vid)
                    labels.append(0)
        
        fake_dir = self.dataset_root / split / 'FAKE'
        if fake_dir.exists():
            for vid in sorted(fake_dir.glob('*')):
                if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    videos.append(vid)
                    labels.append(1)
        
        # Shuffle
        combined = list(zip(videos, labels))
        random.shuffle(combined)
        videos, labels = zip(*combined) if combined else ([], [])
        videos = list(videos)
        labels = list(labels)
        
        logger.info(f"Extracting {len(videos)} videos...")
        extractor = FastFeatureExtractor(num_frames=8)
        features = extractor.extract_batch(videos)
        
        # Cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels}, f)
        
        logger.info(f"Cached to {cache_file}")
        return features, labels
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [TRAIN]")
        
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, pred = logits.max(1)
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
            for features, labels in tqdm(self.val_loader, desc=f"Epoch {epoch} [VAL]"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, pred = logits.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(self.val_loader), 100.0 * correct / total
    
    def save_checkpoint(self, epoch, acc, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_acc
        }
        
        torch.save(ckpt, 'checkpoints/video/latest.pth')
        if is_best:
            torch.save(ckpt, 'checkpoints/video/best_model.pth')
            logger.info(f"✅ Best model: {acc:.2f}%")
    
    def train(self):
        logger.info("\n" + "="*70)
        logger.info("🎬 OPTIMIZED VIDEO MODEL TRAINING")
        logger.info("="*70)
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            logger.info(f"E{epoch}: Train={train_acc:.2f}% | Val={val_acc:.2f}%")
            
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                logger.info(f"🎯 New best: {self.best_acc:.2f}%")
            
            self.save_checkpoint(epoch, val_acc, is_best)
            self.scheduler.step()
        
        logger.info(f"\n✅ Training complete! Best: {self.best_acc:.2f}%\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset/video')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    trainer = OptimizedTrainer(
        dataset_root=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate
    )
    
    trainer.train()


if __name__ == '__main__':
    main()