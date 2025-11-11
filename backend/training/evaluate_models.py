#!/usr/bin/env python3
"""
UNIFIED EVALUATION SCRIPT FOR ALL MODALITIES
Place in: backend/training/evaluate_all_models.py

Evaluates image, video, and audio models with automatic model type detection.
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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import librosa
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== IMAGE DATASET ==========
class ImageDataset(Dataset):
    def __init__(self, root_dir, split='test', transform=None):
        self.images = []
        self.labels = []
        
        real_dir = Path(root_dir) / split / 'REAL'
        if real_dir.exists():
            for img_path in sorted(real_dir.glob('*')):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(img_path)
                    self.labels.append(0)
        
        fake_dir = Path(root_dir) / split / 'FAKE'
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob('*')):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(img_path)
                    self.labels.append(1)
        
        self.transform = transform
        logger.info(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ========== VIDEO DATASET ==========
class VideoDataset(Dataset):
    def __init__(self, root_dir, split='test', num_frames=8, transform=None):
        self.videos = []
        self.labels = []
        self.num_frames = num_frames
        self.transform = transform
        
        real_dir = Path(root_dir) / split / 'REAL'
        if real_dir.exists():
            for vid_path in sorted(real_dir.glob('*')):
                if vid_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    self.videos.append(vid_path)
                    self.labels.append(0)
        
        fake_dir = Path(root_dir) / split / 'FAKE'
        if fake_dir.exists():
            for vid_path in sorted(fake_dir.glob('*')):
                if vid_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    self.videos.append(vid_path)
                    self.labels.append(1)
        
        logger.info(f"Loaded {len(self.videos)} videos")
    
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None
        
        step = max(1, total_frames // self.num_frames)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % step == 0 and len(frames) < self.num_frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames[:self.num_frames])
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        try:
            frames = self._extract_frames(self.videos[idx])
            if frames is None:
                return torch.randn(self.num_frames, 3, 224, 224), self.labels[idx]
            return frames, self.labels[idx]
        except:
            return torch.randn(self.num_frames, 3, 224, 224), self.labels[idx]

# ========== AUDIO DATASET ==========
class AudioDataset(Dataset):
    def __init__(self, root_dir, split='test', sr=22050, n_mfcc=13):
        self.audios = []
        self.labels = []
        self.sr = sr
        self.n_mfcc = n_mfcc
        
        real_dir = Path(root_dir) / split / 'REAL'
        if real_dir.exists():
            for audio_path in sorted(real_dir.glob('*')):
                if audio_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    self.audios.append(audio_path)
                    self.labels.append(0)
        
        fake_dir = Path(root_dir) / split / 'FAKE'
        if fake_dir.exists():
            for audio_path in sorted(fake_dir.glob('*')):
                if audio_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    self.audios.append(audio_path)
                    self.labels.append(1)
        
        logger.info(f"Loaded {len(self.audios)} audio files")
    
    def _extract_features(self, audio_path):
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc = np.mean(mfcc, axis=1)
            return torch.FloatTensor(mfcc)
        except:
            return torch.zeros(self.n_mfcc)
    
    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        features = self._extract_features(self.audios[idx])
        return features, self.labels[idx]

# ========== SIMPLE IMAGE MODEL ==========
class SimpleImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ========== SIMPLE VIDEO MODEL ==========
class SimpleVideoModel(nn.Module):
    def __init__(self, num_classes=2, num_frames=8):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights='DEFAULT')
        self.frame_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        features = self.frame_encoder(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_out, (h_n, c_n) = self.lstm(features)
        return self.classifier(h_n[-1])

# ========== SIMPLE AUDIO MODEL ==========
class SimpleAudioModel(nn.Module):
    def __init__(self, num_classes=2, input_size=13):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, 2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        lstm_out, (h_n, c_n) = self.lstm(x.unsqueeze(1))
        return self.classifier(h_n[-1])

# ========== EVALUATION FUNCTION ==========
def evaluate(model, dataloader, device, model_type='image'):
    model.eval()
    predictions = []
    confidences = []
    labels_list = []
    
    logger.info(f"Evaluating {model_type} model...")
    
    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'audio':
                data, labels = batch
                data = data.to(device)
            else:
                data, labels = batch
                data = data.to(device)
            
            labels = labels.to(device)
            outputs = model(data)
            conf = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            confidences.extend(conf[:, 1].cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(confidences), np.array(labels_list)

# ========== METRICS ==========
def calculate_metrics(predictions, confidences, labels):
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(labels, confidences)
    except:
        metrics['roc_auc'] = 0.0
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

# ========== PRINT RESULTS ==========
def print_results(metrics, model_type, model_path):
    logger.info("\n" + "="*70)
    logger.info(f"{model_type.upper()} MODEL EVALUATION RESULTS")
    logger.info("="*70)
    
    logger.info(f"\n📊 OVERALL METRICS:")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:    {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    
    logger.info(f"\n🔍 PER-CLASS METRICS:")
    logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    
    logger.info(f"\n📈 CONFUSION MATRIX:")
    logger.info(f"  TN: {metrics['tn']}  FP: {metrics['fp']}")
    logger.info(f"  FN: {metrics['fn']}  TP: {metrics['tp']}")
    
    total_errors = metrics['fp'] + metrics['fn']
    total = metrics['tn'] + metrics['fp'] + metrics['fn'] + metrics['tp']
    logger.info(f"\n⚠️  ERROR ANALYSIS:")
    logger.info(f"  Total Errors: {total_errors}/{total} ({(total_errors/total)*100:.2f}%)")
    
    logger.info("\n" + "="*70)

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(description='Evaluate all deepfake models')
    parser.add_argument('--model-type', type=str, required=True, choices=['image', 'video', 'audio'],
                        help='Type of model to evaluate')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-frames', type=int, default=8, help='Frames per video')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Model Path: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if args.model_type == 'image':
        model = SimpleImageModel()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = ImageDataset(Path(args.dataset).parent, split=Path(args.dataset).name, transform=transform)
    
    elif args.model_type == 'video':
        model = SimpleVideoModel(num_frames=args.num_frames)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = VideoDataset(Path(args.dataset).parent, split=Path(args.dataset).name, 
                              num_frames=args.num_frames, transform=transform)
    
    elif args.model_type == 'audio':
        model = SimpleAudioModel()
        dataset = AudioDataset(Path(args.dataset).parent, split=Path(args.dataset).name)
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    logger.info("✓ Model loaded successfully")
    
    # Evaluate
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    predictions, confidences, labels = evaluate(model, dataloader, device, args.model_type)
    
    # Results
    metrics = calculate_metrics(predictions, confidences, labels)
    print_results(metrics, args.model_type, args.model)
    
    logger.info(f"\n✓ Evaluation complete!")

if __name__ == '__main__':
    main()
