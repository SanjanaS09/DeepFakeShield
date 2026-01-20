"""
✅ FRAME-BASED VIDEO DETECTOR
Loads trained frame classifier, extracts video frames, aggregates predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import timm

logger = logging.getLogger(__name__)

class FrameBasedVideoDetector(nn.Module):
    """Detects deepfakes using frame-level classification"""
    
    def __init__(self, 
                 model_path: str = 'checkpoints/video/best_model.pth',
                 device: str = 'cpu',
                 num_frames: int = 8,
                 frame_size: tuple = (224, 224)):
        super().__init__()
        
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Load pretrained EfficientNet backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0
        )
        
        feat_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
        self.to(self.device)
        
        # Load checkpoint if available
        if Path(model_path).exists():
            self._load_checkpoint(model_path)
        else:
            logger.warning(f"Model checkpoint not found: {model_path}")
        
        self.eval()
        
        # Normalization transform
        self.normalize = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("✅ FrameBasedVideoDetector initialized")
    
    def _load_checkpoint(self, model_path: str):
        """Load checkpoint with error handling"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # Extract frame classifier state
                    frame_classifier_state = {}
                    for key, value in state_dict.items():
                        if key.startswith('frame_classifier.'):
                            # Remove 'frame_classifier.' prefix
                            new_key = key.replace('frame_classifier.', '')
                            frame_classifier_state[new_key] = value
                    
                    if frame_classifier_state:
                        # Load backbone
                        backbone_state = {k.replace('backbone.', ''): v 
                                        for k, v in frame_classifier_state.items() 
                                        if k.startswith('backbone.')}
                        if backbone_state:
                            self.backbone.load_state_dict(backbone_state, strict=False)
                        
                        # Load classifier
                        classifier_state = {k.replace('classifier.', ''): v 
                                          for k, v in frame_classifier_state.items() 
                                          if k.startswith('classifier.')}
                        if classifier_state:
                            self.classifier.load_state_dict(classifier_state, strict=False)
                    
                    logger.info("✓ Checkpoint loaded successfully")
                    
                    # Log metadata
                    if 'best_acc' in checkpoint:
                        logger.info(f"  Best accuracy: {checkpoint['best_acc']:.2f}%")
                    if 'epoch' in checkpoint:
                        logger.info(f"  Trained epochs: {checkpoint['epoch']}")
            
            else:
                # Direct state dict
                self.load_state_dict(checkpoint, strict=False)
        
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.warning("Using random initialization")
    
    def forward(self, frame_batch):
        """
        Forward pass for frame batch
        frame_batch: [B, C, H, W]
        """
        features = self.backbone(frame_batch)
        logits = self.classifier(features)
        return logits
    
    def extract_frames(self, video_path: str) -> np.ndarray:
        """Extract uniformly sampled frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.warning(f"No frames in video: {video_path}")
                return None
            
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tensor = self.normalize(frame_pil)
                    frames.append(frame_tensor)
            
            cap.release()
            
            if len(frames) == self.num_frames:
                return torch.stack(frames)  # [T, C, H, W]
        
        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")
        
        return None
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict deepfake probability for video
        
        Args:
            video_path: Path to video file
        
        Returns:
            Detection results
        """
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if frames is None:
                return {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': {'REAL': 0.5, 'FAKE': 0.5},
                    'error': 'Failed to extract frames'
                }
            
            frames = frames.unsqueeze(0).to(self.device)  # [1, T, C, H, W] → [T, C, H, W]
            
            # Classify frames
            with torch.no_grad():
                # Get frame-level logits
                frame_logits = self.forward(frames.squeeze(0))  # [T, 2]
                frame_probs = F.softmax(frame_logits, dim=1)  # [T, 2]
                
                # Aggregate: average probability across frames
                video_probs = torch.mean(frame_probs, dim=0)  # [2]
                confidence = video_probs[1].item()  # Probability of FAKE
            
            prediction = 'FAKE' if confidence > 0.5 else 'REAL'
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    'REAL': float(video_probs[0].item()),
                    'FAKE': float(video_probs[1].item())
                },
                'frames_analyzed': self.num_frames,
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'REAL': 0.5, 'FAKE': 0.5},
                'error': str(e),
                'status': 'error'
            }
