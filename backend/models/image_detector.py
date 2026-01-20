"""
Image Deepfake Detector using ResNet18
Trained model for detecting manipulated images
"""
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class ImageDeepfakeDetector:
    """ResNet18-based image deepfake detector"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 pretrained: bool = True):
        """
        Initialize Image Deepfake Detector
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference ('cpu' or 'cuda')
            pretrained: Use pretrained ImageNet weights
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        logger.info(f"Initializing ImageDeepfakeDetector on {self.device}")
        
        # Build ResNet18 model
        self._build_model(pretrained)
        
        # Load checkpoint if provided
        if model_path and Path(model_path).exists():
            self._load_checkpoint(model_path)
        else:
            if model_path:
                logger.warning(f"Model checkpoint not found: {model_path}")
            logger.info("Using random/pretrained weights")
        
        # Setup transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_model(self, pretrained: bool = True):
        """Build ResNet18 model architecture"""
        # Load ResNet18
        self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # 2 classes: REAL, FAKE
        )
        
        self.backbone.to(self.device)
        logger.info("✓ ResNet18 model built successfully")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint with key remapping"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    epoch = checkpoint.get('epoch', 'unknown')
                    best_acc = checkpoint.get('best_val_acc', 0.0)
                    
                    # ✅ FIX: Remap keys from backbone.* to direct keys
                    remapped_state_dict = {}
                    for key, value in state_dict.items():
                        # Remove 'backbone.' prefix if present
                        if key.startswith('backbone.'):
                            new_key = key.replace('backbone.', '', 1)
                            remapped_state_dict[new_key] = value
                        else:
                            remapped_state_dict[key] = value
                    
                    # Load remapped state dict
                    self.backbone.load_state_dict(remapped_state_dict, strict=True)
                    logger.info(f"✓ Loaded checkpoint from epoch {epoch}")
                    logger.info(f"  Best validation accuracy: {best_acc:.2f}%")
                    
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    
                    # Remap keys
                    remapped_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('backbone.'):
                            new_key = key.replace('backbone.', '', 1)
                            remapped_state_dict[new_key] = value
                        else:
                            remapped_state_dict[key] = value
                    
                    self.backbone.load_state_dict(remapped_state_dict, strict=False)
                    logger.info("✓ Loaded model state_dict")
                else:
                    # Direct state dict
                    remapped_state_dict = {}
                    for key, value in checkpoint.items():
                        if key.startswith('backbone.'):
                            new_key = key.replace('backbone.', '', 1)
                            remapped_state_dict[new_key] = value
                        else:
                            remapped_state_dict[key] = value
                    
                    self.backbone.load_state_dict(remapped_state_dict, strict=False)
                    logger.info("✓ Loaded model weights")
            else:
                # Direct tensor dict (shouldn't happen but handle it)
                remapped_state_dict = {}
                for key, value in checkpoint.items():
                    if key.startswith('backbone.'):
                        new_key = key.replace('backbone.', '', 1)
                        remapped_state_dict[new_key] = value
                    else:
                        remapped_state_dict[key] = value
                
                self.backbone.load_state_dict(remapped_state_dict)
                logger.info("✓ Loaded model")
            
            self.backbone.eval()
            logger.info("✓ Model loaded and set to eval mode")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("⚠️  Continuing with pre-trained weights only")
            # Don't raise - allow fallback to pretrained weights
    
    def detect(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Detect deepfake in image
        
        Args:
            image_input: Image path, numpy array, or PIL Image
        
        Returns:
            Detection result dictionary
        """
        try:
            # Load image
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            elif isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
            
            # Preprocess
            image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.backbone(image_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)
            
            # Map to class labels
            class_names = ['REAL', 'FAKE']
            prediction = class_names[predicted_class.item()]
            confidence_score = float(confidence.item())
            
            logger.info(f"Prediction: {prediction} ({confidence_score:.2%})")
            
            return {
                'prediction': prediction,
                'confidence': confidence_score,
                'probabilities': {
                    'REAL': float(probs[0, 0].item()),
                    'FAKE': float(probs[0, 1].item())
                },
                'class_index': int(predicted_class.item()),
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'probabilities': {'REAL': 0.5, 'FAKE': 0.5},
                'error': str(e),
                'status': 'error'
            }
    
    def get_feature_breakdown(self, image_input: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Get feature-level breakdown of detection
        
        Args:
            image_input: Image path or array
        
        Returns:
            Feature breakdown dictionary
        """
        try:
            # Load image
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
                image_array = np.array(image)
            else:
                image_array = image_input
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            
            # Basic artifact analysis
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Blur detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var() / 1000.0
            blur_score = min(blur_score, 1.0)
            
            # Compression artifacts (JPEG quality detection)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Noise level
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            h, w = magnitude.shape
            
            # High frequency energy
            high_freq_mask = np.zeros_like(magnitude)
            high_freq_mask[:h//4, :] = 1
            high_freq_mask[3*h//4:, :] = 1
            high_freq_mask[:, :w//4] = 1
            high_freq_mask[:, 3*w//4:] = 1
            
            high_freq_energy = np.sum(magnitude * high_freq_mask)
            total_energy = np.sum(magnitude)
            noise_score = high_freq_energy / (total_energy + 1e-8)
            
            return {
                'blur_score': float(blur_score),
                'edge_density': float(edge_density),
                'noise_score': float(noise_score),
                'overall_artifact_score': float((blur_score + noise_score) / 2),
                'overall_visual_quality': float(1.0 - (blur_score + noise_score) / 2)
            }
        
        except Exception as e:
            logger.error(f"Feature breakdown error: {e}")
            return {
                'blur_score': 0.5,
                'edge_density': 0.5,
                'noise_score': 0.5,
                'overall_artifact_score': 0.5,
                'overall_visual_quality': 0.5
            }
