"""
Image Deepfake Detector
Specialized model for detecting manipulated images using CNNs and Vision Transformers
Supports Xception, ResNet, EfficientNet, and Vision Transformer backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import cv2
import numpy as np
from typing import Dict, Tuple, Union, Optional
import logging
from PIL import Image

from .base_model import BaseDetectionModel

logger = logging.getLogger(__name__)

class ImageDetector(BaseDetectionModel):
    """
    Image-based deepfake detection model
    Supports multiple backbone architectures and includes XAI capabilities
    """

    def __init__(self, 
                 backbone: str = 'xception',
                 num_classes: int = 2,
                 device: str = 'cpu',
                 pretrained: bool = True,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize Image Detector

        Args:
            backbone: Backbone architecture ('xception', 'resnet50', 'efficientnet-b0', 'vit-b16')
            num_classes: Number of classes (2 for binary classification)
            device: Device to run on
            pretrained: Use pretrained weights
            input_size: Input image size (height, width)
        """
        super().__init__(f'image_detector_{backbone}', num_classes, device, pretrained)

        self.backbone_name = backbone
        self.input_size = input_size
        self.input_shape = (3, input_size[0], input_size[1])

        # Build model
        self.build_model()

        # Move to device
        self.to(self.device)

        # Initialize transforms
        self._init_transforms()

        logger.info(f"Initialized ImageDetector with {backbone} backbone")

    def build_model(self) -> None:
        """Build the image detection model architecture"""

        if self.backbone_name == 'xception':
            self.backbone = timm.create_model('xception', pretrained=self.pretrained, num_classes=0)
            self.feature_dim = 2048

        elif self.backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=self.pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final FC
            self.feature_dim = 2048

        elif self.backbone_name == 'efficientnet-b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=self.pretrained, num_classes=0)
            self.feature_dim = 1280

        elif self.backbone_name == 'vit-b16':
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=self.pretrained, num_classes=0)
            self.feature_dim = 768

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Feature extraction layers for artifact detection
        self.artifact_detector = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim + 256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def _init_transforms(self) -> None:
        """Initialize image preprocessing transforms"""

        # Standard normalization for pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Transform for artifact detection (no normalization)
        self.artifact_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def preprocess_input(self, input_data) -> torch.Tensor:
        """
        Preprocess input image data

        Args:
            input_data: Input image (PIL Image, numpy array, or path string)

        Returns:
            Preprocessed tensor
        """
        # Handle different input types
        if isinstance(input_data, str):
            # Load image from path
            image = Image.open(input_data).convert('RGB')
        elif isinstance(input_data, np.ndarray):
            # Convert numpy array to PIL
            if input_data.dtype == np.uint8:
                image = Image.fromarray(input_data)
            else:
                image = Image.fromarray((input_data * 255).astype(np.uint8))
        elif isinstance(input_data, Image.Image):
            image = input_data.convert('RGB')
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Apply transforms and add batch dimension
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images

        Args:
            x: Input tensor

        Returns:
            Extracted features
        """
        batch_size = x.shape[0]

        # Extract semantic features using backbone
        semantic_features = self.backbone(x)

        # Handle different backbone output shapes
        if semantic_features.dim() > 2:
            semantic_features = F.adaptive_avg_pool2d(semantic_features, (1, 1))
            semantic_features = semantic_features.view(batch_size, -1)

        # Extract artifact features
        artifact_features = self.artifact_detector(x)

        # Concatenate features
        combined_features = torch.cat([semantic_features, artifact_features], dim=1)

        # Apply attention mechanism
        combined_features = combined_features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(
            combined_features, combined_features, combined_features
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension

        return attended_features

    def get_feature_breakdown(self, input_data) -> Dict[str, float]:
        """
        Get detailed feature breakdown for interpretability

        Args:
            input_data: Input image

        Returns:
            Dictionary with feature scores
        """
        self.eval()

        with torch.no_grad():
            # Preprocess input
            x = self.preprocess_input(input_data)
            if x.device != self.device:
                x = x.to(self.device)

            # Convert to numpy for analysis
            if isinstance(input_data, str):
                img_np = np.array(Image.open(input_data).convert('RGB'))
            elif isinstance(input_data, Image.Image):
                img_np = np.array(input_data.convert('RGB'))
            elif isinstance(input_data, np.ndarray):
                img_np = input_data
            else:
                img_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)

            # Calculate various artifact scores
            scores = {}

            # Blur detection using Laplacian variance
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores['blur_score'] = float(1.0 - min(blur_score / 1000.0, 1.0))  # Normalize and invert

            # Noise analysis using standard deviation of high-frequency components
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            high_freq_mask = np.zeros_like(magnitude_spectrum)
            h, w = high_freq_mask.shape
            high_freq_mask[h//4:3*h//4, w//4:3*w//4] = 1
            high_freq_energy = np.sum(magnitude_spectrum * (1 - high_freq_mask))
            total_energy = np.sum(magnitude_spectrum)
            scores['noise_score'] = float(high_freq_energy / total_energy) if total_energy > 0 else 0.0

            # Compression artifacts detection
            # Simple JPEG blocking artifacts detection
            block_size = 8
            blocking_score = 0.0
            for i in range(0, gray.shape[0] - block_size, block_size):
                for j in range(0, gray.shape[1] - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    if block.shape == (block_size, block_size):
                        block_var = np.var(block)
                        if block_var < 100:  # Low variance indicates possible blocking
                            blocking_score += 1

            total_blocks = (gray.shape[0] // block_size) * (gray.shape[1] // block_size)
            scores['compression_artifacts'] = float(blocking_score / max(total_blocks, 1))

            # Facial warping detection (simplified)
            # This would typically require face detection and landmark analysis
            # For now, use edge density as a proxy
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            scores['facial_warping'] = float(min(edge_density * 2, 1.0))  # Normalize

            # Temporal inconsistency (not applicable for single images)
            scores['temporal_inconsistency'] = 0.0

            return scores

    def get_attention_maps(self, input_data) -> np.ndarray:
        """
        Generate attention maps for visualization

        Args:
            input_data: Input image

        Returns:
            Attention map as numpy array
        """
        self.eval()

        # Enable gradient computation
        x = self.preprocess_input(input_data)
        if x.device != self.device:
            x = x.to(self.device)
        x.requires_grad_(True)

        # Forward pass
        outputs = self.forward(x)

        # Get prediction
        pred_class = torch.argmax(outputs['probabilities'], dim=1)
        class_score = outputs['probabilities'][0, pred_class]

        # Backward pass
        self.zero_grad()
        class_score.backward()

        # Get gradients
        gradients = x.grad.data

        # Generate attention map using gradients
        attention_map = torch.mean(torch.abs(gradients), dim=1).squeeze().cpu().numpy()

        # Normalize to 0-1
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        return attention_map

    def detect_face_regions(self, input_data) -> Dict[str, Union[np.ndarray, list]]:
        """
        Detect and analyze face regions in the image

        Args:
            input_data: Input image

        Returns:
            Dictionary with face detection results
        """
        # Convert input to numpy array
        if isinstance(input_data, str):
            img_np = np.array(Image.open(input_data).convert('RGB'))
        elif isinstance(input_data, Image.Image):
            img_np = np.array(input_data.convert('RGB'))
        elif isinstance(input_data, np.ndarray):
            img_np = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_regions = []
        for (x, y, w, h) in faces:
            face_region = {
                'bbox': [int(x), int(y), int(w), int(h)],
                'center': [int(x + w//2), int(y + h//2)],
                'area': int(w * h)
            }
            face_regions.append(face_region)

        return {
            'num_faces': len(faces),
            'face_regions': face_regions,
            'face_image': img_np
        }

    def analyze_image_quality(self, input_data) -> Dict[str, float]:
        """
        Analyze overall image quality metrics

        Args:
            input_data: Input image

        Returns:
            Dictionary with quality metrics
        """
        # Convert to numpy if needed
        if isinstance(input_data, str):
            img_np = np.array(Image.open(input_data).convert('RGB'))
        elif isinstance(input_data, Image.Image):
            img_np = np.array(input_data.convert('RGB'))
        elif isinstance(input_data, np.ndarray):
            img_np = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Calculate quality metrics
        quality_metrics = {}

        # Contrast (RMS contrast)
        quality_metrics['contrast'] = float(np.std(gray))

        # Brightness
        quality_metrics['brightness'] = float(np.mean(gray))

        # Sharpness (using Laplacian variance)
        quality_metrics['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Saturation (for color images)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        quality_metrics['saturation'] = float(np.mean(hsv[:, :, 1]))

        # Resolution
        quality_metrics['resolution'] = img_np.shape[0] * img_np.shape[1]

        return quality_metrics
