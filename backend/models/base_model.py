"""
Base Detection Model
Foundation class for all deepfake detection models with common functionality
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseDetectionModel(nn.Module, ABC):
    """
    Abstract base class for all deepfake detection models
    Provides common functionality for feature extraction, prediction, and XAI
    """

    def __init__(self, 
                 model_name: str,
                 num_classes: int = 2,
                 device: str = 'cpu',
                 pretrained: bool = True):
        """
        Initialize base detection model

        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes (2 for binary classification)
            device: Device to run model on ('cpu', 'cuda', 'mps')
            pretrained: Whether to use pretrained weights
        """
        super(BaseDetectionModel, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.pretrained = pretrained
        self.is_trained = False

        # Common model components
        self.backbone = None
        self.classifier = None
        self.feature_extractor = None

        # Model metadata
        self.input_shape = None
        self.feature_dim = None
        self.version = "1.0.0"

        # Thresholds
        self.detection_threshold = 0.5
        self.confidence_threshold = 0.7

    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture - must be implemented by subclasses"""
        pass

    @abstractmethod
    def preprocess_input(self, input_data) -> torch.Tensor:
        """Preprocess input data - must be implemented by subclasses"""
        pass

    @abstractmethod
    def extract_features(self, input_data) -> torch.Tensor:
        """Extract features from input - must be implemented by subclasses"""
        pass

    # def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     """
    #     Forward pass through the model

    #     Args:
    #         x: Input tensor

    #     Returns:
    #         Dictionary containing logits, probabilities, and features
    #     """
    #     # Extract features
    #     features = self.extract_features(x)

    #     # Classification
    #     logits = self.classifier(features)
    #     probabilities = torch.softmax(logits, dim=1)

    #     return {
    #         'logits': logits,
    #         'probabilities': probabilities,
    #         'features': features
    #     }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.extract_features(x)
        
        # Classify
        logits = self.classifier(features)
        
        # ✅ RETURN ONLY LOGITS TENSOR
        return logits


    def predict(self, input_data, return_confidence=True) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Make prediction on input data

        Args:
            input_data: Input data to classify
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary with prediction results
        """
        self.eval()

        with torch.no_grad():
            # Preprocess input
            x = self.preprocess_input(input_data)
            if x.device != self.device:
                x = x.to(self.device)

            # Forward pass
            outputs = self.forward(x)

            probabilities = outputs['probabilities'].cpu().numpy()
            logits = outputs['logits'].cpu().numpy()

            # Get predictions
            predictions = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)

            results = {
                'prediction': predictions[0] if len(predictions) == 1 else predictions,
                'label': 'FAKE' if predictions[0] == 1 else 'REAL',
                'probabilities': probabilities[0] if len(probabilities) == 1 else probabilities,
            }

            if return_confidence:
                results['confidence'] = confidences[0] if len(confidences) == 1 else confidences
                results['is_confident'] = confidences[0] >= self.confidence_threshold

            return results

    def get_feature_breakdown(self, input_data) -> Dict[str, float]:
        """
        Get detailed feature breakdown for interpretability
        Must be implemented by subclasses for specific feature analysis

        Args:
            input_data: Input data to analyze

        Returns:
            Dictionary with feature importance scores
        """
        return {
            'blur_score': 0.0,
            'noise_score': 0.0, 
            'compression_artifacts': 0.0,
            'facial_warping': 0.0,
            'temporal_inconsistency': 0.0
        }

    def load_pretrained(self, model_path: Union[str, Path]) -> None:
        """
        Load pretrained model weights

        Args:
            model_path: Path to the pretrained model file
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return

            state_dict = torch.load(model_path, map_location=self.device)

            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                self.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                self.load_state_dict(state_dict['state_dict'])
            else:
                self.load_state_dict(state_dict)

            self.is_trained = True
            logger.info(f"Loaded pretrained model from {model_path}")

        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            raise

    def save_model(self, save_path: Union[str, Path], include_metadata: bool = True) -> None:
        """
        Save model weights and metadata

        Args:
            save_path: Path to save the model
            include_metadata: Whether to include model metadata
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                'model_state_dict': self.state_dict(),
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'version': self.version,
                'is_trained': self.is_trained
            }

            if include_metadata:
                save_dict.update({
                    'input_shape': self.input_shape,
                    'feature_dim': self.feature_dim,
                    'detection_threshold': self.detection_threshold,
                    'confidence_threshold': self.confidence_threshold
                })

            torch.save(save_dict, save_path)
            logger.info(f"Model saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Union[str, int, float, bool]]:
        """
        Get model information and metadata

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'version': self.version,
            'num_classes': self.num_classes,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': self.input_shape,
            'feature_dim': self.feature_dim,
            'detection_threshold': self.detection_threshold,
            'confidence_threshold': self.confidence_threshold
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def set_thresholds(self, detection_threshold: float = None, confidence_threshold: float = None) -> None:
        """
        Set detection and confidence thresholds

        Args:
            detection_threshold: Threshold for binary classification
            confidence_threshold: Minimum confidence for reliable prediction
        """
        if detection_threshold is not None:
            self.detection_threshold = max(0.0, min(1.0, detection_threshold))

        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))

    def get_gradients(self, input_data, target_class: int = None) -> torch.Tensor:
        """
        Get gradients for XAI analysis

        Args:
            input_data: Input data
            target_class: Target class for gradient computation

        Returns:
            Gradients tensor
        """
        self.eval()

        x = self.preprocess_input(input_data)
        if x.device != self.device:
            x = x.to(self.device)

        x.requires_grad_(True)

        outputs = self.forward(x)

        if target_class is None:
            target_class = torch.argmax(outputs['probabilities'], dim=1)

        score = outputs['probabilities'][0, target_class]
        score.backward()

        return x.grad

    def __repr__(self) -> str:
        """String representation of the model"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', num_classes={self.num_classes}, device='{self.device}')"
