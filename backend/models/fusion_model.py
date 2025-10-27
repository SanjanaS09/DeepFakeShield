"""
Multi-Modal Fusion Model
Combines image, video, and audio features using transformer-based cross-attention
Supports both transformer fusion and weighted ensemble approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging
import math

from .base_model import BaseDetectionModel
from .image_detector import ImageDetector
from .video_detector import VideoDetector
from .audio_detector import AudioDetector

logger = logging.getLogger(__name__)

class FusionModel(BaseDetectionModel):
    """
    Multi-modal fusion model for deepfake detection
    Combines features from image, video, and audio modalities
    """

    def __init__(self,
                 fusion_type: str = 'transformer',
                 num_classes: int = 2,
                 device: str = 'cpu',
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """
        Initialize Fusion Model

        Args:
            fusion_type: Type of fusion ('transformer', 'attention', 'concat', 'weighted')
            num_classes: Number of output classes
            device: Device to run on
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__(f'fusion_model_{fusion_type}', num_classes, device, False)

        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Modality feature dimensions (these will be set when loading detectors)
        self.image_dim = 256
        self.video_dim = 256
        self.audio_dim = 256

        # Build fusion architecture
        self.build_model()

        # Move to device
        self.to(self.device)

        # Modality weights for weighted fusion
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially

        logger.info(f"Initialized FusionModel with {fusion_type} fusion")

    def build_model(self) -> None:
        """Build the fusion model architecture"""

        # Modality-specific projection layers
        self.image_projector = nn.Sequential(
            nn.Linear(self.image_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

        self.video_projector = nn.Sequential(
            nn.Linear(self.video_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

        self.audio_projector = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

        # Positional encoding for modalities
        self.modality_embeddings = nn.Parameter(torch.randn(3, self.hidden_dim))

        if self.fusion_type == 'transformer':
            self._build_transformer_fusion()
        elif self.fusion_type == 'attention':
            self._build_attention_fusion()
        elif self.fusion_type == 'concat':
            self._build_concat_fusion()
        elif self.fusion_type == 'weighted':
            self._build_weighted_fusion()
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        # Final classifier
        if self.fusion_type == 'concat':
            classifier_input_dim = self.hidden_dim * 3
        else:
            classifier_input_dim = self.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        self.feature_dim = classifier_input_dim

    def _build_transformer_fusion(self) -> None:
        """Build transformer-based fusion"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_rate,
            activation='relu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )

        # Cross-modal attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )

        # Final pooling
        self.fusion_pooling = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

    def _build_attention_fusion(self) -> None:
        """Build attention-based fusion"""
        self.attention_weights = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True)
        )

    def _build_concat_fusion(self) -> None:
        """Build concatenation-based fusion"""
        self.concat_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

    def _build_weighted_fusion(self) -> None:
        """Build weighted fusion"""
        self.weight_generator = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 3),
            nn.Softmax(dim=1)
        )

    def preprocess_input(self, input_data) -> torch.Tensor:
        """
        Preprocess multi-modal input data

        Args:
            input_data: Dictionary with 'image', 'video', 'audio' features

        Returns:
            Processed features dictionary
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary with modality keys")

        processed = {}

        # Process each modality
        for modality in ['image', 'video', 'audio']:
            if modality in input_data and input_data[modality] is not None:
                processed[modality] = input_data[modality]
            else:
                # Create dummy features if modality is missing
                if modality == 'image':
                    processed[modality] = torch.zeros(1, self.image_dim)
                elif modality == 'video':
                    processed[modality] = torch.zeros(1, self.video_dim)
                elif modality == 'audio':
                    processed[modality] = torch.zeros(1, self.audio_dim)

        return processed

    def extract_features(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract and fuse multi-modal features

        Args:
            input_data: Dictionary containing modality features

        Returns:
            Fused features
        """
        batch_size = list(input_data.values())[0].shape[0]

        # Project features to common dimension
        image_features = self.image_projector(input_data['image'])
        video_features = self.video_projector(input_data['video'])  
        audio_features = self.audio_projector(input_data['audio'])

        # Add modality embeddings
        image_features = image_features + self.modality_embeddings[0].unsqueeze(0)
        video_features = video_features + self.modality_embeddings[1].unsqueeze(0)
        audio_features = audio_features + self.modality_embeddings[2].unsqueeze(0)

        # Fuse features based on fusion type
        if self.fusion_type == 'transformer':
            fused_features = self._transformer_fusion(image_features, video_features, audio_features)
        elif self.fusion_type == 'attention':
            fused_features = self._attention_fusion(image_features, video_features, audio_features)
        elif self.fusion_type == 'concat':
            fused_features = self._concat_fusion(image_features, video_features, audio_features)
        elif self.fusion_type == 'weighted':
            fused_features = self._weighted_fusion(image_features, video_features, audio_features)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return fused_features

    def _transformer_fusion(self, image_feat: torch.Tensor, 
                           video_feat: torch.Tensor, 
                           audio_feat: torch.Tensor) -> torch.Tensor:
        """Apply transformer-based fusion"""
        # Stack modality features
        modality_features = torch.stack([image_feat, video_feat, audio_feat], dim=1)  # [B, 3, D]

        # Apply transformer encoder
        attended_features = self.transformer_encoder(modality_features)  # [B, 3, D]

        # Cross-modal attention
        cross_attended, attention_weights = self.cross_attention(
            attended_features, attended_features, attended_features
        )

        # Global pooling with learned attention
        pooling_weights = self.fusion_pooling(cross_attended)  # [B, 3, 1]
        pooling_weights = F.softmax(pooling_weights, dim=1)

        # Weighted sum
        fused_features = torch.sum(cross_attended * pooling_weights, dim=1)  # [B, D]

        return fused_features

    def _attention_fusion(self, image_feat: torch.Tensor,
                         video_feat: torch.Tensor, 
                         audio_feat: torch.Tensor) -> torch.Tensor:
        """Apply attention-based fusion"""
        # Stack features
        features = torch.stack([image_feat, video_feat, audio_feat], dim=1)  # [B, 3, D]

        # Compute attention weights
        attention_weights = self.attention_weights(features)  # [B, 3, 1]

        # Apply attention
        attended_features = features * attention_weights
        fused_features = torch.sum(attended_features, dim=1)  # [B, D]

        # Final processing
        fused_features = self.feature_fusion(fused_features)

        return fused_features

    def _concat_fusion(self, image_feat: torch.Tensor,
                      video_feat: torch.Tensor,
                      audio_feat: torch.Tensor) -> torch.Tensor:
        """Apply concatenation-based fusion"""
        # Concatenate features
        concatenated = torch.cat([image_feat, video_feat, audio_feat], dim=1)  # [B, 3*D]

        # Process concatenated features
        fused_features = self.concat_fusion(concatenated)

        return fused_features

    def _weighted_fusion(self, image_feat: torch.Tensor,
                        video_feat: torch.Tensor,
                        audio_feat: torch.Tensor) -> torch.Tensor:
        """Apply weighted fusion with learnable weights"""
        # Stack features for weight generation
        all_features = torch.cat([image_feat, video_feat, audio_feat], dim=1)

        # Generate dynamic weights
        dynamic_weights = self.weight_generator(all_features)  # [B, 3]

        # Apply weights
        weighted_image = image_feat * dynamic_weights[:, 0:1]
        weighted_video = video_feat * dynamic_weights[:, 1:2]
        weighted_audio = audio_feat * dynamic_weights[:, 2:3]

        # Sum weighted features
        fused_features = weighted_image + weighted_video + weighted_audio

        return fused_features

    def get_attention_weights(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for interpretability

        Args:
            input_data: Multi-modal input features

        Returns:
            Dictionary with attention weights for each modality
        """
        self.eval()

        with torch.no_grad():
            batch_size = list(input_data.values())[0].shape[0]

            # Project features
            image_features = self.image_projector(input_data['image'])
            video_features = self.video_projector(input_data['video'])
            audio_features = self.audio_projector(input_data['audio'])

            # Add modality embeddings
            image_features = image_features + self.modality_embeddings[0].unsqueeze(0)
            video_features = video_features + self.modality_embeddings[1].unsqueeze(0)
            audio_features = audio_features + self.modality_embeddings[2].unsqueeze(0)

            if self.fusion_type == 'transformer':
                # Get transformer attention weights
                modality_features = torch.stack([image_features, video_features, audio_features], dim=1)
                attended_features = self.transformer_encoder(modality_features)

                _, attention_weights = self.cross_attention(
                    attended_features, attended_features, attended_features
                )

                return {
                    'cross_attention': attention_weights,
                    'modality_importance': torch.mean(attention_weights, dim=1)
                }

            elif self.fusion_type == 'attention':
                # Get attention weights
                features = torch.stack([image_features, video_features, audio_features], dim=1)
                attention_weights = self.attention_weights(features).squeeze(-1)

                return {
                    'modality_weights': attention_weights,
                    'image_weight': attention_weights[:, 0],
                    'video_weight': attention_weights[:, 1], 
                    'audio_weight': attention_weights[:, 2]
                }

            elif self.fusion_type == 'weighted':
                # Get dynamic weights
                all_features = torch.cat([image_features, video_features, audio_features], dim=1)
                dynamic_weights = self.weight_generator(all_features)

                return {
                    'dynamic_weights': dynamic_weights,
                    'image_weight': dynamic_weights[:, 0],
                    'video_weight': dynamic_weights[:, 1],
                    'audio_weight': dynamic_weights[:, 2]
                }

            else:
                # For concat fusion, return equal weights
                equal_weights = torch.ones(batch_size, 3) / 3
                return {
                    'equal_weights': equal_weights,
                    'image_weight': equal_weights[:, 0],
                    'video_weight': equal_weights[:, 1],
                    'audio_weight': equal_weights[:, 2]
                }

    def get_modality_contributions(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Get individual modality contributions to final prediction

        Args:
            input_data: Multi-modal input features

        Returns:
            Dictionary with modality contribution scores
        """
        self.eval()

        # Get full prediction
        full_output = self.predict(input_data, return_confidence=True)
        full_confidence = full_output['confidence']

        contributions = {}

        # Test each modality individually (zero out others)
        for modality in ['image', 'video', 'audio']:
            # Create input with only this modality
            single_modal_input = {}
            for key in input_data.keys():
                if key == modality:
                    single_modal_input[key] = input_data[key]
                else:
                    # Zero out other modalities
                    single_modal_input[key] = torch.zeros_like(input_data[key])

            # Get prediction with single modality
            single_output = self.predict(single_modal_input, return_confidence=True)
            single_confidence = single_output['confidence']

            # Calculate contribution as confidence difference
            contributions[f'{modality}_contribution'] = float(single_confidence)

        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for key in contributions:
                contributions[key] = contributions[key] / total_contribution

        contributions['full_confidence'] = float(full_confidence)

        return contributions

    def get_feature_breakdown(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Get feature breakdown from fused multi-modal analysis

        Args:
            input_data: Multi-modal input features

        Returns:
            Aggregated feature scores from all modalities
        """
        # This would typically aggregate scores from individual modality detectors
        # For now, return placeholder values
        return {
            'blur_score': 0.0,
            'noise_score': 0.0,
            'compression_artifacts': 0.0,
            'facial_warping': 0.0,
            'temporal_inconsistency': 0.0,
            'voice_quality': 0.0,
            'pitch_modulation': 0.0,
            'spectral_artifacts': 0.0
        }

    def ensemble_predict(self, predictions_list: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """
        Ensemble predictions from multiple models

        Args:
            predictions_list: List of prediction dictionaries
            weights: Optional weights for each prediction

        Returns:
            Ensembled prediction
        """
        if not predictions_list:
            raise ValueError("No predictions provided for ensemble")

        if weights is None:
            weights = [1.0 / len(predictions_list)] * len(predictions_list)

        # Ensemble probabilities
        ensemble_probs = np.zeros_like(predictions_list[0]['probabilities'])
        total_weight = 0.0

        for pred, weight in zip(predictions_list, weights):
            ensemble_probs += pred['probabilities'] * weight
            total_weight += weight

        # Normalize
        ensemble_probs = ensemble_probs / total_weight

        # Get final prediction
        ensemble_prediction = np.argmax(ensemble_probs)
        ensemble_confidence = np.max(ensemble_probs)

        return {
            'prediction': ensemble_prediction,
            'label': 'FAKE' if ensemble_prediction == 1 else 'REAL',
            'probabilities': ensemble_probs,
            'confidence': ensemble_confidence,
            'is_confident': ensemble_confidence >= self.confidence_threshold,
            'ensemble_weights': weights,
            'individual_predictions': predictions_list
        }

    def set_modality_dims(self, image_dim: int = None, video_dim: int = None, audio_dim: int = None):
        """
        Set feature dimensions for different modalities

        Args:
            image_dim: Image feature dimension
            video_dim: Video feature dimension
            audio_dim: Audio feature dimension
        """
        if image_dim is not None:
            self.image_dim = image_dim
        if video_dim is not None:
            self.video_dim = video_dim
        if audio_dim is not None:
            self.audio_dim = audio_dim

        # Rebuild projectors with new dimensions
        self.image_projector[0] = nn.Linear(self.image_dim, self.hidden_dim)
        self.video_projector[0] = nn.Linear(self.video_dim, self.hidden_dim)
        self.audio_projector[0] = nn.Linear(self.audio_dim, self.hidden_dim)

        logger.info(f"Updated modality dimensions: image={self.image_dim}, video={self.video_dim}, audio={self.audio_dim}")
