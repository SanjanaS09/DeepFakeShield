"""
Fusion Feature Extractor for Multi-Modal Deepfake Detection
Advanced cross-modal feature fusion and consistency analysis
Implements transformer-based attention and statistical fusion methods
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class FusionFeatureExtractor:
    """
    Advanced fusion feature extractor for multi-modal deepfake detection
    Combines visual, temporal, and audio features using multiple fusion strategies
    """

    def __init__(self, device: str = 'cpu', feature_dim: int = 256):
        """
        Initialize Fusion Feature Extractor

        Args:
            device: Device for computation ('cpu' or 'cuda')
            feature_dim: Dimension of feature embeddings
        """
        self.device = device
        self.feature_dim = feature_dim

        # Initialize fusion components
        self._init_attention_fusion()
        self._init_statistical_fusion()
        self._init_consistency_analysis()

        logger.info(f"Initialized FusionFeatureExtractor with {feature_dim}D features")

    def _init_attention_fusion(self):
        """Initialize transformer-based attention fusion"""

        # Multi-head attention parameters
        self.num_attention_heads = 8
        self.attention_dropout = 0.1

        # Cross-modal attention network
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )

        # Self-attention for each modality
        self.self_attention_visual = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )

        self.self_attention_audio = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )

        # Feature projection layers
        self.visual_projector = nn.Linear(self.feature_dim, self.feature_dim)
        self.audio_projector = nn.Linear(self.feature_dim, self.feature_dim)
        self.temporal_projector = nn.Linear(self.feature_dim, self.feature_dim)

        # Move to device
        self.cross_attention.to(self.device)
        self.self_attention_visual.to(self.device)
        self.self_attention_audio.to(self.device)
        self.visual_projector.to(self.device)
        self.audio_projector.to(self.device)
        self.temporal_projector.to(self.device)

    def _init_statistical_fusion(self):
        """Initialize statistical fusion methods"""

        # Fusion weights for different strategies
        self.fusion_weights = {
            'visual': 0.4,
            'temporal': 0.35,
            'audio': 0.25
        }

        # Consistency thresholds
        self.consistency_thresholds = {
            'visual_temporal': 0.7,
            'visual_audio': 0.6,
            'temporal_audio': 0.65,
            'overall': 0.65
        }

    def _init_consistency_analysis(self):
        """Initialize cross-modal consistency analysis"""

        # Synchronization analysis parameters
        self.sync_window_size = 5
        self.sync_overlap = 2

        # Correlation analysis parameters
        self.correlation_methods = ['pearson', 'spearman', 'kendall']

        # Divergence measures
        self.divergence_methods = ['kl', 'js', 'wasserstein']

    def compute_attention_fusion(self, 
                                visual_features: torch.Tensor,
                                temporal_features: Optional[torch.Tensor] = None,
                                audio_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute attention-based fusion of multi-modal features

        Args:
            visual_features: Visual features [batch, seq_len, feature_dim]
            temporal_features: Temporal features (optional)
            audio_features: Audio features (optional)

        Returns:
            Dictionary of fused features and attention weights
        """
        fusion_results = {}

        try:
            # Ensure features are 3D (batch, seq, features)
            if visual_features.dim() == 2:
                visual_features = visual_features.unsqueeze(1)  # Add sequence dimension

            # Project features to common space
            visual_proj = self.visual_projector(visual_features)

            available_modalities = ['visual']
            all_features = [visual_proj]
            modality_names = ['visual']

            if temporal_features is not None:
                if temporal_features.dim() == 2:
                    temporal_features = temporal_features.unsqueeze(1)
                temporal_proj = self.temporal_projector(temporal_features)
                all_features.append(temporal_proj)
                modality_names.append('temporal')
                available_modalities.append('temporal')

            if audio_features is not None:
                if audio_features.dim() == 2:
                    audio_features = audio_features.unsqueeze(1)
                audio_proj = self.audio_projector(audio_features)
                all_features.append(audio_proj)
                modality_names.append('audio')
                available_modalities.append('audio')

            # Self-attention for individual modalities
            with torch.no_grad():
                # Visual self-attention
                visual_self_att, visual_att_weights = self.self_attention_visual(
                    visual_proj, visual_proj, visual_proj
                )
                fusion_results['visual_self_attended'] = visual_self_att
                fusion_results['visual_attention_weights'] = visual_att_weights

                # Audio self-attention if available
                if audio_features is not None:
                    audio_self_att, audio_att_weights = self.self_attention_audio(
                        audio_proj, audio_proj, audio_proj
                    )
                    fusion_results['audio_self_attended'] = audio_self_att
                    fusion_results['audio_attention_weights'] = audio_att_weights

                # Cross-modal attention
                if len(all_features) > 1:
                    # Use visual as query, others as key and value
                    concatenated_kv = torch.cat(all_features, dim=1)  # Concatenate along sequence dimension

                    cross_attended, cross_att_weights = self.cross_attention(
                        visual_proj, concatenated_kv, concatenated_kv
                    )

                    fusion_results['cross_modal_attended'] = cross_attended
                    fusion_results['cross_attention_weights'] = cross_att_weights

                    # Weighted combination
                    weighted_features = []
                    start_idx = 0

                    for i, (modality, features) in enumerate(zip(modality_names, all_features)):
                        seq_len = features.shape[1]
                        modality_weights = cross_att_weights[:, :, start_idx:start_idx + seq_len]
                        weighted_feat = torch.sum(modality_weights.unsqueeze(-1) * features, dim=1)
                        weighted_features.append(weighted_feat)
                        start_idx += seq_len

                    # Final fusion
                    if len(weighted_features) > 1:
                        final_fusion = torch.stack(weighted_features, dim=1)  # [batch, modalities, features]
                        final_fusion = torch.mean(final_fusion, dim=1)  # Average across modalities
                        fusion_results['final_fusion'] = final_fusion
                    else:
                        fusion_results['final_fusion'] = weighted_features[0]
                else:
                    fusion_results['final_fusion'] = visual_self_att.squeeze(1)

            fusion_results['available_modalities'] = available_modalities

        except Exception as e:
            logger.warning(f"Attention fusion failed: {e}")
            # Return simple concatenation as fallback
            if temporal_features is not None and audio_features is not None:
                fusion_results['final_fusion'] = torch.cat([
                    visual_features.mean(dim=1) if visual_features.dim() > 2 else visual_features,
                    temporal_features.mean(dim=1) if temporal_features.dim() > 2 else temporal_features,
                    audio_features.mean(dim=1) if audio_features.dim() > 2 else audio_features
                ], dim=-1)
            elif temporal_features is not None:
                fusion_results['final_fusion'] = torch.cat([
                    visual_features.mean(dim=1) if visual_features.dim() > 2 else visual_features,
                    temporal_features.mean(dim=1) if temporal_features.dim() > 2 else temporal_features
                ], dim=-1)
            elif audio_features is not None:
                fusion_results['final_fusion'] = torch.cat([
                    visual_features.mean(dim=1) if visual_features.dim() > 2 else visual_features,
                    audio_features.mean(dim=1) if audio_features.dim() > 2 else audio_features
                ], dim=-1)
            else:
                fusion_results['final_fusion'] = visual_features.mean(dim=1) if visual_features.dim() > 2 else visual_features

        return fusion_results

    def compute_statistical_fusion(self, features_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """
        Compute statistical fusion of multi-modal features

        Args:
            features_dict: Dictionary of modality features

        Returns:
            Dictionary of statistical fusion results
        """
        fusion_results = {}

        try:
            # Convert tensors to numpy if needed
            numpy_features = {}
            for modality, features in features_dict.items():
                if isinstance(features, torch.Tensor):
                    numpy_features[modality] = features.detach().cpu().numpy()
                else:
                    numpy_features[modality] = features

            # Flatten features for statistical analysis
            flattened_features = {}
            for modality, features in numpy_features.items():
                if features.ndim > 1:
                    flattened_features[modality] = features.flatten()
                else:
                    flattened_features[modality] = features

            available_modalities = list(flattened_features.keys())

            # Compute pairwise correlations
            correlations = {}
            for i, mod1 in enumerate(available_modalities):
                for j, mod2 in enumerate(available_modalities):
                    if i < j:  # Avoid duplicate pairs
                        key = f"{mod1}_{mod2}"

                        # Ensure same length for correlation
                        min_len = min(len(flattened_features[mod1]), len(flattened_features[mod2]))
                        feat1 = flattened_features[mod1][:min_len]
                        feat2 = flattened_features[mod2][:min_len]

                        # Compute different correlation measures
                        for method in self.correlation_methods:
                            try:
                                if method == 'pearson':
                                    corr, p_value = stats.pearsonr(feat1, feat2)
                                elif method == 'spearman':
                                    corr, p_value = stats.spearmanr(feat1, feat2)
                                elif method == 'kendall':
                                    corr, p_value = stats.kendalltau(feat1, feat2)

                                correlations[f"{key}_{method}"] = {
                                    'correlation': float(corr) if not np.isnan(corr) else 0.0,
                                    'p_value': float(p_value) if not np.isnan(p_value) else 1.0
                                }
                            except Exception as e:
                                correlations[f"{key}_{method}"] = {
                                    'correlation': 0.0,
                                    'p_value': 1.0
                                }

            fusion_results['correlations'] = correlations

            # Weighted fusion based on modality reliability
            feature_means = {}
            feature_stds = {}
            feature_weights = {}

            for modality, features in flattened_features.items():
                feature_means[modality] = np.mean(features)
                feature_stds[modality] = np.std(features)

                # Weight based on inverse of variance (more stable = higher weight)
                weight = 1.0 / (feature_stds[modality] + 1e-8)
                feature_weights[modality] = weight

            # Normalize weights
            total_weight = sum(feature_weights.values())
            if total_weight > 0:
                feature_weights = {k: v / total_weight for k, v in feature_weights.items()}

            fusion_results['feature_statistics'] = {
                'means': feature_means,
                'stds': feature_stds,
                'weights': feature_weights
            }

            # Compute weighted fusion score
            weighted_score = sum(
                feature_means[modality] * feature_weights[modality] 
                for modality in available_modalities
            )
            fusion_results['weighted_fusion_score'] = float(weighted_score)

            # Consensus analysis
            # Check how much modalities agree
            if len(available_modalities) > 1:
                # Normalize features to [0, 1] for comparison
                normalized_features = {}
                for modality, features in flattened_features.items():
                    feat_min, feat_max = np.min(features), np.max(features)
                    if feat_max > feat_min:
                        normalized_features[modality] = (features - feat_min) / (feat_max - feat_min)
                    else:
                        normalized_features[modality] = np.ones_like(features) * 0.5

                # Compute pairwise agreements
                agreements = []
                for i, mod1 in enumerate(available_modalities):
                    for j, mod2 in enumerate(available_modalities):
                        if i < j:
                            feat1 = normalized_features[mod1]
                            feat2 = normalized_features[mod2]
                            min_len = min(len(feat1), len(feat2))

                            # Mean absolute difference (lower = better agreement)
                            agreement = 1.0 - np.mean(np.abs(feat1[:min_len] - feat2[:min_len]))
                            agreements.append(agreement)

                fusion_results['consensus_score'] = float(np.mean(agreements)) if agreements else 0.5
            else:
                fusion_results['consensus_score'] = 1.0  # Single modality = perfect consensus

        except Exception as e:
            logger.warning(f"Statistical fusion failed: {e}")
            fusion_results = {
                'correlations': {},
                'feature_statistics': {'means': {}, 'stds': {}, 'weights': {}},
                'weighted_fusion_score': 0.5,
                'consensus_score': 0.5
            }

        return fusion_results

    def analyze_cross_modal_consistency(self, features_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze consistency between different modalities

        Args:
            features_dict: Dictionary of modality features

        Returns:
            Dictionary of consistency measures
        """
        consistency_results = {}

        try:
            available_modalities = list(features_dict.keys())

            # Temporal alignment analysis (for video-audio synchronization)
            if 'temporal' in features_dict and 'audio' in features_dict:
                sync_score = self._compute_av_synchronization(
                    features_dict['temporal'], 
                    features_dict['audio']
                )
                consistency_results['audio_visual_sync'] = sync_score

            # Cross-modal prediction consistency
            # Check if features from different modalities lead to similar predictions
            if len(available_modalities) > 1:
                # Simplified consistency based on feature similarity
                modality_predictions = {}

                for modality, features in features_dict.items():
                    # Simple prediction based on feature magnitude
                    if isinstance(features, torch.Tensor):
                        feat_array = features.detach().cpu().numpy()
                    else:
                        feat_array = np.array(features) if not isinstance(features, np.ndarray) else features

                    # Flatten and compute simple score
                    if feat_array.ndim > 1:
                        feat_array = feat_array.flatten()

                    # Normalize to [0, 1]
                    if len(feat_array) > 0:
                        feat_min, feat_max = np.min(feat_array), np.max(feat_array)
                        if feat_max > feat_min:
                            normalized = (feat_array - feat_min) / (feat_max - feat_min)
                        else:
                            normalized = np.ones_like(feat_array) * 0.5

                        # Simple prediction score (mean of normalized features)
                        prediction = np.mean(normalized)
                        modality_predictions[modality] = prediction
                    else:
                        modality_predictions[modality] = 0.5

                # Compute prediction consistency
                if len(modality_predictions) > 1:
                    pred_values = list(modality_predictions.values())
                    pred_consistency = 1.0 - np.std(pred_values)  # Lower std = higher consistency
                    consistency_results['prediction_consistency'] = float(max(0, pred_consistency))

                    # Individual modality consistencies
                    for i, mod1 in enumerate(available_modalities):
                        for j, mod2 in enumerate(available_modalities):
                            if i < j:
                                consistency = 1.0 - abs(modality_predictions[mod1] - modality_predictions[mod2])
                                consistency_results[f'{mod1}_{mod2}_consistency'] = float(consistency)

            # Feature distribution consistency
            # Compare feature distributions across modalities
            if len(available_modalities) > 1:
                distribution_similarities = []

                for i, mod1 in enumerate(available_modalities):
                    for j, mod2 in enumerate(available_modalities):
                        if i < j:
                            sim = self._compute_distribution_similarity(
                                features_dict[mod1], 
                                features_dict[mod2]
                            )
                            distribution_similarities.append(sim)
                            consistency_results[f'{mod1}_{mod2}_distribution_similarity'] = sim

                if distribution_similarities:
                    consistency_results['overall_distribution_consistency'] = float(np.mean(distribution_similarities))

            # Confidence-based consistency
            # If confidence scores are available, check their consistency
            confidence_scores = {}
            for modality, features in features_dict.items():
                if isinstance(features, dict) and 'confidence' in features:
                    confidence_scores[modality] = features['confidence']
                elif hasattr(features, 'confidence'):
                    confidence_scores[modality] = features.confidence

            if len(confidence_scores) > 1:
                conf_values = list(confidence_scores.values())
                conf_consistency = 1.0 - np.std(conf_values) / (np.mean(conf_values) + 1e-8)
                consistency_results['confidence_consistency'] = float(max(0, min(conf_consistency, 1)))

        except Exception as e:
            logger.warning(f"Cross-modal consistency analysis failed: {e}")
            consistency_results = {
                'prediction_consistency': 0.5,
                'overall_distribution_consistency': 0.5
            }

        return consistency_results

    def _compute_av_synchronization(self, temporal_features: Any, audio_features: Any) -> float:
        """
        Compute audio-visual synchronization score

        Args:
            temporal_features: Temporal visual features
            audio_features: Audio features

        Returns:
            Synchronization score (0-1)
        """
        try:
            # Convert to numpy arrays
            if isinstance(temporal_features, torch.Tensor):
                temporal_array = temporal_features.detach().cpu().numpy()
            else:
                temporal_array = np.array(temporal_features) if not isinstance(temporal_features, np.ndarray) else temporal_features

            if isinstance(audio_features, torch.Tensor):
                audio_array = audio_features.detach().cpu().numpy()
            else:
                audio_array = np.array(audio_features) if not isinstance(audio_features, np.ndarray) else audio_features

            # Flatten if multi-dimensional
            if temporal_array.ndim > 1:
                temporal_array = temporal_array.flatten()
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

            # Ensure same length
            min_len = min(len(temporal_array), len(audio_array))
            if min_len < 10:  # Too short for meaningful analysis
                return 0.5

            temporal_array = temporal_array[:min_len]
            audio_array = audio_array[:min_len]

            # Cross-correlation for synchronization
            correlation = np.correlate(temporal_array, audio_array, mode='full')

            # Find the peak correlation
            peak_idx = np.argmax(correlation)
            max_correlation = correlation[peak_idx]

            # Normalize correlation
            norm_factor = np.sqrt(np.sum(temporal_array**2) * np.sum(audio_array**2))
            if norm_factor > 0:
                sync_score = max_correlation / norm_factor
                return float(max(0, min(sync_score, 1)))
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"A-V synchronization computation failed: {e}")
            return 0.5

    def _compute_distribution_similarity(self, features1: Any, features2: Any) -> float:
        """
        Compute similarity between feature distributions

        Args:
            features1: First feature set
            features2: Second feature set

        Returns:
            Distribution similarity score (0-1)
        """
        try:
            # Convert to numpy
            if isinstance(features1, torch.Tensor):
                array1 = features1.detach().cpu().numpy()
            else:
                array1 = np.array(features1) if not isinstance(features1, np.ndarray) else features1

            if isinstance(features2, torch.Tensor):
                array2 = features2.detach().cpu().numpy()
            else:
                array2 = np.array(features2) if not isinstance(features2, np.ndarray) else features2

            # Flatten
            if array1.ndim > 1:
                array1 = array1.flatten()
            if array2.ndim > 1:
                array2 = array2.flatten()

            # Compute histograms
            bins = 50
            hist1, bin_edges = np.histogram(array1, bins=bins, density=True)
            hist2, _ = np.histogram(array2, bins=bin_edges, density=True)

            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-8)
            hist2 = hist2 / (np.sum(hist2) + 1e-8)

            # Compute similarity using multiple measures
            similarities = []

            # Cosine similarity
            cos_sim = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-8)
            similarities.append(cos_sim)

            # Correlation
            if np.std(hist1) > 0 and np.std(hist2) > 0:
                corr = np.corrcoef(hist1, hist2)[0, 1]
                if not np.isnan(corr):
                    similarities.append(corr)

            # Inverse of KL divergence (Jensen-Shannon divergence)
            # Add small epsilon to avoid division by zero
            hist1_smooth = hist1 + 1e-10
            hist2_smooth = hist2 + 1e-10

            # Jensen-Shannon divergence
            m = 0.5 * (hist1_smooth + hist2_smooth)
            js_div = 0.5 * stats.entropy(hist1_smooth, m) + 0.5 * stats.entropy(hist2_smooth, m)
            js_similarity = 1.0 / (1.0 + js_div)
            similarities.append(js_similarity)

            # Return average similarity
            return float(np.mean(similarities)) if similarities else 0.5

        except Exception as e:
            logger.warning(f"Distribution similarity computation failed: {e}")
            return 0.5

    def extract_fusion_features(self, 
                               visual_features: Optional[torch.Tensor] = None,
                               temporal_features: Optional[torch.Tensor] = None,
                               audio_features: Optional[torch.Tensor] = None,
                               feature_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract comprehensive fusion features

        Args:
            visual_features: Visual features tensor (optional)
            temporal_features: Temporal features tensor (optional)
            audio_features: Audio features tensor (optional)
            feature_dict: Dictionary of pre-computed features (optional)

        Returns:
            Dictionary of fusion features and analysis
        """
        fusion_results = {}

        try:
            # Prepare feature dictionary
            if feature_dict is None:
                feature_dict = {}
                if visual_features is not None:
                    feature_dict['visual'] = visual_features
                if temporal_features is not None:
                    feature_dict['temporal'] = temporal_features
                if audio_features is not None:
                    feature_dict['audio'] = audio_features

            # Attention-based fusion
            if visual_features is not None:
                attention_results = self.compute_attention_fusion(
                    visual_features, temporal_features, audio_features
                )
                fusion_results['attention_fusion'] = attention_results

            # Statistical fusion
            statistical_results = self.compute_statistical_fusion(feature_dict)
            fusion_results['statistical_fusion'] = statistical_results

            # Cross-modal consistency
            consistency_results = self.analyze_cross_modal_consistency(feature_dict)
            fusion_results['consistency_analysis'] = consistency_results

            # Overall fusion score
            fusion_scores = []

            # Weight attention fusion score
            if 'attention_fusion' in fusion_results and 'final_fusion' in fusion_results['attention_fusion']:
                attention_score = torch.mean(fusion_results['attention_fusion']['final_fusion']).item()
                fusion_scores.append(attention_score)

            # Weight statistical fusion score
            if 'statistical_fusion' in fusion_results:
                stat_score = fusion_results['statistical_fusion']['weighted_fusion_score']
                fusion_scores.append(stat_score)

            # Weight consistency score
            if 'consistency_analysis' in fusion_results:
                consistency = fusion_results['consistency_analysis'].get('prediction_consistency', 0.5)
                fusion_scores.append(consistency)

            # Compute overall score
            if fusion_scores:
                overall_score = np.mean(fusion_scores)
                fusion_results['overall_fusion_score'] = float(overall_score)
            else:
                fusion_results['overall_fusion_score'] = 0.5

            # Modality contributions
            modality_contributions = {}
            available_modalities = list(feature_dict.keys())

            for modality in available_modalities:
                # Base contribution from statistical weights
                stat_weight = fusion_results['statistical_fusion']['feature_statistics']['weights'].get(modality, 1.0 / len(available_modalities))

                # Adjust based on consistency
                consistency_boost = 1.0
                for key, value in fusion_results['consistency_analysis'].items():
                    if modality in key and 'consistency' in key:
                        consistency_boost *= (1.0 + value * 0.5)  # Boost up to 50%

                final_contribution = stat_weight * consistency_boost
                modality_contributions[modality] = float(final_contribution)

            # Normalize contributions
            total_contribution = sum(modality_contributions.values())
            if total_contribution > 0:
                modality_contributions = {k: v / total_contribution for k, v in modality_contributions.items()}

            fusion_results['modality_contributions'] = modality_contributions

            # Summary statistics
            fusion_results['summary'] = {
                'num_modalities': len(available_modalities),
                'dominant_modality': max(modality_contributions.items(), key=lambda x: x[1])[0] if modality_contributions else 'none',
                'modality_balance': 1.0 - np.std(list(modality_contributions.values())) if modality_contributions else 0.0,
                'fusion_confidence': float(np.mean([
                    fusion_results.get('overall_fusion_score', 0.5),
                    fusion_results['consistency_analysis'].get('prediction_consistency', 0.5),
                    fusion_results['statistical_fusion'].get('consensus_score', 0.5)
                ]))
            }

        except Exception as e:
            logger.error(f"Fusion feature extraction failed: {e}")
            # Return minimal fusion results
            fusion_results = {
                'overall_fusion_score': 0.5,
                'modality_contributions': {'visual': 0.4, 'temporal': 0.35, 'audio': 0.25},
                'summary': {
                    'num_modalities': 1,
                    'dominant_modality': 'visual',
                    'modality_balance': 0.0,
                    'fusion_confidence': 0.5
                }
            }

        return fusion_results

    def compute_fusion_quality_score(self, fusion_results: Dict[str, Any]) -> float:
        """
        Compute overall fusion quality score

        Args:
            fusion_results: Fusion analysis results

        Returns:
            Quality score (0-1, higher = better quality)
        """
        try:
            quality_indicators = []

            # Overall fusion score
            if 'overall_fusion_score' in fusion_results:
                quality_indicators.append(fusion_results['overall_fusion_score'])

            # Consistency scores
            if 'consistency_analysis' in fusion_results:
                consistency = fusion_results['consistency_analysis']
                if 'prediction_consistency' in consistency:
                    quality_indicators.append(consistency['prediction_consistency'])
                if 'overall_distribution_consistency' in consistency:
                    quality_indicators.append(consistency['overall_distribution_consistency'])

            # Statistical fusion quality
            if 'statistical_fusion' in fusion_results:
                consensus = fusion_results['statistical_fusion'].get('consensus_score', 0.5)
                quality_indicators.append(consensus)

            # Modality balance (more balanced = better quality)
            if 'summary' in fusion_results:
                balance = fusion_results['summary'].get('modality_balance', 0.0)
                quality_indicators.append(balance)

            if quality_indicators:
                return float(np.mean(quality_indicators))
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"Fusion quality score computation failed: {e}")
            return 0.5
