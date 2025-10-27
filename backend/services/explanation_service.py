"""
Explanation Service for Multi-Modal Deepfake Detection System
Provides explainable AI (XAI) capabilities for deepfake detection results
Generates human-readable explanations, visualizations, and evidence for predictions
"""

import numpy as np
import cv2
import torch
import logging
import time
import json
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from utils.face_detection import FaceDetector
from utils.logger import get_logger
from features.visual_features import VisualFeatureExtractor
from features.temporal_features import TemporalFeatureExtractor
from features.audio_features import AudioFeatureExtractor

logger = get_logger(__name__)

class ExplanationType(Enum):
    """Types of explanations available"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    VISUAL = "visual"
    STATISTICAL = "statistical"
    COMPARATIVE = "comparative"
    TECHNICAL = "technical"

class EvidenceType(Enum):
    """Types of evidence for explanations"""
    VISUAL_ARTIFACTS = "visual_artifacts"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    AUDIO_ANOMALIES = "audio_anomalies"
    FACIAL_MANIPULATION = "facial_manipulation"
    QUALITY_DEGRADATION = "quality_degradation"
    STATISTICAL_OUTLIERS = "statistical_outliers"

@dataclass
class ExplanationRequest:
    """Request object for explanation generation"""
    detection_result: Dict[str, Any]
    features: Dict[str, Any]
    media_data: Any
    media_type: str
    explanation_types: List[ExplanationType]
    confidence_threshold: float = 0.5
    generate_visualizations: bool = True
    include_evidence: bool = True
    target_audience: str = "general"  # general, technical, expert
    request_id: str = None

@dataclass
class ExplanationResult:
    """Result object for explanations"""
    request_id: str
    decision: str  # REAL, DEEPFAKE, UNCERTAIN
    confidence: float
    summary: str
    detailed_explanation: Dict[str, Any] = None
    evidence: List[Dict[str, Any]] = None
    visualizations: List[str] = None
    statistical_analysis: Dict[str, Any] = None
    recommendations: List[str] = None
    limitations: List[str] = None
    processing_time: float = 0.0
    error_message: str = None

class ExplanationService:
    """
    Service for generating explainable AI outputs for deepfake detection
    Provides various types of explanations tailored to different audiences
    """

    def __init__(self, 
                 output_dir: str = 'explanations',
                 enable_visualizations: bool = True,
                 device: str = 'cpu'):
        """
        Initialize Explanation Service

        Args:
            output_dir: Directory for saving explanation outputs
            enable_visualizations: Enable generation of visualization outputs
            device: Device for computation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_visualizations = enable_visualizations
        self.device = device

        # Initialize components
        self.face_detector = FaceDetector()

        # Explanation templates and thresholds
        self._init_explanation_templates()
        self._init_evidence_thresholds()

        # Statistics
        self.explanation_stats = {
            'total_explanations': 0,
            'explanation_types': {},
            'average_processing_time': 0.0
        }

        logger.info("Initialized ExplanationService")

    def _init_explanation_templates(self):
        """Initialize explanation templates for different audiences"""
        self.templates = {
            'general': {
                'deepfake_detected': [
                    "This content appears to be artificially generated or manipulated.",
                    "Multiple indicators suggest this is not authentic content.",
                    "The analysis shows characteristics typical of deepfake content."
                ],
                'real_content': [
                    "This content appears to be authentic and unmanipulated.",
                    "No significant indicators of artificial manipulation were found.",
                    "The analysis suggests this is genuine content."
                ],
                'uncertain': [
                    "The analysis results are inconclusive.",
                    "There are mixed indicators that make classification difficult.",
                    "Additional analysis may be needed for a definitive assessment."
                ]
            },
            'technical': {
                'deepfake_detected': [
                    "Multiple technical indicators suggest artificial content generation.",
                    "Statistical analysis reveals patterns inconsistent with natural media.",
                    "Feature analysis indicates probable synthetic content creation."
                ],
                'real_content': [
                    "Technical analysis indicates authentic content characteristics.",
                    "Feature patterns are consistent with natural media generation.",
                    "No significant technical artifacts suggesting manipulation."
                ]
            },
            'expert': {
                'deepfake_detected': [
                    "Comprehensive feature analysis reveals multiple manipulation signatures.",
                    "Statistical and spectral analysis indicates synthetic content generation.",
                    "Cross-modal consistency analysis suggests coordinated manipulation."
                ],
                'real_content': [
                    "Multi-modal analysis confirms authentic content characteristics.",
                    "Feature distributions align with natural content generation patterns.",
                    "No significant manipulation artifacts detected across modalities."
                ]
            }
        }

    def _init_evidence_thresholds(self):
        """Initialize thresholds for different types of evidence"""
        self.evidence_thresholds = {
            'visual_artifacts': {
                'blocking_artifacts': 0.3,
                'ringing_artifacts': 0.25,
                'checkerboard_artifacts': 0.2,
                'upsampling_artifacts': 0.3
            },
            'temporal_inconsistency': {
                'temporal_consistency': 0.7,
                'motion_smoothness': 0.6,
                'flickering_score': 0.8
            },
            'audio_anomalies': {
                'voice_quality_score': 0.6,
                'synthesis_artifact_score': 0.3,
                'spectral_artifacts': 0.25
            },
            'facial_manipulation': {
                'landmark_quality': 0.7,
                'face_quality_score': 0.6,
                'expression_consistency': 0.7
            },
            'quality_degradation': {
                'overall_image_quality': 0.5,
                'overall_video_quality': 0.5,
                'overall_audio_quality': 0.6
            }
        }

    async def generate_explanation(self, request: ExplanationRequest) -> ExplanationResult:
        """
        Generate comprehensive explanation for detection result

        Args:
            request: Explanation request object

        Returns:
            Explanation result object
        """
        start_time = time.time()
        request_id = request.request_id or f"exp_{int(time.time()*1000)}"

        logger.info(f"Generating explanation for request {request_id}", extra={
            'request_id': request_id,
            'explanation_types': [et.value for et in request.explanation_types]
        })

        try:
            # Extract key information from detection result
            decision_info = self._extract_decision_info(request.detection_result)

            # Generate summary explanation
            summary = await self._generate_summary(
                decision_info, request.target_audience
            )

            # Initialize result
            result = ExplanationResult(
                request_id=request_id,
                decision=decision_info['decision'],
                confidence=decision_info['confidence'],
                summary=summary,
                evidence=[],
                visualizations=[],
                recommendations=[]
            )

            # Generate requested explanation types
            for explanation_type in request.explanation_types:
                try:
                    if explanation_type == ExplanationType.DETAILED:
                        result.detailed_explanation = await self._generate_detailed_explanation(
                            request, decision_info
                        )

                    elif explanation_type == ExplanationType.VISUAL and self.enable_visualizations:
                        if request.generate_visualizations:
                            visualizations = await self._generate_visual_explanations(
                                request, decision_info
                            )
                            result.visualizations.extend(visualizations)

                    elif explanation_type == ExplanationType.STATISTICAL:
                        result.statistical_analysis = await self._generate_statistical_explanation(
                            request, decision_info
                        )

                    elif explanation_type == ExplanationType.COMPARATIVE:
                        comparative_analysis = await self._generate_comparative_explanation(
                            request, decision_info
                        )
                        if result.detailed_explanation is None:
                            result.detailed_explanation = {}
                        result.detailed_explanation['comparative_analysis'] = comparative_analysis

                    elif explanation_type == ExplanationType.TECHNICAL:
                        technical_details = await self._generate_technical_explanation(
                            request, decision_info
                        )
                        if result.detailed_explanation is None:
                            result.detailed_explanation = {}
                        result.detailed_explanation['technical_details'] = technical_details

                except Exception as e:
                    logger.warning(f"Failed to generate {explanation_type.value} explanation: {e}")

            # Generate evidence if requested
            if request.include_evidence:
                result.evidence = await self._generate_evidence(request, decision_info)

            # Generate recommendations
            result.recommendations = await self._generate_recommendations(
                request, decision_info
            )

            # Add limitations
            result.limitations = self._get_limitations(request.media_type, decision_info)

            # Calculate processing time
            result.processing_time = time.time() - start_time

            # Update statistics
            self._update_stats(request, result)

            logger.info(f"Explanation generated for request {request_id}", extra={
                'request_id': request_id,
                'processing_time': result.processing_time,
                'decision': result.decision
            })

            return result

        except Exception as e:
            logger.error(f"Explanation generation failed for request {request_id}: {e}", exc_info=True)
            return ExplanationResult(
                request_id=request_id,
                decision="UNKNOWN",
                confidence=0.0,
                summary="Explanation generation failed",
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    def _extract_decision_info(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key decision information from detection result"""
        try:
            is_deepfake = detection_result.get('is_deepfake', False)
            confidence = detection_result.get('confidence_score', 0.0)

            # Determine decision category
            if confidence < 0.3:
                decision = "UNCERTAIN"
                category = "uncertain"
            elif is_deepfake:
                decision = "DEEPFAKE"
                category = "deepfake_detected"
            else:
                decision = "REAL"
                category = "real_content"

            return {
                'decision': decision,
                'category': category,
                'confidence': confidence,
                'is_deepfake': is_deepfake,
                'model_results': detection_result.get('model_results', {}),
                'processing_time': detection_result.get('processing_time', 0.0)
            }

        except Exception as e:
            logger.error(f"Decision info extraction failed: {e}")
            return {
                'decision': 'UNKNOWN',
                'category': 'uncertain',
                'confidence': 0.0,
                'is_deepfake': False,
                'model_results': {},
                'processing_time': 0.0
            }

    async def _generate_summary(self, decision_info: Dict[str, Any], 
                              target_audience: str) -> str:
        """Generate summary explanation"""
        try:
            category = decision_info['category']
            confidence = decision_info['confidence']

            # Get base template
            templates = self.templates.get(target_audience, self.templates['general'])
            base_messages = templates.get(category, ["Analysis completed."])

            # Select message based on confidence
            if len(base_messages) > 1:
                if confidence < 0.4:
                    message = base_messages[-1]  # Least confident
                elif confidence > 0.8:
                    message = base_messages[0]  # Most confident
                else:
                    message = base_messages[1] if len(base_messages) > 2 else base_messages[0]
            else:
                message = base_messages[0]

            # Add confidence information
            confidence_phrase = self._get_confidence_phrase(confidence, target_audience)

            return f"{message} {confidence_phrase}"

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Analysis completed with inconclusive results."

    def _get_confidence_phrase(self, confidence: float, target_audience: str) -> str:
        """Generate confidence phrase based on audience"""
        if target_audience == 'technical' or target_audience == 'expert':
            return f"(Confidence: {confidence:.3f})"
        else:
            if confidence > 0.9:
                return "(Very high confidence)"
            elif confidence > 0.7:
                return "(High confidence)"
            elif confidence > 0.5:
                return "(Moderate confidence)"
            elif confidence > 0.3:
                return "(Low confidence)"
            else:
                return "(Very low confidence)"

    async def _generate_detailed_explanation(self, request: ExplanationRequest,
                                           decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed explanation"""
        try:
            detailed_explanation = {
                'decision_rationale': self._explain_decision_rationale(
                    decision_info, request.features
                ),
                'model_analysis': self._analyze_model_contributions(
                    decision_info['model_results']
                ),
                'feature_importance': self._analyze_feature_importance(
                    request.features
                ),
                'quality_assessment': self._assess_content_quality(
                    request.features
                ),
                'consistency_analysis': self._analyze_consistency(
                    request.features, request.media_type
                )
            }

            # Add modality-specific analysis
            if request.media_type == 'image':
                detailed_explanation['visual_analysis'] = self._analyze_visual_features(
                    request.features.get('visual', {})
                )
            elif request.media_type == 'video':
                detailed_explanation['temporal_analysis'] = self._analyze_temporal_features(
                    request.features.get('temporal', {})
                )
            elif request.media_type == 'audio':
                detailed_explanation['audio_analysis'] = self._analyze_audio_features(
                    request.features.get('audio', {})
                )
            elif request.media_type == 'multimodal':
                detailed_explanation['fusion_analysis'] = self._analyze_fusion_features(
                    request.features.get('fusion', {})
                )

            return detailed_explanation

        except Exception as e:
            logger.error(f"Detailed explanation generation failed: {e}")
            return {'error': str(e)}

    def _explain_decision_rationale(self, decision_info: Dict[str, Any], 
                                  features: Dict[str, Any]) -> Dict[str, Any]:
        """Explain the rationale behind the decision"""
        try:
            rationale = {
                'primary_factors': [],
                'supporting_evidence': [],
                'conflicting_indicators': []
            }

            confidence = decision_info['confidence']
            is_deepfake = decision_info['is_deepfake']

            # Analyze primary factors
            if is_deepfake:
                if confidence > 0.7:
                    rationale['primary_factors'].append(
                        "Multiple strong indicators of artificial content generation"
                    )
                else:
                    rationale['primary_factors'].append(
                        "Several indicators suggest possible manipulation"
                    )
            else:
                if confidence > 0.7:
                    rationale['primary_factors'].append(
                        "Strong indicators of authentic content"
                    )
                else:
                    rationale['primary_factors'].append(
                        "Content appears to be authentic with some uncertainty"
                    )

            # Analyze supporting evidence from features
            for modality, modal_features in features.items():
                if isinstance(modal_features, dict):
                    evidence = self._extract_evidence_from_features(modality, modal_features)
                    rationale['supporting_evidence'].extend(evidence)

            # Identify conflicting indicators
            if confidence < 0.6:
                rationale['conflicting_indicators'].append(
                    "Mixed signals from different analysis components"
                )

            return rationale

        except Exception as e:
            logger.error(f"Decision rationale explanation failed: {e}")
            return {'error': str(e)}

    def _extract_evidence_from_features(self, modality: str, 
                                      features: Dict[str, Any]) -> List[str]:
        """Extract evidence from features"""
        evidence = []

        try:
            if modality == 'visual':
                # Visual evidence
                quality = features.get('overall_visual_quality', 0.5)
                if quality < 0.4:
                    evidence.append("Poor visual quality detected")

                artifact_score = features.get('overall_artifact_score', 0.1)
                if artifact_score > 0.3:
                    evidence.append("High levels of compression artifacts")

            elif modality == 'temporal':
                # Temporal evidence
                consistency = features.get('temporal_consistency', 0.8)
                if consistency < 0.6:
                    evidence.append("Poor temporal consistency between frames")

                motion_smoothness = features.get('motion_smoothness', 0.7)
                if motion_smoothness < 0.5:
                    evidence.append("Unnatural motion patterns detected")

            elif modality == 'audio':
                # Audio evidence
                voice_quality = features.get('voice_quality_score', 0.7)
                if voice_quality < 0.5:
                    evidence.append("Unnatural voice characteristics")

                synthesis_artifacts = features.get('synthesis_artifact_score', 0.1)
                if synthesis_artifacts > 0.3:
                    evidence.append("Audio synthesis artifacts detected")

            elif modality == 'fusion':
                # Fusion evidence
                overall_score = features.get('overall_fusion_score', 0.5)
                if overall_score > 0.6:
                    evidence.append("Cross-modal inconsistencies detected")

        except Exception as e:
            logger.debug(f"Feature evidence extraction failed for {modality}: {e}")

        return evidence

    def _analyze_model_contributions(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual model contributions"""
        try:
            analysis = {
                'model_agreement': 0.0,
                'individual_models': {},
                'consensus': 'unknown'
            }

            if not model_results:
                return analysis

            # Analyze individual models
            predictions = []
            confidences = []

            for model_name, result in model_results.items():
                if 'error' not in result:
                    prediction = result.get('prediction', 0)
                    confidence = result.get('confidence', 0.5)

                    predictions.append(prediction)
                    confidences.append(confidence)

                    analysis['individual_models'][model_name] = {
                        'prediction': 'DEEPFAKE' if prediction == 1 else 'REAL',
                        'confidence': confidence,
                        'contribution': confidence / (sum(confidences) + 1e-10)
                    }

            # Calculate agreement
            if predictions:
                agreement = 1.0 - np.std(predictions)
                analysis['model_agreement'] = float(agreement)

                # Determine consensus
                avg_prediction = np.mean(predictions)
                if agreement > 0.8:
                    analysis['consensus'] = 'strong'
                elif agreement > 0.6:
                    analysis['consensus'] = 'moderate'
                else:
                    analysis['consensus'] = 'weak'

            return analysis

        except Exception as e:
            logger.error(f"Model contribution analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_feature_importance(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature importance for the decision"""
        try:
            importance_analysis = {
                'top_features': [],
                'feature_categories': {},
                'modality_contributions': {}
            }

            # Collect all numerical features with their modalities
            all_features = {}
            modality_features = {}

            for modality, modal_features in features.items():
                if isinstance(modal_features, dict):
                    modality_features[modality] = []
                    for key, value in modal_features.items():
                        if isinstance(value, (int, float)):
                            feature_name = f"{modality}_{key}"
                            all_features[feature_name] = abs(float(value) - 0.5)  # Distance from neutral
                            modality_features[modality].append((key, abs(float(value) - 0.5)))

            # Top features overall
            sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
            importance_analysis['top_features'] = [
                {'name': name, 'importance': importance}
                for name, importance in sorted_features[:10]
            ]

            # Modality contributions
            for modality, modal_feat_list in modality_features.items():
                if modal_feat_list:
                    avg_importance = np.mean([importance for _, importance in modal_feat_list])
                    importance_analysis['modality_contributions'][modality] = float(avg_importance)

            return importance_analysis

        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {'error': str(e)}

    def _assess_content_quality(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall content quality"""
        try:
            quality_assessment = {
                'overall_quality': 'unknown',
                'quality_factors': {},
                'quality_issues': []
            }

            quality_scores = []

            # Extract quality metrics from different modalities
            for modality, modal_features in features.items():
                if isinstance(modal_features, dict):
                    # Look for quality-related features
                    for key, value in modal_features.items():
                        if 'quality' in key.lower() and isinstance(value, (int, float)):
                            quality_scores.append(float(value))
                            quality_assessment['quality_factors'][f"{modality}_{key}"] = float(value)

                            # Identify quality issues
                            if value < 0.5:
                                quality_assessment['quality_issues'].append(
                                    f"Low {key.replace('_', ' ')} in {modality}"
                                )

            # Overall quality assessment
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                if avg_quality > 0.8:
                    quality_assessment['overall_quality'] = 'high'
                elif avg_quality > 0.6:
                    quality_assessment['overall_quality'] = 'good'
                elif avg_quality > 0.4:
                    quality_assessment['overall_quality'] = 'fair'
                else:
                    quality_assessment['overall_quality'] = 'poor'

            return quality_assessment

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {'error': str(e)}

    def _analyze_consistency(self, features: Dict[str, Any], 
                           media_type: str) -> Dict[str, Any]:
        """Analyze consistency across different aspects"""
        try:
            consistency_analysis = {
                'overall_consistency': 'unknown',
                'consistency_scores': {},
                'inconsistencies': []
            }

            consistency_scores = []

            # Look for consistency-related features
            for modality, modal_features in features.items():
                if isinstance(modal_features, dict):
                    for key, value in modal_features.items():
                        if 'consistency' in key.lower() and isinstance(value, (int, float)):
                            consistency_scores.append(float(value))
                            consistency_analysis['consistency_scores'][f"{modality}_{key}"] = float(value)

                            # Identify inconsistencies
                            if value < 0.6:
                                consistency_analysis['inconsistencies'].append(
                                    f"Poor {key.replace('_', ' ')} in {modality}"
                                )

            # Cross-modal consistency for multimodal content
            if media_type == 'multimodal' and 'fusion' in features:
                fusion_features = features['fusion']
                if isinstance(fusion_features, dict):
                    cross_modal_consistency = fusion_features.get('consistency_analysis', {})
                    if cross_modal_consistency:
                        consistency_analysis['cross_modal'] = cross_modal_consistency

            # Overall consistency assessment
            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
                if avg_consistency > 0.8:
                    consistency_analysis['overall_consistency'] = 'high'
                elif avg_consistency > 0.6:
                    consistency_analysis['overall_consistency'] = 'good'
                elif avg_consistency > 0.4:
                    consistency_analysis['overall_consistency'] = 'fair'
                else:
                    consistency_analysis['overall_consistency'] = 'poor'

            return consistency_analysis

        except Exception as e:
            logger.error(f"Consistency analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_visual_features(self, visual_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual-specific features"""
        try:
            analysis = {
                'face_analysis': {},
                'texture_analysis': {},
                'artifact_analysis': {},
                'quality_analysis': {}
            }

            # Face analysis
            if 'face_features' in visual_features:
                face_features = visual_features['face_features']
                analysis['face_analysis'] = {
                    'face_detected': 'landmarks' in face_features,
                    'face_quality': face_features.get('face_quality_score', 0.5),
                    'landmark_quality': face_features.get('landmark_quality', 0.5)
                }

            # Texture analysis
            texture_keys = ['texture_measures', 'lbp_features', 'glcm_features']
            for key in texture_keys:
                if key in visual_features:
                    analysis['texture_analysis'][key] = 'present'

            # Artifact analysis
            artifact_keys = [k for k in visual_features.keys() if 'artifact' in k.lower()]
            for key in artifact_keys:
                analysis['artifact_analysis'][key] = visual_features[key]

            # Quality analysis
            quality_keys = [k for k in visual_features.keys() if 'quality' in k.lower()]
            for key in quality_keys:
                analysis['quality_analysis'][key] = visual_features[key]

            return analysis

        except Exception as e:
            logger.error(f"Visual feature analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_temporal_features(self, temporal_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal-specific features"""
        try:
            analysis = {
                'motion_analysis': {},
                'consistency_analysis': {},
                'quality_analysis': {}
            }

            # Motion analysis
            motion_keys = [k for k in temporal_features.keys() if 'motion' in k.lower()]
            for key in motion_keys:
                analysis['motion_analysis'][key] = temporal_features[key]

            # Consistency analysis
            consistency_keys = [k for k in temporal_features.keys() if 'consistency' in k.lower()]
            for key in consistency_keys:
                analysis['consistency_analysis'][key] = temporal_features[key]

            # Quality analysis
            quality_keys = [k for k in temporal_features.keys() if 'quality' in k.lower()]
            for key in quality_keys:
                analysis['quality_analysis'][key] = temporal_features[key]

            return analysis

        except Exception as e:
            logger.error(f"Temporal feature analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_audio_features(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio-specific features"""
        try:
            analysis = {
                'voice_quality': {},
                'spectral_analysis': {},
                'synthesis_artifacts': {}
            }

            # Voice quality
            voice_keys = [k for k in audio_features.keys() if 'voice' in k.lower()]
            for key in voice_keys:
                analysis['voice_quality'][key] = audio_features[key]

            # Spectral analysis
            spectral_keys = [k for k in audio_features.keys() if 'spectral' in k.lower()]
            for key in spectral_keys:
                analysis['spectral_analysis'][key] = audio_features[key]

            # Synthesis artifacts
            artifact_keys = [k for k in audio_features.keys() if 'artifact' in k.lower() or 'synthesis' in k.lower()]
            for key in artifact_keys:
                analysis['synthesis_artifacts'][key] = audio_features[key]

            return analysis

        except Exception as e:
            logger.error(f"Audio feature analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_fusion_features(self, fusion_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fusion-specific features"""
        try:
            analysis = {
                'modality_contributions': {},
                'consistency_analysis': {},
                'fusion_confidence': {}
            }

            # Modality contributions
            if 'modality_contributions' in fusion_features:
                analysis['modality_contributions'] = fusion_features['modality_contributions']

            # Consistency analysis
            if 'consistency_analysis' in fusion_features:
                analysis['consistency_analysis'] = fusion_features['consistency_analysis']

            # Fusion confidence
            confidence_keys = [k for k in fusion_features.keys() if 'confidence' in k.lower() or 'score' in k.lower()]
            for key in confidence_keys:
                analysis['fusion_confidence'][key] = fusion_features[key]

            return analysis

        except Exception as e:
            logger.error(f"Fusion feature analysis failed: {e}")
            return {'error': str(e)}

    async def _generate_visual_explanations(self, request: ExplanationRequest,
                                          decision_info: Dict[str, Any]) -> List[str]:
        """Generate visual explanations and save as images"""
        try:
            visualizations = []

            # Only generate visualizations for visual media
            if request.media_type in ['image', 'video']:
                # Feature importance visualization
                importance_viz = await self._create_feature_importance_visualization(
                    request.features, request_id=request.request_id
                )
                if importance_viz:
                    visualizations.append(importance_viz)

                # Model confidence visualization
                confidence_viz = await self._create_confidence_visualization(
                    decision_info['model_results'], request_id=request.request_id
                )
                if confidence_viz:
                    visualizations.append(confidence_viz)

                # Face analysis visualization (if applicable)
                if request.media_type == 'image' and 'visual' in request.features:
                    face_viz = await self._create_face_analysis_visualization(
                        request.media_data, request.features['visual'], 
                        request_id=request.request_id
                    )
                    if face_viz:
                        visualizations.append(face_viz)

            return visualizations

        except Exception as e:
            logger.error(f"Visual explanation generation failed: {e}")
            return []

    async def _create_feature_importance_visualization(self, features: Dict[str, Any],
                                                     request_id: str) -> Optional[str]:
        """Create feature importance bar chart"""
        try:
            import matplotlib.pyplot as plt

            # Extract top features
            all_features = {}
            for modality, modal_features in features.items():
                if isinstance(modal_features, dict):
                    for key, value in modal_features.items():
                        if isinstance(value, (int, float)):
                            feature_name = f"{modality}_{key}"[:30]  # Truncate long names
                            all_features[feature_name] = abs(float(value) - 0.5)

            if not all_features:
                return None

            # Sort and take top 10
            sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:10]
            names, values = zip(*sorted_features) if sorted_features else ([], [])

            # Create visualization
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(names)), values, color='skyblue')
            plt.yticks(range(len(names)), names)
            plt.xlabel('Feature Importance')
            plt.title('Top Feature Importance for Detection Decision')
            plt.tight_layout()

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', ha='left', va='center')

            # Save visualization
            output_path = self.output_dir / f"feature_importance_{request_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Generated feature importance visualization: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Feature importance visualization failed: {e}")
            return None

    async def _create_confidence_visualization(self, model_results: Dict[str, Any],
                                             request_id: str) -> Optional[str]:
        """Create model confidence comparison chart"""
        try:
            import matplotlib.pyplot as plt

            if not model_results:
                return None

            # Extract model confidences
            models = []
            confidences = []
            predictions = []

            for model_name, result in model_results.items():
                if 'error' not in result:
                    models.append(model_name)
                    confidences.append(result.get('confidence', 0.5))
                    predictions.append('Deepfake' if result.get('prediction', 0) == 1 else 'Real')

            if not models:
                return None

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Confidence scores
            colors = ['red' if pred == 'Deepfake' else 'green' for pred in predictions]
            bars1 = ax1.bar(models, confidences, color=colors, alpha=0.7)
            ax1.set_ylabel('Confidence Score')
            ax1.set_title('Model Confidence Scores')
            ax1.set_ylim(0, 1)

            # Add value labels
            for bar, conf in zip(bars1, confidences):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{conf:.3f}', ha='center', va='bottom')

            # Predictions
            pred_colors = ['red' if pred == 'Deepfake' else 'green' for pred in predictions]
            bars2 = ax2.bar(models, [1]*len(models), color=pred_colors, alpha=0.7)
            ax2.set_ylabel('Prediction')
            ax2.set_title('Model Predictions')
            ax2.set_yticks([0.5])
            ax2.set_yticklabels([''])

            # Add prediction labels
            for bar, pred in zip(bars2, predictions):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        pred, ha='center', va='center', fontweight='bold', color='white')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save visualization
            output_path = self.output_dir / f"model_confidence_{request_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Generated model confidence visualization: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Confidence visualization failed: {e}")
            return None

    async def _create_face_analysis_visualization(self, image_data: np.ndarray,
                                                visual_features: Dict[str, Any],
                                                request_id: str) -> Optional[str]:
        """Create face analysis visualization with landmarks"""
        try:
            if image_data is None:
                return None

            # Detect faces
            faces = self.face_detector.detect_faces(image_data)
            if not faces:
                return None

            # Create visualization
            vis_image = image_data.copy()

            # Draw face bounding boxes and landmarks
            for i, face in enumerate(faces[:3]):  # Limit to 3 faces
                bbox = face['bbox']
                x, y, w, h = bbox

                # Draw bounding box
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Add confidence label
                confidence = face.get('confidence', 0.0)
                quality = face.get('quality_score', 0.0)
                label = f"Face {i+1}: Conf={confidence:.2f}, Qual={quality:.2f}"
                cv2.putText(vis_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw landmarks if available
                if 'landmarks' in face and face['landmarks']:
                    landmarks = face['landmarks']
                    for lm in landmarks[:20]:  # Limit landmarks for clarity
                        cv2.circle(vis_image, tuple(lm), 2, (0, 255, 0), -1)

            # Save visualization
            output_path = self.output_dir / f"face_analysis_{request_id}.png"

            # Convert RGB to BGR for OpenCV
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(output_path), vis_image_bgr)

            if success:
                logger.info(f"Generated face analysis visualization: {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"Face analysis visualization failed: {e}")
            return None

    async def _generate_statistical_explanation(self, request: ExplanationRequest,
                                              decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis of the decision"""
        try:
            stats = {
                'confidence_analysis': self._analyze_confidence_statistics(decision_info),
                'feature_statistics': self._analyze_feature_statistics(request.features),
                'model_statistics': self._analyze_model_statistics(decision_info['model_results']),
                'threshold_analysis': self._analyze_threshold_statistics(
                    decision_info, request.confidence_threshold
                )
            }

            return stats

        except Exception as e:
            logger.error(f"Statistical explanation generation failed: {e}")
            return {'error': str(e)}

    def _analyze_confidence_statistics(self, decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence-related statistics"""
        try:
            confidence = decision_info['confidence']

            # Confidence distribution analysis
            confidence_bands = {
                'very_high': (0.9, 1.0),
                'high': (0.7, 0.9),
                'moderate': (0.5, 0.7),
                'low': (0.3, 0.5),
                'very_low': (0.0, 0.3)
            }

            current_band = 'unknown'
            for band, (low, high) in confidence_bands.items():
                if low <= confidence < high:
                    current_band = band
                    break

            return {
                'confidence_value': confidence,
                'confidence_band': current_band,
                'confidence_interpretation': self._interpret_confidence(confidence),
                'statistical_significance': 'high' if confidence > 0.8 or confidence < 0.2 else 'moderate'
            }

        except Exception as e:
            logger.error(f"Confidence statistics analysis failed: {e}")
            return {'error': str(e)}

    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence value"""
        if confidence > 0.95:
            return "Extremely confident in the prediction"
        elif confidence > 0.8:
            return "High confidence in the prediction"
        elif confidence > 0.6:
            return "Moderately confident in the prediction"
        elif confidence > 0.4:
            return "Low confidence in the prediction"
        else:
            return "Very uncertain about the prediction"

    def _analyze_feature_statistics(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature-related statistics"""
        try:
            stats = {
                'feature_count': 0,
                'modality_distribution': {},
                'feature_ranges': {},
                'outlier_features': []
            }

            all_values = []

            for modality, modal_features in features.items():
                if isinstance(modal_features, dict):
                    modality_count = 0
                    modality_values = []

                    for key, value in modal_features.items():
                        if isinstance(value, (int, float)):
                            all_values.append(float(value))
                            modality_values.append(float(value))
                            modality_count += 1

                            # Check for outliers (very high or low values)
                            if value > 0.95 or value < 0.05:
                                stats['outlier_features'].append(f"{modality}_{key}")

                    stats['modality_distribution'][modality] = modality_count
                    if modality_values:
                        stats['feature_ranges'][modality] = {
                            'min': min(modality_values),
                            'max': max(modality_values),
                            'mean': np.mean(modality_values),
                            'std': np.std(modality_values)
                        }

            stats['feature_count'] = len(all_values)
            if all_values:
                stats['overall_statistics'] = {
                    'min': min(all_values),
                    'max': max(all_values),
                    'mean': np.mean(all_values),
                    'std': np.std(all_values),
                    'median': np.median(all_values)
                }

            return stats

        except Exception as e:
            logger.error(f"Feature statistics analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_model_statistics(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model-related statistics"""
        try:
            stats = {
                'model_count': 0,
                'prediction_distribution': {'deepfake': 0, 'real': 0},
                'confidence_statistics': {},
                'model_agreement': 0.0
            }

            confidences = []
            predictions = []

            for model_name, result in model_results.items():
                if 'error' not in result:
                    stats['model_count'] += 1

                    prediction = result.get('prediction', 0)
                    confidence = result.get('confidence', 0.5)

                    predictions.append(prediction)
                    confidences.append(confidence)

                    if prediction == 1:
                        stats['prediction_distribution']['deepfake'] += 1
                    else:
                        stats['prediction_distribution']['real'] += 1

            if confidences:
                stats['confidence_statistics'] = {
                    'min': min(confidences),
                    'max': max(confidences),
                    'mean': np.mean(confidences),
                    'std': np.std(confidences)
                }

                # Calculate model agreement
                if len(predictions) > 1:
                    agreement = 1.0 - np.std(predictions)
                    stats['model_agreement'] = float(agreement)

            return stats

        except Exception as e:
            logger.error(f"Model statistics analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_threshold_statistics(self, decision_info: Dict[str, Any],
                                    threshold: float) -> Dict[str, Any]:
        """Analyze threshold-related statistics"""
        try:
            confidence = decision_info['confidence']

            stats = {
                'threshold_value': threshold,
                'confidence_vs_threshold': confidence - threshold,
                'threshold_interpretation': '',
                'sensitivity_analysis': {}
            }

            # Interpret confidence relative to threshold
            diff = confidence - threshold
            if abs(diff) < 0.1:
                stats['threshold_interpretation'] = "Decision is very close to threshold"
            elif diff > 0.3:
                stats['threshold_interpretation'] = "High confidence above threshold"
            elif diff < -0.3:
                stats['threshold_interpretation'] = "High confidence below threshold"
            else:
                stats['threshold_interpretation'] = "Moderate confidence relative to threshold"

            # Sensitivity analysis - how decision would change with different thresholds
            test_thresholds = [0.3, 0.5, 0.7, 0.9]
            for test_thresh in test_thresholds:
                would_be_deepfake = confidence > test_thresh
                stats['sensitivity_analysis'][f'threshold_{test_thresh}'] = {
                    'decision': 'DEEPFAKE' if would_be_deepfake else 'REAL',
                    'same_as_current': would_be_deepfake == decision_info['is_deepfake']
                }

            return stats

        except Exception as e:
            logger.error(f"Threshold statistics analysis failed: {e}")
            return {'error': str(e)}

    async def _generate_comparative_explanation(self, request: ExplanationRequest,
                                              decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis explanation"""
        try:
            # This would compare against typical patterns for real vs deepfake content
            # For now, provide a simplified analysis
            comparative_analysis = {
                'comparison_baseline': 'typical_patterns',
                'deviation_analysis': {},
                'similarity_to_known_patterns': {}
            }

            # Analyze how features compare to expected ranges
            for modality, modal_features in request.features.items():
                if isinstance(modal_features, dict):
                    modality_deviations = []

                    for key, value in modal_features.items():
                        if isinstance(value, (int, float)):
                            # Compare to expected "normal" range (0.4 to 0.8 for most metrics)
                            expected_min, expected_max = 0.4, 0.8

                            if value < expected_min:
                                deviation = expected_min - value
                                modality_deviations.append({
                                    'feature': key,
                                    'type': 'below_expected',
                                    'deviation': deviation
                                })
                            elif value > expected_max:
                                deviation = value - expected_max
                                modality_deviations.append({
                                    'feature': key,
                                    'type': 'above_expected',
                                    'deviation': deviation
                                })

                    if modality_deviations:
                        comparative_analysis['deviation_analysis'][modality] = modality_deviations

            return comparative_analysis

        except Exception as e:
            logger.error(f"Comparative explanation generation failed: {e}")
            return {'error': str(e)}

    async def _generate_technical_explanation(self, request: ExplanationRequest,
                                            decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical details explanation"""
        try:
            technical_details = {
                'algorithm_details': self._get_algorithm_details(),
                'feature_extraction': self._get_feature_extraction_details(request.features),
                'model_architecture': self._get_model_architecture_details(decision_info['model_results']),
                'computational_details': {
                    'processing_time': decision_info.get('processing_time', 0.0),
                    'device_used': self.device,
                    'feature_dimensions': self._calculate_feature_dimensions(request.features)
                },
                'limitations_and_assumptions': self._get_technical_limitations()
            }

            return technical_details

        except Exception as e:
            logger.error(f"Technical explanation generation failed: {e}")
            return {'error': str(e)}

    def _get_algorithm_details(self) -> Dict[str, str]:
        """Get algorithm implementation details"""
        return {
            'detection_approach': 'Multi-modal ensemble learning',
            'feature_fusion': 'Attention-based cross-modal fusion',
            'decision_aggregation': 'Weighted confidence averaging',
            'explanation_method': 'Feature importance and evidence-based reasoning'
        }

    def _get_feature_extraction_details(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get feature extraction technical details"""
        details = {
            'modalities_processed': list(features.keys()),
            'feature_types': {},
            'extraction_methods': {}
        }

        for modality in features.keys():
            if modality == 'visual':
                details['feature_types'][modality] = [
                    'CNN features', 'Texture descriptors', 'Frequency domain', 'Face landmarks'
                ]
                details['extraction_methods'][modality] = [
                    'Pre-trained CNN models', 'LBP and GLCM', 'DCT and DFT', 'dlib face detection'
                ]
            elif modality == 'temporal':
                details['feature_types'][modality] = [
                    'Optical flow', 'Motion patterns', 'Temporal consistency'
                ]
                details['extraction_methods'][modality] = [
                    'Lucas-Kanade flow', 'Frame differencing', 'Temporal smoothness analysis'
                ]
            elif modality == 'audio':
                details['feature_types'][modality] = [
                    'Spectral features', 'Voice quality', 'Prosodic features'
                ]
                details['extraction_methods'][modality] = [
                    'MFCC extraction', 'Pitch tracking', 'Voice activity detection'
                ]

        return details

    def _get_model_architecture_details(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get model architecture details"""
        details = {
            'models_used': list(model_results.keys()),
            'ensemble_method': 'Weighted averaging by confidence',
            'model_types': {}
        }

        # Map model names to architectures (simplified)
        for model_name in model_results.keys():
            if 'xception' in model_name.lower():
                details['model_types'][model_name] = 'Xception CNN'
            elif 'efficientnet' in model_name.lower():
                details['model_types'][model_name] = 'EfficientNet CNN'
            elif 'i3d' in model_name.lower():
                details['model_types'][model_name] = 'Inflated 3D CNN'
            elif 'slowfast' in model_name.lower():
                details['model_types'][model_name] = 'SlowFast Networks'
            elif 'ecapa' in model_name.lower():
                details['model_types'][model_name] = 'ECAPA-TDNN'
            elif 'wav2vec' in model_name.lower():
                details['model_types'][model_name] = 'Wav2Vec2 Transformer'
            else:
                details['model_types'][model_name] = 'Unknown architecture'

        return details

    def _calculate_feature_dimensions(self, features: Dict[str, Any]) -> Dict[str, int]:
        """Calculate feature dimensions for each modality"""
        dimensions = {}

        for modality, modal_features in features.items():
            if isinstance(modal_features, dict):
                feature_count = 0
                for key, value in modal_features.items():
                    if isinstance(value, (int, float)):
                        feature_count += 1
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        try:
                            if isinstance(value, np.ndarray):
                                feature_count += value.size
                            else:
                                feature_count += len(value)
                        except:
                            feature_count += 1

                dimensions[modality] = feature_count

        return dimensions

    def _get_technical_limitations(self) -> List[str]:
        """Get technical limitations of the system"""
        return [
            "Model performance depends on training data quality and diversity",
            "Feature extraction may be affected by media compression and quality",
            "Cross-modal fusion requires synchronized audio-visual content",
            "Explanation quality depends on feature interpretability",
            "Real-time processing may require computational trade-offs",
            "Novel manipulation techniques may not be detected effectively"
        ]

    async def _generate_evidence(self, request: ExplanationRequest,
                                decision_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate evidence supporting the decision"""
        try:
            evidence_list = []

            # Visual evidence
            if 'visual' in request.features:
                visual_evidence = self._extract_visual_evidence(request.features['visual'])
                evidence_list.extend(visual_evidence)

            # Temporal evidence
            if 'temporal' in request.features:
                temporal_evidence = self._extract_temporal_evidence(request.features['temporal'])
                evidence_list.extend(temporal_evidence)

            # Audio evidence
            if 'audio' in request.features:
                audio_evidence = self._extract_audio_evidence(request.features['audio'])
                evidence_list.extend(audio_evidence)

            # Model agreement evidence
            model_evidence = self._extract_model_evidence(decision_info['model_results'])
            evidence_list.extend(model_evidence)

            # Sort evidence by strength/importance
            evidence_list.sort(key=lambda x: x.get('strength', 0.5), reverse=True)

            return evidence_list[:10]  # Return top 10 pieces of evidence

        except Exception as e:
            logger.error(f"Evidence generation failed: {e}")
            return []

    def _extract_visual_evidence(self, visual_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract visual evidence"""
        evidence = []

        try:
            # Check for artifacts
            for artifact_type, threshold in self.evidence_thresholds['visual_artifacts'].items():
                if artifact_type in visual_features:
                    value = visual_features[artifact_type]
                    if isinstance(value, (int, float)) and value > threshold:
                        evidence.append({
                            'type': EvidenceType.VISUAL_ARTIFACTS.value,
                            'description': f"High {artifact_type.replace('_', ' ')}: {value:.3f}",
                            'value': float(value),
                            'strength': min(value / threshold, 1.0),
                            'supports': 'manipulation'
                        })

            # Check face-related evidence
            if 'face_features' in visual_features:
                face_features = visual_features['face_features']

                landmark_quality = face_features.get('landmark_quality', 1.0)
                if landmark_quality < self.evidence_thresholds['facial_manipulation']['landmark_quality']:
                    evidence.append({
                        'type': EvidenceType.FACIAL_MANIPULATION.value,
                        'description': f"Poor facial landmark quality: {landmark_quality:.3f}",
                        'value': float(landmark_quality),
                        'strength': 1.0 - landmark_quality,
                        'supports': 'manipulation'
                    })

            # Check quality degradation
            overall_quality = visual_features.get('overall_visual_quality', 0.8)
            if overall_quality < self.evidence_thresholds['quality_degradation']['overall_image_quality']:
                evidence.append({
                    'type': EvidenceType.QUALITY_DEGRADATION.value,
                    'description': f"Low overall visual quality: {overall_quality:.3f}",
                    'value': float(overall_quality),
                    'strength': 1.0 - overall_quality,
                    'supports': 'manipulation'
                })

        except Exception as e:
            logger.debug(f"Visual evidence extraction failed: {e}")

        return evidence

    def _extract_temporal_evidence(self, temporal_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract temporal evidence"""
        evidence = []

        try:
            # Check temporal consistency
            for consistency_type, threshold in self.evidence_thresholds['temporal_inconsistency'].items():
                if consistency_type in temporal_features:
                    value = temporal_features[consistency_type]
                    if isinstance(value, (int, float)):
                        if consistency_type == 'temporal_consistency' and value < threshold:
                            evidence.append({
                                'type': EvidenceType.TEMPORAL_INCONSISTENCY.value,
                                'description': f"Poor {consistency_type.replace('_', ' ')}: {value:.3f}",
                                'value': float(value),
                                'strength': max(0, threshold - value),
                                'supports': 'manipulation'
                            })
                        elif consistency_type == 'motion_smoothness' and value < threshold:
                            evidence.append({
                                'type': EvidenceType.TEMPORAL_INCONSISTENCY.value,
                                'description': f"Unnatural motion patterns: {value:.3f}",
                                'value': float(value),
                                'strength': max(0, threshold - value),
                                'supports': 'manipulation'
                            })
                        elif consistency_type == 'flickering_score' and value < threshold:
                            evidence.append({
                                'type': EvidenceType.TEMPORAL_INCONSISTENCY.value,
                                'description': f"Temporal flickering detected: {value:.3f}",
                                'value': float(value),
                                'strength': max(0, threshold - value),
                                'supports': 'manipulation'
                            })

        except Exception as e:
            logger.debug(f"Temporal evidence extraction failed: {e}")

        return evidence

    def _extract_audio_evidence(self, audio_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract audio evidence"""
        evidence = []

        try:
            # Check audio quality and synthesis artifacts
            for audio_type, threshold in self.evidence_thresholds['audio_anomalies'].items():
                if audio_type in audio_features:
                    value = audio_features[audio_type]
                    if isinstance(value, (int, float)):
                        if audio_type == 'voice_quality_score' and value < threshold:
                            evidence.append({
                                'type': EvidenceType.AUDIO_ANOMALIES.value,
                                'description': f"Unnatural voice characteristics: {value:.3f}",
                                'value': float(value),
                                'strength': max(0, threshold - value),
                                'supports': 'manipulation'
                            })
                        elif audio_type == 'synthesis_artifact_score' and value > threshold:
                            evidence.append({
                                'type': EvidenceType.AUDIO_ANOMALIES.value,
                                'description': f"Audio synthesis artifacts: {value:.3f}",
                                'value': float(value),
                                'strength': min(value / threshold, 1.0),
                                'supports': 'manipulation'
                            })

        except Exception as e:
            logger.debug(f"Audio evidence extraction failed: {e}")

        return evidence

    def _extract_model_evidence(self, model_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract model agreement evidence"""
        evidence = []

        try:
            if not model_results:
                return evidence

            # Model consensus analysis
            predictions = []
            confidences = []

            for model_name, result in model_results.items():
                if 'error' not in result:
                    predictions.append(result.get('prediction', 0))
                    confidences.append(result.get('confidence', 0.5))

            if predictions:
                # High consensus
                consensus = 1.0 - np.std(predictions) if len(predictions) > 1 else 1.0
                if consensus > 0.8:
                    avg_confidence = np.mean(confidences)
                    evidence.append({
                        'type': 'model_consensus',
                        'description': f"Strong model agreement (consensus: {consensus:.3f})",
                        'value': float(consensus),
                        'strength': min(consensus * avg_confidence, 1.0),
                        'supports': 'prediction_confidence'
                    })

                # Individual high-confidence models
                for model_name, result in model_results.items():
                    if 'error' not in result:
                        confidence = result.get('confidence', 0.5)
                        if confidence > 0.9:
                            prediction = 'manipulation' if result.get('prediction', 0) == 1 else 'authentic'
                            evidence.append({
                                'type': 'high_confidence_model',
                                'description': f"{model_name} shows high confidence for {prediction}",
                                'value': float(confidence),
                                'strength': confidence,
                                'supports': prediction
                            })

        except Exception as e:
            logger.debug(f"Model evidence extraction failed: {e}")

        return evidence

    async def _generate_recommendations(self, request: ExplanationRequest,
                                       decision_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the analysis"""
        try:
            recommendations = []

            confidence = decision_info['confidence']
            is_deepfake = decision_info['is_deepfake']

            # Confidence-based recommendations
            if confidence < 0.6:
                recommendations.append(
                    "Consider additional analysis or human expert review due to low confidence"
                )

            if confidence < 0.4:
                recommendations.append(
                    "Results are highly uncertain - seek additional verification methods"
                )

            # Content-specific recommendations
            if is_deepfake and confidence > 0.7:
                recommendations.append(
                    "High probability of manipulation detected - verify source authenticity"
                )
                recommendations.append(
                    "Consider forensic analysis for legal or evidential purposes"
                )

            # Quality-based recommendations
            for modality, modal_features in request.features.items():
                if isinstance(modal_features, dict):
                    quality_keys = [k for k in modal_features.keys() if 'quality' in k.lower()]
                    for key in quality_keys:
                        value = modal_features[key]
                        if isinstance(value, (int, float)) and value < 0.4:
                            recommendations.append(
                                f"Poor {modality} quality may affect analysis accuracy - consider higher quality source"
                            )
                            break

            # General recommendations
            recommendations.extend([
                "Always cross-reference with multiple detection tools when possible",
                "Consider the context and source credibility alongside technical analysis",
                "Be aware that new manipulation techniques may evade current detection methods"
            ])

            return recommendations[:5]  # Return top 5 recommendations

        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Consider expert human review for verification"]

    def _get_limitations(self, media_type: str, decision_info: Dict[str, Any]) -> List[str]:
        """Get limitations specific to the analysis"""
        limitations = [
            "Detection accuracy depends on the quality and type of manipulation",
            "False positives may occur with heavily compressed or low-quality content",
            "Novel manipulation techniques may not be detected effectively",
            "Analysis is based on statistical patterns and may not capture all subtleties"
        ]

        # Add media-specific limitations
        if media_type == 'image':
            limitations.append("Single image analysis has limited temporal context")
        elif media_type == 'video':
            limitations.append("Video analysis may be computationally intensive and time-consuming")
        elif media_type == 'audio':
            limitations.append("Audio analysis may be affected by background noise and compression")
        elif media_type == 'multimodal':
            limitations.append("Multimodal analysis requires synchronized audio-visual content")

        # Add confidence-specific limitations
        confidence = decision_info.get('confidence', 0.5)
        if confidence < 0.7:
            limitations.append("Low confidence results require additional verification")

        return limitations

    def _update_stats(self, request: ExplanationRequest, result: ExplanationResult):
        """Update explanation statistics"""
        try:
            self.explanation_stats['total_explanations'] += 1

            # Track explanation types
            for exp_type in request.explanation_types:
                type_name = exp_type.value
                if type_name not in self.explanation_stats['explanation_types']:
                    self.explanation_stats['explanation_types'][type_name] = 0
                self.explanation_stats['explanation_types'][type_name] += 1

            # Update average processing time
            if result.processing_time > 0:
                current_avg = self.explanation_stats['average_processing_time']
                total_count = self.explanation_stats['total_explanations']

                new_avg = ((current_avg * (total_count - 1)) + result.processing_time) / total_count
                self.explanation_stats['average_processing_time'] = new_avg

        except Exception as e:
            logger.error(f"Stats update failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get explanation service statistics"""
        return self.explanation_stats.copy()

    def cleanup_outputs(self, older_than_hours: int = 24) -> int:
        """Clean up old explanation outputs"""
        try:
            import time

            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)

            removed_count = 0

            for output_file in self.output_dir.glob('*'):
                if output_file.is_file():
                    file_mtime = output_file.stat().st_mtime
                    if file_mtime < cutoff_time:
                        try:
                            output_file.unlink()
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Could not remove output file {output_file}: {e}")

            logger.info(f"Cleaned up {removed_count} explanation output files")
            return removed_count

        except Exception as e:
            logger.error(f"Output cleanup failed: {e}")
            return 0
