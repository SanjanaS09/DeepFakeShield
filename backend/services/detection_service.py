"""
Deepfake Detection Service for Multi-Modal Deepfake Detection System
Orchestrates the complete deepfake detection pipeline across image, video, and audio
Handles model management, feature extraction coordination, and result aggregation
"""

import asyncio
import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Union, Optional, Any, AsyncGenerator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum

from features.visual_features import VisualFeatureExtractor
from features.temporal_features import TemporalFeatureExtractor
from features.audio_features import AudioFeatureExtractor
from features.fusion_features import FusionFeatureExtractor
from utils.face_detection import FaceDetector
from utils.quality_assessment import QualityAssessor
from utils.validators import InputValidator, ModelValidator
from utils.file_handlers import FileHandler
from utils.logger import get_logger

logger = get_logger(__name__)

class DetectionMode(Enum):
    """Detection modes for different media types"""
    IMAGE = "image"
    VIDEO = "video" 
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class ModelType(Enum):
    """Supported model architectures"""
    XCEPTION = "xception"
    EFFICIENTNET = "efficientnet"
    RESNET = "resnet"
    I3D = "i3d"
    SLOWFAST = "slowfast"
    X3D = "x3d"
    ECAPA_TDNN = "ecapa_tdnn"
    WAV2VEC2 = "wav2vec2"
    FUSION_TRANSFORMER = "fusion_transformer"

@dataclass
class DetectionRequest:
    """Request object for deepfake detection"""
    file_path: str
    mode: DetectionMode
    model_types: List[ModelType] = None
    confidence_threshold: float = 0.5
    batch_size: int = 8
    enable_xai: bool = True
    extract_features: bool = True
    quality_assessment: bool = True
    request_id: str = None
    user_id: str = None
    metadata: Dict[str, Any] = None

@dataclass 
class DetectionResult:
    """Result object for deepfake detection"""
    request_id: str
    file_path: str
    mode: DetectionMode
    is_deepfake: bool
    confidence_score: float
    model_results: Dict[str, Any]
    features: Dict[str, Any] = None
    quality_metrics: Dict[str, Any] = None
    explanation: Dict[str, Any] = None
    processing_time: float = 0.0
    error_message: str = None
    metadata: Dict[str, Any] = None

class DeepfakeDetectionService:
    """
    Main service class for deepfake detection across all modalities
    Orchestrates model loading, feature extraction, and inference pipeline
    """

    def __init__(self, 
                 device: str = 'auto',
                 model_dir: str = 'models',
                 cache_features: bool = True,
                 max_workers: int = 4,
                 enable_gpu: bool = True):
        """
        Initialize Deepfake Detection Service

        Args:
            device: Device for computation ('cpu', 'cuda', 'auto')
            model_dir: Directory containing trained models
            cache_features: Enable feature caching
            max_workers: Maximum number of worker threads
            enable_gpu: Enable GPU acceleration
        """
        self.device = self._setup_device(device, enable_gpu)
        self.model_dir = Path(model_dir)
        self.cache_features = cache_features
        self.max_workers = max_workers

        # Initialize components
        self._init_validators()
        self._init_feature_extractors()
        self._init_models()
        self._init_thread_pools()

        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'average_processing_time': 0.0
        }

        logger.info(f"Initialized DeepfakeDetectionService on {self.device}")

    def _setup_device(self, device: str, enable_gpu: bool) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if enable_gpu and torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("Using CPU for inference")

        return torch.device(device)

    def _init_validators(self):
        """Initialize validation components"""
        self.input_validator = InputValidator()
        self.model_validator = ModelValidator()
        logger.info("Initialized validators")

    def _init_feature_extractors(self):
        """Initialize feature extraction components"""
        self.visual_extractor = VisualFeatureExtractor(device=str(self.device))
        self.temporal_extractor = TemporalFeatureExtractor(device=str(self.device))
        self.audio_extractor = AudioFeatureExtractor(device=str(self.device))
        self.fusion_extractor = FusionFeatureExtractor(device=str(self.device))

        # Utility components
        self.face_detector = FaceDetector()
        self.quality_assessor = QualityAssessor(device=str(self.device))
        self.file_handler = FileHandler()

        logger.info("Initialized feature extractors")

    def _init_models(self):
        """Initialize detection models"""
        self.models = {}
        self.model_metadata = {}

        # Model loading will be done on-demand to save memory
        self._available_models = self._scan_available_models()
        logger.info(f"Found {len(self._available_models)} available models")

    def _init_thread_pools(self):
        """Initialize thread pools for concurrent processing"""
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max(1, self.max_workers // 2))
        logger.info(f"Initialized thread pools with {self.max_workers} workers")

    def _scan_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Scan for available trained models"""
        available_models = {}

        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return available_models

        # Scan for model files
        model_extensions = {'.pth', '.pt', '.pkl', '.joblib', '.h5', '.onnx'}

        for model_file in self.model_dir.rglob('*'):
            if model_file.suffix.lower() in model_extensions:
                # Extract model information from filename/path
                model_info = self._parse_model_info(model_file)
                if model_info:
                    model_key = f"{model_info['type']}_{model_info['modality']}"
                    available_models[model_key] = {
                        'path': model_file,
                        'type': model_info['type'],
                        'modality': model_info['modality'],
                        'architecture': model_info.get('architecture', 'unknown'),
                        'size_mb': model_file.stat().st_size / (1024 * 1024)
                    }

        return available_models

    def _parse_model_info(self, model_path: Path) -> Optional[Dict[str, str]]:
        """Parse model information from file path"""
        try:
            # Expected naming convention: {modality}_{architecture}_{version}.ext
            name_parts = model_path.stem.lower().split('_')

            if len(name_parts) >= 2:
                modality = name_parts[0]  # image, video, audio, fusion
                architecture = name_parts[1]  # xception, efficientnet, etc.

                # Validate modality
                valid_modalities = {'image', 'video', 'audio', 'fusion'}
                if modality in valid_modalities:
                    return {
                        'modality': modality,
                        'type': architecture,
                        'architecture': architecture
                    }

        except Exception as e:
            logger.warning(f"Could not parse model info from {model_path}: {e}")

        return None

    async def detect_deepfake(self, request: DetectionRequest) -> DetectionResult:
        """
        Main detection method for deepfake analysis

        Args:
            request: Detection request object

        Returns:
            Detection result object
        """
        start_time = time.time()
        request_id = request.request_id or f"req_{int(time.time()*1000)}"

        logger.info(f"Starting detection for request {request_id}", extra={
            'request_id': request_id,
            'file_path': request.file_path,
            'mode': request.mode.value
        })

        try:
            # Validate request
            validation_result = await self._validate_request(request)
            if not validation_result['valid']:
                return DetectionResult(
                    request_id=request_id,
                    file_path=request.file_path,
                    mode=request.mode,
                    is_deepfake=False,
                    confidence_score=0.0,
                    model_results={},
                    error_message=f"Validation failed: {validation_result['errors']}",
                    processing_time=time.time() - start_time
                )

            # Load and prepare media
            media_data = await self._load_media(request.file_path, request.mode)
            if media_data is None:
                return DetectionResult(
                    request_id=request_id,
                    file_path=request.file_path,
                    mode=request.mode,
                    is_deepfake=False,
                    confidence_score=0.0,
                    model_results={},
                    error_message="Failed to load media file",
                    processing_time=time.time() - start_time
                )

            # Quality assessment
            quality_metrics = None
            if request.quality_assessment:
                quality_metrics = await self._assess_quality(media_data, request.mode)

            # Feature extraction
            features = None
            if request.extract_features:
                features = await self._extract_features(media_data, request.mode)

            # Model inference
            model_results = await self._run_inference(
                media_data, features, request.mode, request.model_types, request.batch_size
            )

            # Aggregate results
            final_result = self._aggregate_results(model_results, request.confidence_threshold)

            # Generate explanation if requested
            explanation = None
            if request.enable_xai and features:
                explanation = await self._generate_explanation(
                    features, model_results, final_result
                )

            # Create result object
            result = DetectionResult(
                request_id=request_id,
                file_path=request.file_path,
                mode=request.mode,
                is_deepfake=final_result['is_deepfake'],
                confidence_score=final_result['confidence'],
                model_results=model_results,
                features=features,
                quality_metrics=quality_metrics,
                explanation=explanation,
                processing_time=time.time() - start_time,
                metadata=request.metadata
            )

            # Update statistics
            self._update_stats(result)

            logger.info(f"Detection completed for request {request_id}", extra={
                'request_id': request_id,
                'is_deepfake': result.is_deepfake,
                'confidence': result.confidence_score,
                'processing_time': result.processing_time
            })

            return result

        except Exception as e:
            logger.error(f"Detection failed for request {request_id}: {e}", exc_info=True)
            return DetectionResult(
                request_id=request_id,
                file_path=request.file_path,
                mode=request.mode,
                is_deepfake=False,
                confidence_score=0.0,
                model_results={},
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    async def _validate_request(self, request: DetectionRequest) -> Dict[str, Any]:
        """Validate detection request"""
        try:
            # File validation
            file_validation = self.input_validator.validate_file_upload(
                {'file_path': request.file_path, 'size_bytes': 0},
                request.mode.value
            )

            if not file_validation['valid']:
                return {
                    'valid': False,
                    'errors': file_validation['errors']
                }

            # Parameter validation
            param_validation = self.input_validator.validate_model_parameters({
                'confidence_threshold': request.confidence_threshold,
                'batch_size': request.batch_size,
                'device': str(self.device)
            })

            if not param_validation['valid']:
                return {
                    'valid': False, 
                    'errors': param_validation['errors']
                }

            return {'valid': True, 'errors': []}

        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            return {'valid': False, 'errors': [str(e)]}

    async def _load_media(self, file_path: str, mode: DetectionMode) -> Optional[Any]:
        """Load media file based on mode"""
        try:
            file_path = Path(file_path)

            if mode == DetectionMode.IMAGE:
                return self.file_handler.load_image(file_path)

            elif mode == DetectionMode.VIDEO:
                return self.file_handler.load_video_frames(file_path, max_frames=64)

            elif mode == DetectionMode.AUDIO:
                return self.file_handler.load_audio(file_path)

            elif mode == DetectionMode.MULTIMODAL:
                # Load all available modalities
                media_data = {}

                # Try to load as video (includes visual and potentially audio)
                video_frames = self.file_handler.load_video_frames(file_path, max_frames=64)
                if video_frames:
                    media_data['video'] = video_frames

                # Try to load audio
                audio_data = self.file_handler.load_audio(file_path)
                if audio_data:
                    media_data['audio'] = audio_data

                # Try to load as image if video failed
                if 'video' not in media_data:
                    image_data = self.file_handler.load_image(file_path)
                    if image_data is not None:
                        media_data['image'] = image_data

                return media_data if media_data else None

            return None

        except Exception as e:
            logger.error(f"Media loading failed: {e}")
            return None

    async def _assess_quality(self, media_data: Any, mode: DetectionMode) -> Dict[str, Any]:
        """Assess media quality"""
        try:
            if mode == DetectionMode.IMAGE:
                return self.quality_assessor.assess_image_quality(media_data)

            elif mode == DetectionMode.VIDEO:
                return self.quality_assessor.assess_video_quality(media_data)

            elif mode == DetectionMode.AUDIO:
                audio_data, sample_rate = media_data
                return self.quality_assessor.assess_audio_quality(audio_data, sample_rate)

            elif mode == DetectionMode.MULTIMODAL:
                quality_results = {}

                if 'image' in media_data:
                    quality_results['image'] = self.quality_assessor.assess_image_quality(
                        media_data['image']
                    )

                if 'video' in media_data:
                    quality_results['video'] = self.quality_assessor.assess_video_quality(
                        media_data['video']
                    )

                if 'audio' in media_data:
                    audio_data, sample_rate = media_data['audio']
                    quality_results['audio'] = self.quality_assessor.assess_audio_quality(
                        audio_data, sample_rate
                    )

                # Overall multimodal quality
                if len(quality_results) > 1:
                    multimodal_quality = self.quality_assessor.assess_multimodal_quality(
                        image=media_data.get('image'),
                        frames=media_data.get('video'),
                        audio=media_data.get('audio', (None, None))[0] if 'audio' in media_data else None,
                        sample_rate=media_data.get('audio', (None, None))[1] if 'audio' in media_data else 16000
                    )
                    quality_results['multimodal'] = multimodal_quality

                return quality_results

            return {}

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {}

    async def _extract_features(self, media_data: Any, mode: DetectionMode) -> Dict[str, Any]:
        """Extract features from media"""
        try:
            features = {}

            if mode == DetectionMode.IMAGE:
                # Extract visual features
                visual_features = self.visual_extractor.extract_all_visual_features(media_data)
                features['visual'] = visual_features

                # Face-specific analysis
                faces = self.face_detector.detect_faces(media_data)
                if faces:
                    face_features = self.visual_extractor.extract_face_features(
                        media_data, faces[0]  # Use best face
                    )
                    features['face'] = face_features

            elif mode == DetectionMode.VIDEO:
                # Extract temporal features
                temporal_features = self.temporal_extractor.extract_all_temporal_features(media_data)
                features['temporal'] = temporal_features

                # Extract visual features from key frames
                key_frames = media_data[::len(media_data)//5] if len(media_data) > 5 else media_data
                visual_features_list = []

                for frame in key_frames:
                    visual_feat = self.visual_extractor.extract_all_visual_features(frame)
                    visual_features_list.append(visual_feat)

                features['visual_sequence'] = visual_features_list

            elif mode == DetectionMode.AUDIO:
                # Extract audio features
                audio_data, sample_rate = media_data
                audio_features = self.audio_extractor.extract_all_audio_features(audio_data)
                features['audio'] = audio_features

            elif mode == DetectionMode.MULTIMODAL:
                # Extract features from all available modalities
                if 'image' in media_data:
                    visual_features = self.visual_extractor.extract_all_visual_features(
                        media_data['image']
                    )
                    features['visual'] = visual_features

                if 'video' in media_data:
                    temporal_features = self.temporal_extractor.extract_all_temporal_features(
                        media_data['video']
                    )
                    features['temporal'] = temporal_features

                if 'audio' in media_data:
                    audio_data, sample_rate = media_data['audio']
                    audio_features = self.audio_extractor.extract_all_audio_features(audio_data)
                    features['audio'] = audio_features

                # Multi-modal fusion
                if len(features) > 1:
                    # Convert features to tensors for fusion
                    fusion_input = {}
                    for modality, modal_features in features.items():
                        if isinstance(modal_features, dict):
                            # Convert dict features to tensor representation
                            feature_tensor = self._features_dict_to_tensor(modal_features)
                            fusion_input[modality] = feature_tensor

                    # Perform fusion
                    fusion_results = self.fusion_extractor.extract_fusion_features(
                        **fusion_input
                    )
                    features['fusion'] = fusion_results

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def _features_dict_to_tensor(self, features_dict: Dict[str, Any]) -> torch.Tensor:
        """Convert feature dictionary to tensor representation"""
        try:
            # Extract numerical features
            numerical_features = []

            for key, value in features_dict.items():
                if isinstance(value, (int, float)):
                    numerical_features.append(float(value))
                elif isinstance(value, (list, tuple, np.ndarray)):
                    if isinstance(value, np.ndarray):
                        value = value.flatten()
                    numerical_features.extend([float(v) for v in value])

            # Convert to tensor
            if numerical_features:
                tensor = torch.tensor(numerical_features, dtype=torch.float32, device=self.device)
                return tensor.unsqueeze(0)  # Add batch dimension
            else:
                # Return zero tensor if no numerical features
                return torch.zeros(1, 256, device=self.device)

        except Exception as e:
            logger.warning(f"Feature tensor conversion failed: {e}")
            return torch.zeros(1, 256, device=self.device)

    async def _run_inference(self, media_data: Any, features: Dict[str, Any], 
                           mode: DetectionMode, model_types: List[ModelType], 
                           batch_size: int) -> Dict[str, Any]:
        """Run model inference on extracted features"""
        try:
            model_results = {}

            # Determine which models to use
            if model_types is None:
                model_types = self._get_default_models(mode)

            # Load and run each model
            for model_type in model_types:
                try:
                    model = await self._load_model(model_type, mode)
                    if model is None:
                        logger.warning(f"Model not available: {model_type}")
                        continue

                    # Run inference
                    with torch.no_grad():
                        if mode == DetectionMode.IMAGE:
                            prediction = await self._predict_image(model, media_data, features)
                        elif mode == DetectionMode.VIDEO:
                            prediction = await self._predict_video(model, media_data, features)
                        elif mode == DetectionMode.AUDIO:
                            prediction = await self._predict_audio(model, media_data, features)
                        elif mode == DetectionMode.MULTIMODAL:
                            prediction = await self._predict_multimodal(model, media_data, features)
                        else:
                            prediction = {'confidence': 0.5, 'prediction': 0}

                    model_results[model_type.value] = prediction

                except Exception as e:
                    logger.error(f"Model inference failed for {model_type}: {e}")
                    model_results[model_type.value] = {
                        'confidence': 0.0,
                        'prediction': 0,
                        'error': str(e)
                    }

            return model_results

        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            return {}

    def _get_default_models(self, mode: DetectionMode) -> List[ModelType]:
        """Get default model types for each mode"""
        defaults = {
            DetectionMode.IMAGE: [ModelType.XCEPTION, ModelType.EFFICIENTNET],
            DetectionMode.VIDEO: [ModelType.I3D, ModelType.SLOWFAST],
            DetectionMode.AUDIO: [ModelType.ECAPA_TDNN, ModelType.WAV2VEC2],
            DetectionMode.MULTIMODAL: [ModelType.FUSION_TRANSFORMER]
        }

        return defaults.get(mode, [])

    async def _load_model(self, model_type: ModelType, mode: DetectionMode) -> Optional[Any]:
        """Load a specific model"""
        try:
            model_key = f"{model_type.value}_{mode.value}"

            # Check if model is already loaded
            if model_key in self.models:
                return self.models[model_key]

            # Check if model is available
            if model_key not in self._available_models:
                logger.warning(f"Model not available: {model_key}")
                return None

            # Load model
            model_info = self._available_models[model_key]
            model_path = model_info['path']

            model = self.file_handler.load_model(model_path)
            if model is None:
                logger.error(f"Failed to load model: {model_path}")
                return None

            # Move to device and set eval mode
            if hasattr(model, 'to'):
                model = model.to(self.device)

            if hasattr(model, 'eval'):
                model.eval()

            # Cache model
            self.models[model_key] = model

            logger.info(f"Loaded model: {model_key}")
            return model

        except Exception as e:
            logger.error(f"Model loading failed for {model_type}: {e}")
            return None

    async def _predict_image(self, model: Any, image_data: np.ndarray, 
                           features: Dict[str, Any]) -> Dict[str, Any]:
        """Run image deepfake prediction"""
        try:
            # Mock prediction - replace with actual model inference
            # This would typically involve:
            # 1. Preprocessing the image
            # 2. Running model inference
            # 3. Post-processing results

            # For now, use features to simulate prediction
            visual_features = features.get('visual', {})

            # Simple heuristic based on quality and artifact features
            quality_score = visual_features.get('overall_visual_quality', 0.5)
            artifact_score = visual_features.get('overall_artifact_score', 0.1)

            # Simulate model confidence
            confidence = 0.3 + 0.4 * (1 - quality_score) + 0.3 * artifact_score
            confidence = max(0.0, min(1.0, confidence))

            prediction = 1 if confidence > 0.5 else 0

            return {
                'confidence': float(confidence),
                'prediction': int(prediction),
                'probabilities': [1-confidence, confidence],
                'model_type': 'image_classifier'
            }

        except Exception as e:
            logger.error(f"Image prediction failed: {e}")
            return {'confidence': 0.5, 'prediction': 0, 'error': str(e)}

    async def _predict_video(self, model: Any, video_data: List[np.ndarray], 
                           features: Dict[str, Any]) -> Dict[str, Any]:
        """Run video deepfake prediction"""
        try:
            # Mock prediction for video
            temporal_features = features.get('temporal', {})

            # Use temporal consistency as indicator
            temporal_consistency = temporal_features.get('temporal_consistency', 0.8)
            motion_smoothness = temporal_features.get('motion_smoothness', 0.7)

            # Simulate video-based confidence
            confidence = 0.4 + 0.3 * (1 - temporal_consistency) + 0.3 * (1 - motion_smoothness)
            confidence = max(0.0, min(1.0, confidence))

            prediction = 1 if confidence > 0.5 else 0

            return {
                'confidence': float(confidence),
                'prediction': int(prediction),
                'probabilities': [1-confidence, confidence],
                'model_type': 'video_classifier',
                'num_frames': len(video_data)
            }

        except Exception as e:
            logger.error(f"Video prediction failed: {e}")
            return {'confidence': 0.5, 'prediction': 0, 'error': str(e)}

    async def _predict_audio(self, model: Any, audio_data: Tuple[np.ndarray, int], 
                           features: Dict[str, Any]) -> Dict[str, Any]:
        """Run audio deepfake prediction"""
        try:
            # Mock prediction for audio
            audio_features = features.get('audio', {})
            audio_array, sample_rate = audio_data

            # Use voice quality metrics
            voice_quality = audio_features.get('voice_quality_score', 0.7)
            synthesis_artifacts = audio_features.get('synthesis_artifact_score', 0.1)

            # Simulate audio-based confidence
            confidence = 0.2 + 0.4 * (1 - voice_quality) + 0.4 * synthesis_artifacts
            confidence = max(0.0, min(1.0, confidence))

            prediction = 1 if confidence > 0.5 else 0

            return {
                'confidence': float(confidence),
                'prediction': int(prediction),
                'probabilities': [1-confidence, confidence],
                'model_type': 'audio_classifier',
                'duration': len(audio_array) / sample_rate
            }

        except Exception as e:
            logger.error(f"Audio prediction failed: {e}")
            return {'confidence': 0.5, 'prediction': 0, 'error': str(e)}

    async def _predict_multimodal(self, model: Any, media_data: Dict[str, Any], 
                                features: Dict[str, Any]) -> Dict[str, Any]:
        """Run multimodal deepfake prediction"""
        try:
            # Mock multimodal prediction
            fusion_features = features.get('fusion', {})

            # Use fusion consistency scores
            overall_score = fusion_features.get('overall_fusion_score', 0.5)
            consistency_score = fusion_features.get('consistency_analysis', {}).get('prediction_consistency', 0.7)

            # Simulate fusion-based confidence
            confidence = 0.3 + 0.4 * (1 - consistency_score) + 0.3 * overall_score
            confidence = max(0.0, min(1.0, confidence))

            prediction = 1 if confidence > 0.5 else 0

            # Get modality contributions
            modality_contributions = fusion_features.get('modality_contributions', {})

            return {
                'confidence': float(confidence),
                'prediction': int(prediction),
                'probabilities': [1-confidence, confidence],
                'model_type': 'fusion_classifier',
                'modality_contributions': modality_contributions,
                'num_modalities': len(media_data)
            }

        except Exception as e:
            logger.error(f"Multimodal prediction failed: {e}")
            return {'confidence': 0.5, 'prediction': 0, 'error': str(e)}

    def _aggregate_results(self, model_results: Dict[str, Any], 
                          confidence_threshold: float) -> Dict[str, Any]:
        """Aggregate results from multiple models"""
        try:
            if not model_results:
                return {'is_deepfake': False, 'confidence': 0.0, 'method': 'default'}

            # Extract confidences and predictions
            confidences = []
            predictions = []

            for model_name, result in model_results.items():
                if 'error' not in result:
                    confidences.append(result['confidence'])
                    predictions.append(result['prediction'])

            if not confidences:
                return {'is_deepfake': False, 'confidence': 0.0, 'method': 'error'}

            # Ensemble methods
            # Method 1: Average confidence
            avg_confidence = np.mean(confidences)

            # Method 2: Weighted voting (by confidence)
            weights = np.array(confidences)
            weighted_pred = np.average(predictions, weights=weights)

            # Method 3: Majority voting
            majority_pred = 1 if np.mean(predictions) > 0.5 else 0

            # Final decision (use weighted approach)
            final_confidence = avg_confidence
            final_prediction = 1 if weighted_pred > 0.5 else 0

            # Apply threshold
            is_deepfake = final_confidence > confidence_threshold and final_prediction == 1

            return {
                'is_deepfake': is_deepfake,
                'confidence': float(final_confidence),
                'method': 'weighted_ensemble',
                'individual_confidences': confidences,
                'individual_predictions': predictions
            }

        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            return {'is_deepfake': False, 'confidence': 0.0, 'method': 'error'}

    async def _generate_explanation(self, features: Dict[str, Any], 
                                  model_results: Dict[str, Any], 
                                  final_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for the detection result"""
        try:
            explanation = {
                'decision': 'DEEPFAKE' if final_result['is_deepfake'] else 'REAL',
                'confidence': final_result['confidence'],
                'reasoning': [],
                'evidence': {},
                'feature_importance': {},
                'model_contributions': {}
            }

            # Add reasoning based on features
            if 'visual' in features:
                visual_quality = features['visual'].get('overall_visual_quality', 0.5)
                artifact_score = features['visual'].get('overall_artifact_score', 0.1)

                if visual_quality < 0.3:
                    explanation['reasoning'].append("Low visual quality detected")
                if artifact_score > 0.5:
                    explanation['reasoning'].append("High artifact levels detected")

                explanation['evidence']['visual_quality'] = visual_quality
                explanation['evidence']['artifact_score'] = artifact_score

            if 'temporal' in features:
                temporal_consistency = features['temporal'].get('temporal_consistency', 0.8)
                if temporal_consistency < 0.5:
                    explanation['reasoning'].append("Poor temporal consistency")
                explanation['evidence']['temporal_consistency'] = temporal_consistency

            if 'audio' in features:
                voice_quality = features['audio'].get('voice_quality_score', 0.7)
                if voice_quality < 0.4:
                    explanation['reasoning'].append("Unnatural voice characteristics")
                explanation['evidence']['voice_quality'] = voice_quality

            # Model contribution analysis
            for model_name, result in model_results.items():
                if 'error' not in result:
                    explanation['model_contributions'][model_name] = {
                        'confidence': result['confidence'],
                        'prediction': 'DEEPFAKE' if result['prediction'] == 1 else 'REAL'
                    }

            # Feature importance (simplified)
            all_features = {}
            for modality, modal_features in features.items():
                if isinstance(modal_features, dict):
                    for key, value in modal_features.items():
                        if isinstance(value, (int, float)):
                            all_features[f"{modality}_{key}"] = abs(float(value) - 0.5)

            # Top 5 most important features
            sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
            explanation['feature_importance'] = dict(sorted_features[:5])

            return explanation

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {'decision': 'UNKNOWN', 'error': str(e)}

    def _update_stats(self, result: DetectionResult):
        """Update detection statistics"""
        try:
            self.detection_stats['total_detections'] += 1

            if result.error_message:
                self.detection_stats['failed_detections'] += 1
            else:
                self.detection_stats['successful_detections'] += 1

                # Update average processing time
                current_avg = self.detection_stats['average_processing_time']
                total_successful = self.detection_stats['successful_detections']

                new_avg = ((current_avg * (total_successful - 1)) + result.processing_time) / total_successful
                self.detection_stats['average_processing_time'] = new_avg

        except Exception as e:
            logger.error(f"Stats update failed: {e}")

    async def batch_detect(self, requests: List[DetectionRequest]) -> List[DetectionResult]:
        """Process multiple detection requests concurrently"""
        try:
            logger.info(f"Starting batch detection for {len(requests)} requests")

            # Process requests concurrently
            tasks = [self.detect_deepfake(request) for request in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch request {i} failed: {result}")
                    final_results.append(DetectionResult(
                        request_id=requests[i].request_id or f"batch_{i}",
                        file_path=requests[i].file_path,
                        mode=requests[i].mode,
                        is_deepfake=False,
                        confidence_score=0.0,
                        model_results={},
                        error_message=str(result)
                    ))
                else:
                    final_results.append(result)

            logger.info(f"Batch detection completed: {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [DetectionResult(
                request_id=req.request_id or f"batch_{i}",
                file_path=req.file_path,
                mode=req.mode,
                is_deepfake=False,
                confidence_score=0.0,
                model_results={},
                error_message=str(e)
            ) for i, req in enumerate(requests)]

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        stats = self.detection_stats.copy()
        stats['success_rate'] = (
            stats['successful_detections'] / max(stats['total_detections'], 1)
        )
        stats['available_models'] = list(self._available_models.keys())
        stats['device'] = str(self.device)
        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            health = {
                'status': 'healthy',
                'device': str(self.device),
                'models_loaded': len(self.models),
                'models_available': len(self._available_models),
                'feature_extractors': {
                    'visual': self.visual_extractor is not None,
                    'temporal': self.temporal_extractor is not None,
                    'audio': self.audio_extractor is not None,
                    'fusion': self.fusion_extractor is not None
                },
                'utilities': {
                    'face_detector': self.face_detector is not None,
                    'quality_assessor': self.quality_assessor is not None,
                    'file_handler': self.file_handler is not None
                }
            }

            # Check GPU availability if using CUDA
            if self.device.type == 'cuda':
                health['gpu_memory'] = {
                    'allocated': torch.cuda.memory_allocated(),
                    'cached': torch.cuda.memory_reserved(),
                    'max_allocated': torch.cuda.max_memory_allocated()
                }

            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear models from memory
            self.models.clear()

            # Cleanup thread pools
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)

            # Clear GPU cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            logger.info("Detection service cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
