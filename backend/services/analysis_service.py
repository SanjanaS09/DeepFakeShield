"""
Media Analysis Service for Multi-Modal Deepfake Detection System
Provides comprehensive media analysis, preprocessing, and statistical analysis
Handles metadata extraction, format conversion, and content analysis
"""

import asyncio
import cv2
import numpy as np
import logging
import json
import time
import hashlib
from typing import Dict, List, Tuple, Union, Optional, Any, AsyncGenerator
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import mimetypes

from utils.file_handlers import FileHandler
from utils.quality_assessment import QualityAssessor
from utils.face_detection import FaceDetector
from utils.artifact_analysis import ArtifactAnalyzer
from utils.validators import DataValidator
from utils.logger import get_logger
from features.visual_features import VisualFeatureExtractor
from features.temporal_features import TemporalFeatureExtractor
from features.audio_features import AudioFeatureExtractor

logger = get_logger(__name__)

class AnalysisType(Enum):
    """Types of analysis available"""
    METADATA = "metadata"
    QUALITY = "quality" 
    CONTENT = "content"
    ARTIFACTS = "artifacts"
    STATISTICS = "statistics"
    PREPROCESSING = "preprocessing"
    COMPREHENSIVE = "comprehensive"

class MediaType(Enum):
    """Supported media types"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"

@dataclass
class AnalysisRequest:
    """Request object for media analysis"""
    file_path: str
    analysis_types: List[AnalysisType]
    output_format: str = "json"  # json, detailed, summary
    include_thumbnails: bool = False
    extract_frames: bool = False
    max_frames: int = 10
    quality_assessment: bool = True
    face_analysis: bool = True
    artifact_detection: bool = True
    request_id: str = None
    user_id: str = None
    metadata: Dict[str, Any] = None

@dataclass
class AnalysisResult:
    """Result object for media analysis"""
    request_id: str
    file_path: str
    media_type: MediaType
    file_info: Dict[str, Any]
    metadata_analysis: Dict[str, Any] = None
    quality_analysis: Dict[str, Any] = None
    content_analysis: Dict[str, Any] = None
    artifact_analysis: Dict[str, Any] = None
    statistical_analysis: Dict[str, Any] = None
    preprocessing_info: Dict[str, Any] = None
    thumbnails: List[str] = None
    extracted_frames: List[str] = None
    processing_time: float = 0.0
    error_message: str = None
    warnings: List[str] = None

class MediaAnalysisService:
    """
    Comprehensive media analysis service
    Provides detailed analysis of images, videos, and audio files
    """

    def __init__(self, 
                 temp_dir: str = 'temp',
                 cache_results: bool = True,
                 max_file_size: int = 500 * 1024 * 1024,  # 500MB
                 enable_gpu: bool = True):
        """
        Initialize Media Analysis Service

        Args:
            temp_dir: Temporary directory for processing
            cache_results: Enable result caching
            max_file_size: Maximum file size to process
            enable_gpu: Enable GPU acceleration
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.cache_results = cache_results
        self.max_file_size = max_file_size
        self.enable_gpu = enable_gpu

        # Initialize components
        self._init_components()

        # Cache for results
        self.result_cache = {} if cache_results else None

        # Analysis statistics
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'average_processing_time': 0.0
        }

        logger.info("Initialized MediaAnalysisService")

    def _init_components(self):
        """Initialize analysis components"""
        device = 'cuda' if self.enable_gpu else 'cpu'

        self.file_handler = FileHandler(max_file_size=self.max_file_size)
        self.quality_assessor = QualityAssessor(device=device)
        self.face_detector = FaceDetector()
        self.artifact_analyzer = ArtifactAnalyzer(device=device)
        self.data_validator = DataValidator()

        # Feature extractors for content analysis
        self.visual_extractor = VisualFeatureExtractor(device=device)
        self.temporal_extractor = TemporalFeatureExtractor(device=device)
        self.audio_extractor = AudioFeatureExtractor(device=device)

        logger.info("Initialized analysis components")

    async def analyze_media(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Main media analysis method

        Args:
            request: Analysis request object

        Returns:
            Analysis result object
        """
        start_time = time.time()
        request_id = request.request_id or f"analysis_{int(time.time()*1000)}"

        logger.info(f"Starting media analysis for request {request_id}", extra={
            'request_id': request_id,
            'file_path': request.file_path,
            'analysis_types': [at.value for at in request.analysis_types]
        })

        try:
            # Check cache first
            if self.cache_results:
                cached_result = await self._check_cache(request)
                if cached_result:
                    self.analysis_stats['cache_hits'] += 1
                    logger.info(f"Cache hit for request {request_id}")
                    return cached_result

            # Validate file
            file_validation = self._validate_file(request.file_path)
            if not file_validation['valid']:
                return AnalysisResult(
                    request_id=request_id,
                    file_path=request.file_path,
                    media_type=MediaType.UNKNOWN,
                    file_info={},
                    error_message=f"File validation failed: {file_validation['errors']}",
                    processing_time=time.time() - start_time
                )

            # Get file info and determine media type
            file_info = self.file_handler.get_file_info(request.file_path)
            media_type = self._determine_media_type(file_info)

            # Initialize result
            result = AnalysisResult(
                request_id=request_id,
                file_path=request.file_path,
                media_type=media_type,
                file_info=file_info,
                warnings=[]
            )

            # Load media data
            media_data = await self._load_media_data(request.file_path, media_type)
            if media_data is None:
                result.error_message = "Failed to load media data"
                result.processing_time = time.time() - start_time
                return result

            # Perform requested analyses
            for analysis_type in request.analysis_types:
                try:
                    if analysis_type == AnalysisType.METADATA:
                        result.metadata_analysis = await self._analyze_metadata(
                            media_data, media_type, file_info
                        )

                    elif analysis_type == AnalysisType.QUALITY:
                        result.quality_analysis = await self._analyze_quality(
                            media_data, media_type
                        )

                    elif analysis_type == AnalysisType.CONTENT:
                        result.content_analysis = await self._analyze_content(
                            media_data, media_type, request
                        )

                    elif analysis_type == AnalysisType.ARTIFACTS:
                        result.artifact_analysis = await self._analyze_artifacts(
                            media_data, media_type
                        )

                    elif analysis_type == AnalysisType.STATISTICS:
                        result.statistical_analysis = await self._analyze_statistics(
                            media_data, media_type
                        )

                    elif analysis_type == AnalysisType.PREPROCESSING:
                        result.preprocessing_info = await self._analyze_preprocessing(
                            media_data, media_type
                        )

                    elif analysis_type == AnalysisType.COMPREHENSIVE:
                        # Run all analyses
                        result.metadata_analysis = await self._analyze_metadata(
                            media_data, media_type, file_info
                        )
                        result.quality_analysis = await self._analyze_quality(
                            media_data, media_type
                        )
                        result.content_analysis = await self._analyze_content(
                            media_data, media_type, request
                        )
                        result.artifact_analysis = await self._analyze_artifacts(
                            media_data, media_type
                        )
                        result.statistical_analysis = await self._analyze_statistics(
                            media_data, media_type
                        )
                        result.preprocessing_info = await self._analyze_preprocessing(
                            media_data, media_type
                        )

                except Exception as e:
                    error_msg = f"Analysis {analysis_type.value} failed: {str(e)}"
                    result.warnings.append(error_msg)
                    logger.warning(error_msg)

            # Generate thumbnails and extract frames if requested
            if request.include_thumbnails or request.extract_frames:
                await self._generate_visual_outputs(
                    media_data, media_type, request, result
                )

            # Calculate processing time
            result.processing_time = time.time() - start_time

            # Cache result
            if self.cache_results:
                await self._cache_result(request, result)

            # Update statistics
            self._update_stats(result)

            logger.info(f"Media analysis completed for request {request_id}", extra={
                'request_id': request_id,
                'processing_time': result.processing_time,
                'media_type': media_type.value
            })

            return result

        except Exception as e:
            logger.error(f"Media analysis failed for request {request_id}: {e}", exc_info=True)
            return AnalysisResult(
                request_id=request_id,
                file_path=request.file_path,
                media_type=MediaType.UNKNOWN,
                file_info={},
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    def _validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate media file"""
        try:
            validation = self.file_handler.validate_file(file_path)
            return validation
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return {'valid': False, 'errors': [str(e)]}

    def _determine_media_type(self, file_info: Dict[str, Any]) -> MediaType:
        """Determine media type from file info"""
        try:
            file_type = file_info.get('file_type', 'unknown')
            mime_type = file_info.get('mime_type', '')

            if file_type == 'image' or (mime_type and mime_type.startswith('image/')):
                return MediaType.IMAGE
            elif file_type == 'video' or (mime_type and mime_type.startswith('video/')):
                return MediaType.VIDEO
            elif file_type == 'audio' or (mime_type and mime_type.startswith('audio/')):
                return MediaType.AUDIO
            else:
                return MediaType.UNKNOWN

        except Exception as e:
            logger.warning(f"Media type determination failed: {e}")
            return MediaType.UNKNOWN

    async def _load_media_data(self, file_path: str, media_type: MediaType) -> Optional[Any]:
        """Load media data based on type"""
        try:
            if media_type == MediaType.IMAGE:
                return self.file_handler.load_image(file_path)
            elif media_type == MediaType.VIDEO:
                return self.file_handler.load_video_frames(file_path, max_frames=100)
            elif media_type == MediaType.AUDIO:
                return self.file_handler.load_audio(file_path)
            else:
                logger.warning(f"Unsupported media type: {media_type}")
                return None

        except Exception as e:
            logger.error(f"Media data loading failed: {e}")
            return None

    async def _analyze_metadata(self, media_data: Any, media_type: MediaType, 
                              file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze media metadata"""
        try:
            metadata = {
                'file_metadata': file_info,
                'technical_metadata': {},
                'content_metadata': {},
                'creation_info': {}
            }

            if media_type == MediaType.IMAGE:
                image = media_data
                metadata['technical_metadata'] = {
                    'dimensions': {
                        'width': image.shape[1],
                        'height': image.shape[0],
                        'channels': image.shape[2] if len(image.shape) > 2 else 1
                    },
                    'color_space': 'RGB' if len(image.shape) == 3 else 'Grayscale',
                    'data_type': str(image.dtype),
                    'pixel_count': image.size
                }

                # Extract EXIF data if possible
                try:
                    from PIL import Image
                    from PIL.ExifTags import TAGS

                    pil_image = Image.open(file_info['path'])
                    exif_data = {}

                    if hasattr(pil_image, '_getexif') and pil_image._getexif():
                        exif = pil_image._getexif()
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            exif_data[tag] = str(value)

                    metadata['content_metadata']['exif'] = exif_data

                except Exception as e:
                    logger.debug(f"EXIF extraction failed: {e}")

            elif media_type == MediaType.VIDEO:
                frames = media_data
                if frames:
                    first_frame = frames[0]
                    metadata['technical_metadata'] = {
                        'frame_count': len(frames),
                        'dimensions': {
                            'width': first_frame.shape[1],
                            'height': first_frame.shape[0],
                            'channels': first_frame.shape[2] if len(first_frame.shape) > 2 else 1
                        },
                        'estimated_fps': 30,  # Default, would need video metadata for actual FPS
                        'estimated_duration': len(frames) / 30,
                        'color_space': 'RGB' if len(first_frame.shape) == 3 else 'Grayscale'
                    }

            elif media_type == MediaType.AUDIO:
                audio_data, sample_rate = media_data
                duration = len(audio_data) / sample_rate

                metadata['technical_metadata'] = {
                    'sample_rate': sample_rate,
                    'duration': duration,
                    'samples': len(audio_data),
                    'channels': 1,  # Assuming mono
                    'data_type': str(audio_data.dtype),
                    'bit_depth': 16 if audio_data.dtype == np.int16 else 32
                }

            # Add creation timestamp
            metadata['creation_info'] = {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'file_modified_time': datetime.fromtimestamp(
                    file_info.get('modified_time', time.time()), timezone.utc
                ).isoformat() if 'modified_time' in file_info else None
            }

            return metadata

        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            return {'error': str(e)}

    async def _analyze_quality(self, media_data: Any, media_type: MediaType) -> Dict[str, Any]:
        """Analyze media quality"""
        try:
            if media_type == MediaType.IMAGE:
                return self.quality_assessor.assess_image_quality(media_data)

            elif media_type == MediaType.VIDEO:
                return self.quality_assessor.assess_video_quality(media_data)

            elif media_type == MediaType.AUDIO:
                audio_data, sample_rate = media_data
                return self.quality_assessor.assess_audio_quality(audio_data, sample_rate)

            return {'error': f'Quality analysis not supported for {media_type}'}

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {'error': str(e)}

    async def _analyze_content(self, media_data: Any, media_type: MediaType, 
                             request: AnalysisRequest) -> Dict[str, Any]:
        """Analyze media content"""
        try:
            content_analysis = {}

            if media_type == MediaType.IMAGE:
                image = media_data

                # Face detection and analysis
                if request.face_analysis:
                    faces = self.face_detector.detect_faces(image)
                    content_analysis['faces'] = {
                        'count': len(faces),
                        'details': faces[:5] if faces else [],  # Limit to 5 faces
                        'has_faces': len(faces) > 0
                    }

                    # Face quality analysis
                    if faces:
                        best_face = self.face_detector.get_best_face(image)
                        if best_face:
                            content_analysis['best_face'] = {
                                'bbox': best_face['bbox'],
                                'confidence': best_face['confidence'],
                                'quality_score': best_face.get('quality_score', 0.5)
                            }

                # Visual content features
                try:
                    visual_features = self.visual_extractor.extract_basic_features(image)
                    content_analysis['visual_features'] = {
                        'color_histogram': visual_features.get('color_histogram', {})[:10],  # Limit size
                        'edge_density': visual_features.get('edge_density', 0.0),
                        'texture_measures': visual_features.get('texture_measures', {}),
                        'brightness': float(np.mean(image)),
                        'contrast': float(np.std(image))
                    }
                except Exception as e:
                    logger.debug(f"Visual feature extraction failed: {e}")

            elif media_type == MediaType.VIDEO:
                frames = media_data

                # Temporal analysis
                try:
                    temporal_features = self.temporal_extractor.analyze_motion_patterns(frames)
                    content_analysis['temporal_features'] = {
                        'motion_intensity': temporal_features.get('motion_intensity', 0.0),
                        'motion_consistency': temporal_features.get('motion_consistency', 0.0),
                        'scene_changes': temporal_features.get('scene_changes', 0),
                        'motion_smoothness': temporal_features.get('motion_smoothness', 0.0)
                    }
                except Exception as e:
                    logger.debug(f"Temporal feature extraction failed: {e}")

                # Face analysis in video
                if request.face_analysis and frames:
                    face_detection_results = []
                    sample_frames = frames[::max(1, len(frames)//5)]  # Sample 5 frames

                    for i, frame in enumerate(sample_frames):
                        faces = self.face_detector.detect_faces(frame)
                        face_detection_results.append({
                            'frame_index': i * max(1, len(frames)//5),
                            'face_count': len(faces),
                            'faces': faces[:3] if faces else []  # Limit to 3 faces per frame
                        })

                    content_analysis['face_tracking'] = {
                        'sampled_frames': len(sample_frames),
                        'results': face_detection_results,
                        'average_face_count': np.mean([r['face_count'] for r in face_detection_results])
                    }

            elif media_type == MediaType.AUDIO:
                audio_data, sample_rate = media_data

                # Audio content analysis
                try:
                    audio_features = self.audio_extractor.extract_basic_features(audio_data, sample_rate)
                    content_analysis['audio_features'] = {
                        'spectral_centroid': audio_features.get('spectral_centroid', 0.0),
                        'zero_crossing_rate': audio_features.get('zero_crossing_rate', 0.0),
                        'energy': audio_features.get('energy', 0.0),
                        'pitch_features': audio_features.get('pitch_features', {}),
                        'voice_activity': audio_features.get('voice_activity', 0.0)
                    }
                except Exception as e:
                    logger.debug(f"Audio feature extraction failed: {e}")

            return content_analysis

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {'error': str(e)}

    async def _analyze_artifacts(self, media_data: Any, media_type: MediaType) -> Dict[str, Any]:
        """Analyze media artifacts"""
        try:
            if media_type == MediaType.IMAGE:
                return self.artifact_analyzer.analyze_all_artifacts(media_data)

            elif media_type == MediaType.VIDEO:
                # Analyze artifacts in sample frames
                frames = media_data
                if not frames:
                    return {'error': 'No frames to analyze'}

                sample_frames = frames[::max(1, len(frames)//3)]  # Sample every 3rd frame
                frame_artifacts = []

                for i, frame in enumerate(sample_frames):
                    artifacts = self.artifact_analyzer.analyze_all_artifacts(frame)
                    frame_artifacts.append({
                        'frame_index': i * max(1, len(frames)//3),
                        'artifacts': artifacts
                    })

                # Aggregate results
                aggregated_artifacts = {}
                if frame_artifacts:
                    # Average artifact scores across frames
                    artifact_keys = frame_artifacts[0]['artifacts'].keys()
                    for key in artifact_keys:
                        values = [fa['artifacts'].get(key, 0) for fa in frame_artifacts if isinstance(fa['artifacts'].get(key), (int, float))]
                        if values:
                            aggregated_artifacts[key] = float(np.mean(values))

                return {
                    'aggregated_artifacts': aggregated_artifacts,
                    'frame_analysis': frame_artifacts,
                    'frames_analyzed': len(sample_frames)
                }

            elif media_type == MediaType.AUDIO:
                # Audio artifact analysis (simplified)
                audio_data, sample_rate = media_data

                artifacts = {
                    'clipping_detected': float(np.max(np.abs(audio_data)) >= 0.99),
                    'silence_ratio': float(np.sum(np.abs(audio_data) < 0.01) / len(audio_data)),
                    'dynamic_range': float(20 * np.log10(np.max(np.abs(audio_data)) / (np.mean(np.abs(audio_data)) + 1e-10))),
                    'spectral_artifacts': self._detect_audio_spectral_artifacts(audio_data, sample_rate)
                }

                return artifacts

            return {'error': f'Artifact analysis not supported for {media_type}'}

        except Exception as e:
            logger.error(f"Artifact analysis failed: {e}")
            return {'error': str(e)}

    def _detect_audio_spectral_artifacts(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Detect spectral artifacts in audio"""
        try:
            # Simple spectral analysis
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]

            # High frequency noise detection
            high_freq_mask = freqs > sample_rate * 0.4  # Above Nyquist * 0.8
            high_freq_energy = np.sum(magnitude[high_freq_mask]) if np.sum(high_freq_mask) > 0 else 0
            total_energy = np.sum(magnitude)

            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)

            # Spectral flatness
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

            return {
                'high_frequency_noise': float(high_freq_ratio),
                'spectral_flatness': float(spectral_flatness),
                'spectral_centroid': float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10))
            }

        except Exception as e:
            logger.debug(f"Spectral artifact detection failed: {e}")
            return {}

    async def _analyze_statistics(self, media_data: Any, media_type: MediaType) -> Dict[str, Any]:
        """Analyze media statistics"""
        try:
            statistics = {}

            if media_type == MediaType.IMAGE:
                image = media_data

                # Basic image statistics
                statistics = {
                    'pixel_statistics': {
                        'min': float(np.min(image)),
                        'max': float(np.max(image)),
                        'mean': float(np.mean(image)),
                        'std': float(np.std(image)),
                        'median': float(np.median(image))
                    },
                    'histogram_stats': self._compute_histogram_stats(image),
                    'color_statistics': self._compute_color_stats(image) if len(image.shape) == 3 else {},
                    'spatial_statistics': self._compute_spatial_stats(image)
                }

            elif media_type == MediaType.VIDEO:
                frames = media_data

                if frames:
                    # Frame-level statistics
                    frame_means = [float(np.mean(frame)) for frame in frames]
                    frame_stds = [float(np.std(frame)) for frame in frames]

                    statistics = {
                        'frame_statistics': {
                            'count': len(frames),
                            'mean_brightness': {
                                'min': min(frame_means),
                                'max': max(frame_means),
                                'mean': float(np.mean(frame_means)),
                                'std': float(np.std(frame_means))
                            },
                            'frame_variability': {
                                'min': min(frame_stds),
                                'max': max(frame_stds),
                                'mean': float(np.mean(frame_stds)),
                                'std': float(np.std(frame_stds))
                            }
                        },
                        'temporal_statistics': self._compute_temporal_stats(frames)
                    }

            elif media_type == MediaType.AUDIO:
                audio_data, sample_rate = media_data

                statistics = {
                    'amplitude_statistics': {
                        'min': float(np.min(audio_data)),
                        'max': float(np.max(audio_data)),
                        'mean': float(np.mean(audio_data)),
                        'std': float(np.std(audio_data)),
                        'rms': float(np.sqrt(np.mean(audio_data**2)))
                    },
                    'spectral_statistics': self._compute_spectral_stats(audio_data, sample_rate),
                    'temporal_statistics': {
                        'duration': len(audio_data) / sample_rate,
                        'zero_crossings': int(np.sum(np.diff(np.sign(audio_data)) != 0)),
                        'energy': float(np.sum(audio_data**2))
                    }
                }

            return statistics

        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}

    def _compute_histogram_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Compute histogram-based statistics"""
        try:
            if len(image.shape) == 3:
                # Color image - compute for each channel
                hist_stats = {}
                for i, channel in enumerate(['red', 'green', 'blue']):
                    hist, _ = np.histogram(image[:,:,i], bins=256, range=(0, 256))
                    hist_stats[channel] = {
                        'entropy': float(-np.sum((hist / np.sum(hist)) * np.log2((hist / np.sum(hist)) + 1e-10))),
                        'peak_bin': int(np.argmax(hist)),
                        'uniformity': float(np.sum((hist / np.sum(hist))**2))
                    }
                return hist_stats
            else:
                # Grayscale image
                hist, _ = np.histogram(image, bins=256, range=(0, 256))
                return {
                    'entropy': float(-np.sum((hist / np.sum(hist)) * np.log2((hist / np.sum(hist)) + 1e-10))),
                    'peak_bin': int(np.argmax(hist)),
                    'uniformity': float(np.sum((hist / np.sum(hist))**2))
                }
        except Exception as e:
            logger.debug(f"Histogram stats computation failed: {e}")
            return {}

    def _compute_color_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Compute color-specific statistics"""
        try:
            if len(image.shape) != 3:
                return {}

            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            return {
                'dominant_colors': self._find_dominant_colors(image),
                'color_diversity': float(np.std(image.reshape(-1, 3), axis=0).mean()),
                'saturation_stats': {
                    'mean': float(np.mean(hsv[:,:,1])),
                    'std': float(np.std(hsv[:,:,1]))
                },
                'hue_stats': {
                    'mean': float(np.mean(hsv[:,:,0])),
                    'std': float(np.std(hsv[:,:,0]))
                }
            }
        except Exception as e:
            logger.debug(f"Color stats computation failed: {e}")
            return {}

    def _find_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[List[int]]:
        """Find dominant colors using K-means clustering"""
        try:
            from sklearn.cluster import KMeans

            # Reshape image to pixel array
            pixels = image.reshape(-1, 3)

            # Sample pixels for efficiency
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]

            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)

            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()

        except Exception as e:
            logger.debug(f"Dominant color extraction failed: {e}")
            return []

    def _compute_spatial_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Compute spatial statistics"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Local variance
            kernel = np.ones((5,5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)

            return {
                'edge_density': float(np.mean(gradient_magnitude > np.percentile(gradient_magnitude, 90))),
                'local_variance_mean': float(np.mean(local_variance)),
                'gradient_statistics': {
                    'mean': float(np.mean(gradient_magnitude)),
                    'std': float(np.std(gradient_magnitude)),
                    'max': float(np.max(gradient_magnitude))
                }
            }
        except Exception as e:
            logger.debug(f"Spatial stats computation failed: {e}")
            return {}

    def _compute_temporal_stats(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Compute temporal statistics for video"""
        try:
            if len(frames) < 2:
                return {}

            # Frame differences
            frame_diffs = []
            for i in range(len(frames) - 1):
                diff = np.mean(np.abs(frames[i+1].astype(np.float32) - frames[i].astype(np.float32)))
                frame_diffs.append(diff)

            return {
                'frame_differences': {
                    'mean': float(np.mean(frame_diffs)),
                    'std': float(np.std(frame_diffs)),
                    'min': float(min(frame_diffs)),
                    'max': float(max(frame_diffs))
                },
                'temporal_consistency': float(1.0 / (1.0 + np.std(frame_diffs))),
                'motion_intensity': float(np.mean(frame_diffs))
            }
        except Exception as e:
            logger.debug(f"Temporal stats computation failed: {e}")
            return {}

    def _compute_spectral_stats(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Compute spectral statistics for audio"""
        try:
            # FFT analysis
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]

            # Spectral centroid
            spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)

            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitude) / (np.sum(magnitude) + 1e-10))

            # Spectral rolloff (frequency below which 85% of energy lies)
            cumsum = np.cumsum(magnitude)
            rolloff_index = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            spectral_rolloff = freqs[rolloff_index[0]] if len(rolloff_index) > 0 else 0

            return {
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_flatness': float(np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)),
                'fundamental_frequency': float(freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0)
            }
        except Exception as e:
            logger.debug(f"Spectral stats computation failed: {e}")
            return {}

    async def _analyze_preprocessing(self, media_data: Any, media_type: MediaType) -> Dict[str, Any]:
        """Analyze preprocessing requirements and suggestions"""
        try:
            preprocessing_info = {
                'recommendations': [],
                'detected_issues': [],
                'preprocessing_steps': []
            }

            if media_type == MediaType.IMAGE:
                image = media_data

                # Check image properties
                if len(image.shape) == 3 and image.shape[2] == 4:
                    preprocessing_info['detected_issues'].append('Image has alpha channel')
                    preprocessing_info['recommendations'].append('Remove alpha channel for processing')

                # Check resolution
                height, width = image.shape[:2]
                if width > 1920 or height > 1920:
                    preprocessing_info['detected_issues'].append('High resolution image')
                    preprocessing_info['recommendations'].append('Consider resizing for efficiency')

                # Check brightness
                brightness = np.mean(image)
                if brightness < 50:
                    preprocessing_info['detected_issues'].append('Low brightness')
                    preprocessing_info['recommendations'].append('Apply brightness enhancement')
                elif brightness > 200:
                    preprocessing_info['detected_issues'].append('High brightness')
                    preprocessing_info['recommendations'].append('Apply brightness reduction')

                # Check contrast
                contrast = np.std(image)
                if contrast < 20:
                    preprocessing_info['detected_issues'].append('Low contrast')
                    preprocessing_info['recommendations'].append('Apply contrast enhancement')

                preprocessing_info['preprocessing_steps'] = [
                    'Normalize pixel values to [0, 1]',
                    'Resize to standard dimensions (224x224 or 299x299)',
                    'Apply data augmentation if needed for training'
                ]

            elif media_type == MediaType.VIDEO:
                frames = media_data

                if len(frames) > 100:
                    preprocessing_info['detected_issues'].append('High frame count')
                    preprocessing_info['recommendations'].append('Sample frames or use temporal compression')

                if len(frames) < 10:
                    preprocessing_info['detected_issues'].append('Low frame count')
                    preprocessing_info['recommendations'].append('Consider frame interpolation')

                preprocessing_info['preprocessing_steps'] = [
                    'Extract key frames or uniform sampling',
                    'Normalize frame pixel values',
                    'Resize frames to standard dimensions',
                    'Apply temporal alignment if needed'
                ]

            elif media_type == MediaType.AUDIO:
                audio_data, sample_rate = media_data
                duration = len(audio_data) / sample_rate

                if sample_rate < 16000:
                    preprocessing_info['detected_issues'].append('Low sample rate')
                    preprocessing_info['recommendations'].append('Upsample to at least 16kHz')

                if duration > 60:
                    preprocessing_info['detected_issues'].append('Long audio duration')
                    preprocessing_info['recommendations'].append('Segment into smaller chunks')

                if duration < 1:
                    preprocessing_info['detected_issues'].append('Very short audio')
                    preprocessing_info['recommendations'].append('Ensure minimum duration for analysis')

                preprocessing_info['preprocessing_steps'] = [
                    'Normalize audio amplitude',
                    'Apply pre-emphasis filter',
                    'Extract spectrograms or MFCC features',
                    'Apply voice activity detection if needed'
                ]

            return preprocessing_info

        except Exception as e:
            logger.error(f"Preprocessing analysis failed: {e}")
            return {'error': str(e)}

    async def _generate_visual_outputs(self, media_data: Any, media_type: MediaType, 
                                     request: AnalysisRequest, result: AnalysisResult):
        """Generate thumbnails and extract frames"""
        try:
            if request.include_thumbnails:
                result.thumbnails = []

                if media_type == MediaType.IMAGE:
                    # Create thumbnail
                    thumbnail_path = await self._create_image_thumbnail(media_data, result.request_id)
                    if thumbnail_path:
                        result.thumbnails.append(thumbnail_path)

                elif media_type == MediaType.VIDEO:
                    # Create thumbnails from key frames
                    frames = media_data
                    if frames:
                        key_frame_indices = np.linspace(0, len(frames)-1, min(3, len(frames)), dtype=int)
                        for i, frame_idx in enumerate(key_frame_indices):
                            thumbnail_path = await self._create_image_thumbnail(
                                frames[frame_idx], f"{result.request_id}_frame_{i}"
                            )
                            if thumbnail_path:
                                result.thumbnails.append(thumbnail_path)

            if request.extract_frames and media_type == MediaType.VIDEO:
                result.extracted_frames = []
                frames = media_data
                if frames:
                    extract_count = min(request.max_frames, len(frames))
                    frame_indices = np.linspace(0, len(frames)-1, extract_count, dtype=int)

                    for i, frame_idx in enumerate(frame_indices):
                        frame_path = await self._save_extracted_frame(
                            frames[frame_idx], f"{result.request_id}_extracted_{i}"
                        )
                        if frame_path:
                            result.extracted_frames.append(frame_path)

        except Exception as e:
            logger.error(f"Visual output generation failed: {e}")
            if result.warnings is None:
                result.warnings = []
            result.warnings.append(f"Visual output generation failed: {str(e)}")

    async def _create_image_thumbnail(self, image: np.ndarray, identifier: str) -> Optional[str]:
        """Create thumbnail image"""
        try:
            # Resize image to thumbnail size
            thumbnail_size = (150, 150)
            h, w = image.shape[:2]

            # Calculate scaling to maintain aspect ratio
            scale = min(thumbnail_size[0]/w, thumbnail_size[1]/h)
            new_w, new_h = int(w * scale), int(h * scale)

            thumbnail = cv2.resize(image, (new_w, new_h))

            # Create canvas and center thumbnail
            canvas = np.zeros((thumbnail_size[1], thumbnail_size[0], 3), dtype=np.uint8)
            y_offset = (thumbnail_size[1] - new_h) // 2
            x_offset = (thumbnail_size[0] - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail

            # Save thumbnail
            thumbnail_path = self.temp_dir / f"thumbnail_{identifier}.jpg"
            success = self.file_handler.save_image(canvas, thumbnail_path, quality=85)

            return str(thumbnail_path) if success else None

        except Exception as e:
            logger.debug(f"Thumbnail creation failed: {e}")
            return None

    async def _save_extracted_frame(self, frame: np.ndarray, identifier: str) -> Optional[str]:
        """Save extracted frame"""
        try:
            frame_path = self.temp_dir / f"frame_{identifier}.jpg"
            success = self.file_handler.save_image(frame, frame_path, quality=95)
            return str(frame_path) if success else None

        except Exception as e:
            logger.debug(f"Frame extraction failed: {e}")
            return None

    async def _check_cache(self, request: AnalysisRequest) -> Optional[AnalysisResult]:
        """Check if analysis result is cached"""
        try:
            if not self.result_cache:
                return None

            # Create cache key
            cache_key = hashlib.md5(
                f"{request.file_path}_{sorted([at.value for at in request.analysis_types])}".encode()
            ).hexdigest()

            return self.result_cache.get(cache_key)

        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
            return None

    async def _cache_result(self, request: AnalysisRequest, result: AnalysisResult):
        """Cache analysis result"""
        try:
            if not self.result_cache:
                return

            # Create cache key
            cache_key = hashlib.md5(
                f"{request.file_path}_{sorted([at.value for at in request.analysis_types])}".encode()
            ).hexdigest()

            # Cache result (limit cache size)
            if len(self.result_cache) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]

            self.result_cache[cache_key] = result

        except Exception as e:
            logger.debug(f"Result caching failed: {e}")

    def _update_stats(self, result: AnalysisResult):
        """Update analysis statistics"""
        try:
            self.analysis_stats['total_analyses'] += 1

            if result.error_message:
                self.analysis_stats['failed_analyses'] += 1
            else:
                self.analysis_stats['successful_analyses'] += 1

                # Update average processing time
                current_avg = self.analysis_stats['average_processing_time']
                total_successful = self.analysis_stats['successful_analyses']

                new_avg = ((current_avg * (total_successful - 1)) + result.processing_time) / total_successful
                self.analysis_stats['average_processing_time'] = new_avg

        except Exception as e:
            logger.error(f"Stats update failed: {e}")

    async def batch_analyze(self, requests: List[AnalysisRequest]) -> List[AnalysisResult]:
        """Process multiple analysis requests concurrently"""
        try:
            logger.info(f"Starting batch analysis for {len(requests)} requests")

            # Process requests concurrently
            tasks = [self.analyze_media(request) for request in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis {i} failed: {result}")
                    final_results.append(AnalysisResult(
                        request_id=requests[i].request_id or f"batch_{i}",
                        file_path=requests[i].file_path,
                        media_type=MediaType.UNKNOWN,
                        file_info={},
                        error_message=str(result)
                    ))
                else:
                    final_results.append(result)

            logger.info(f"Batch analysis completed: {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return [AnalysisResult(
                request_id=req.request_id or f"batch_{i}",
                file_path=req.file_path,
                media_type=MediaType.UNKNOWN,
                file_info={},
                error_message=str(e)
            ) for i, req in enumerate(requests)]

    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        stats = self.analysis_stats.copy()
        stats['success_rate'] = (
            stats['successful_analyses'] / max(stats['total_analyses'], 1)
        )
        stats['cache_size'] = len(self.result_cache) if self.result_cache else 0
        return stats

    def clear_cache(self):
        """Clear result cache"""
        if self.result_cache:
            self.result_cache.clear()
            logger.info("Analysis cache cleared")

    def cleanup_temp_files(self) -> int:
        """Cleanup temporary files"""
        try:
            return self.file_handler.cleanup_temp_files(older_than_hours=24)
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")
            return 0
