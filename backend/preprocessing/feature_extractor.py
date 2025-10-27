"""
Feature Extractor for Multi-Modal Deepfake Detection
Unified feature extraction interface for image, video, and audio modalities
Supports cross-modal feature analysis and temporal feature extraction
"""

import numpy as np
import torch
import torch.nn as nn
import logging
import time
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Import preprocessors
from .image_preprocessor import ImagePreprocessor
from .video_preprocessor import VideoPreprocessor  
from .audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Unified feature extractor for multi-modal deepfake detection
    Coordinates feature extraction across image, video, and audio modalities
    """

    def __init__(self, 
                 device: str = 'cpu',
                 image_size: Tuple[int, int] = (224, 224),
                 frames_per_clip: int = 16,
                 audio_sample_rate: int = 16000):
        """
        Initialize Feature Extractor

        Args:
            device: Device for processing
            image_size: Target image size
            frames_per_clip: Number of frames per video clip
            audio_sample_rate: Audio sample rate
        """
        self.device = device
        self.image_size = image_size
        self.frames_per_clip = frames_per_clip
        self.audio_sample_rate = audio_sample_rate

        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(
            target_size=image_size, 
            device=device
        )

        self.video_preprocessor = VideoPreprocessor(
            target_size=image_size,
            frames_per_clip=frames_per_clip,
            device=device
        )

        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=audio_sample_rate,
            device=device
        )

        # Initialize feature extraction networks (placeholder)
        self._init_feature_networks()

        logger.info(f"Initialized FeatureExtractor on {device}")

    def _init_feature_networks(self):
        """Initialize feature extraction networks for each modality"""

        # Image feature extractor (CNN backbone)
        self.image_feature_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ResNet-like blocks
            self._make_layer(64, 128, 2, stride=1),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256)
        )

        # Video temporal feature extractor
        self.video_temporal_net = nn.LSTM(
            input_size=256,  # Features from image network
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Audio feature extractor (1D CNN for spectrograms)
        self.audio_feature_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 256)
        )

        # Move networks to device
        self.image_feature_net.to(self.device)
        self.video_temporal_net.to(self.device)
        self.audio_feature_net.to(self.device)

    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Create ResNet-like layer"""
        layers = []

        # First block with potential downsampling
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Additional blocks
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def extract_image_features(self, 
                              image_input: Union[torch.Tensor, str, Path],
                              extract_artifacts: bool = True) -> Dict[str, torch.Tensor]:
        """
        Extract features from images

        Args:
            image_input: Image tensor or path to image
            extract_artifacts: Whether to extract artifact features

        Returns:
            Dictionary of extracted features
        """
        if isinstance(image_input, (str, Path)):
            # Preprocess image from path
            processed = self.image_preprocessor.preprocess_single(image_input)
            image_tensor = processed['processed_image'].unsqueeze(0)  # Add batch dimension
        else:
            image_tensor = image_input
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            # Extract deep features
            deep_features = self.image_feature_net(image_tensor)

            features = {
                'deep_features': deep_features,
                'spatial_size': image_tensor.shape[-2:],
                'feature_dim': deep_features.shape[-1]
            }

            if extract_artifacts:
                # Extract artifact-specific features
                artifact_features = self._extract_image_artifacts(image_tensor)
                features.update(artifact_features)

        return features

    def _extract_image_artifacts(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract image artifact features"""

        # Convert to numpy for analysis
        if image_tensor.dim() == 4:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Convert to 0-255 range if normalized
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        artifacts = {}

        # Blur detection using Laplacian variance
        gray = np.mean(image_np, axis=2).astype(np.uint8)
        laplacian_var = np.var(np.gradient(gray))
        artifacts['blur_score'] = torch.tensor(laplacian_var / 1000.0).float()  # Normalize

        # Noise analysis using high-frequency energy
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        h, w = magnitude.shape

        # High frequency mask (outer regions)
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[:h//4, :] = 1
        high_freq_mask[3*h//4:, :] = 1
        high_freq_mask[:, :w//4] = 1
        high_freq_mask[:, 3*w//4:] = 1

        high_freq_energy = np.sum(magnitude * high_freq_mask)
        total_energy = np.sum(magnitude)
        artifacts['noise_score'] = torch.tensor(high_freq_energy / total_energy).float()

        # Compression artifacts (JPEG blocking)
        block_score = self._detect_blocking_artifacts(gray)
        artifacts['compression_artifacts'] = torch.tensor(block_score).float()

        return artifacts

    def _detect_blocking_artifacts(self, gray_image: np.ndarray, block_size: int = 8) -> float:
        """Detect JPEG-like blocking artifacts"""
        h, w = gray_image.shape

        # Calculate gradients at block boundaries
        block_gradients = []

        # Vertical block boundaries
        for i in range(block_size, w, block_size):
            if i < w - 1:
                grad = np.mean(np.abs(gray_image[:, i] - gray_image[:, i-1]))
                block_gradients.append(grad)

        # Horizontal block boundaries  
        for i in range(block_size, h, block_size):
            if i < h - 1:
                grad = np.mean(np.abs(gray_image[i, :] - gray_image[i-1, :]))
                block_gradients.append(grad)

        # Compare with random boundaries
        random_gradients = []
        for _ in range(len(block_gradients)):
            i = np.random.randint(1, min(h, w) - 1)
            if np.random.choice([True, False]):  # Vertical or horizontal
                if i < w - 1:
                    grad = np.mean(np.abs(gray_image[:, i] - gray_image[:, i-1]))
                else:
                    grad = np.mean(np.abs(gray_image[i, :] - gray_image[i-1, :]))
            else:
                if i < h - 1:
                    grad = np.mean(np.abs(gray_image[i, :] - gray_image[i-1, :]))
                else:
                    grad = np.mean(np.abs(gray_image[:, i] - gray_image[:, i-1]))
            random_gradients.append(grad)

        # Blocking score based on difference
        if len(block_gradients) > 0 and len(random_gradients) > 0:
            block_mean = np.mean(block_gradients)
            random_mean = np.mean(random_gradients)
            blocking_score = max(0, block_mean - random_mean) / (block_mean + 1e-8)
            return min(blocking_score, 1.0)

        return 0.0

    def extract_video_features(self, 
                              video_input: Union[torch.Tensor, str, Path],
                              extract_temporal: bool = True) -> Dict[str, torch.Tensor]:
        """
        Extract features from videos

        Args:
            video_input: Video tensor or path to video
            extract_temporal: Whether to extract temporal features

        Returns:
            Dictionary of extracted features
        """
        if isinstance(video_input, (str, Path)):
            # Preprocess video from path
            processed = self.video_preprocessor.preprocess_single(video_input)
            video_tensor = processed['processed_frames'].unsqueeze(0)  # Add batch dimension
        else:
            video_tensor = video_input
            if video_tensor.dim() == 4:  # [T, C, H, W] -> [1, T, C, H, W]
                video_tensor = video_tensor.unsqueeze(0)

        video_tensor = video_tensor.to(self.device)
        batch_size, num_frames, channels, height, width = video_tensor.shape

        with torch.no_grad():
            # Extract frame-level features
            frame_features = []

            for i in range(num_frames):
                frame = video_tensor[:, i, :, :, :]  # [B, C, H, W]
                frame_feat = self.image_feature_net(frame)  # [B, 256]
                frame_features.append(frame_feat)

            frame_features = torch.stack(frame_features, dim=1)  # [B, T, 256]

            features = {
                'frame_features': frame_features,
                'num_frames': num_frames,
                'spatial_size': (height, width)
            }

            if extract_temporal:
                # Extract temporal features using LSTM
                temporal_output, (hidden, cell) = self.video_temporal_net(frame_features)
                temporal_features = temporal_output[:, -1, :]  # Use last output

                # Temporal consistency analysis
                temporal_consistency = self._analyze_temporal_consistency(frame_features)

                features.update({
                    'temporal_features': temporal_features,
                    'temporal_consistency': temporal_consistency,
                    'temporal_dim': temporal_features.shape[-1]
                })

        return features

    def _analyze_temporal_consistency(self, frame_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze temporal consistency in video features"""

        # Calculate frame-to-frame differences
        frame_diffs = torch.diff(frame_features, dim=1)  # [B, T-1, D]

        # Temporal variance
        temporal_var = torch.var(frame_features, dim=1)  # [B, D]
        temporal_var_mean = torch.mean(temporal_var, dim=1)  # [B]

        # Sudden change detection
        diff_magnitudes = torch.norm(frame_diffs, dim=2)  # [B, T-1]
        sudden_changes = (diff_magnitudes > torch.mean(diff_magnitudes, dim=1, keepdim=True) + 2 * torch.std(diff_magnitudes, dim=1, keepdim=True)).float()
        sudden_change_ratio = torch.mean(sudden_changes, dim=1)  # [B]

        # Motion smoothness
        motion_smoothness = 1.0 / (1.0 + torch.mean(diff_magnitudes, dim=1))  # [B]

        return {
            'temporal_variance': temporal_var_mean,
            'sudden_change_ratio': sudden_change_ratio,
            'motion_smoothness': motion_smoothness,
            'frame_differences': torch.mean(diff_magnitudes, dim=1)
        }

    def extract_audio_features(self, 
                              audio_input: Union[torch.Tensor, str, Path],
                              extract_spectral: bool = True) -> Dict[str, torch.Tensor]:
        """
        Extract features from audio

        Args:
            audio_input: Audio tensor or path to audio
            extract_spectral: Whether to extract spectral features

        Returns:
            Dictionary of extracted features
        """
        if isinstance(audio_input, (str, Path)):
            # Preprocess audio from path
            processed = self.audio_preprocessor.preprocess_single(audio_input, extract_features=True)
            audio_tensor = processed['processed_audio']
        else:
            audio_tensor = audio_input

        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

        audio_tensor = audio_tensor.to(self.device)

        with torch.no_grad():
            # Extract deep audio features
            deep_features = self.audio_feature_net(audio_tensor)

            features = {
                'deep_features': deep_features,
                'audio_length': audio_tensor.shape[-1],
                'feature_dim': deep_features.shape[-1]
            }

            if extract_spectral:
                # Extract spectral and voice features
                spectral_features = self._extract_audio_spectral_features(audio_tensor)
                features.update(spectral_features)

        return features

    def _extract_audio_spectral_features(self, audio_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract spectral features from audio tensor"""

        # Convert to numpy for librosa analysis
        audio_np = audio_tensor.squeeze().cpu().numpy()

        spectral_features = {}

        try:
            import librosa

            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_np, sr=self.audio_sample_rate
            )[0]
            spectral_features['spectral_centroid'] = torch.tensor(np.mean(spectral_centroids)).float()

            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_np, sr=self.audio_sample_rate
            )[0]
            spectral_features['spectral_rolloff'] = torch.tensor(np.mean(spectral_rolloff)).float()

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_np)[0]
            spectral_features['zero_crossing_rate'] = torch.tensor(np.mean(zcr)).float()

            # RMS energy
            rms = librosa.feature.rms(y=audio_np)[0]
            spectral_features['rms_energy'] = torch.tensor(np.mean(rms)).float()

            # Pitch features (simplified)
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                    sr=self.audio_sample_rate
                )
                valid_f0 = f0[~np.isnan(f0)]

                if len(valid_f0) > 0:
                    spectral_features['pitch_mean'] = torch.tensor(np.mean(valid_f0)).float()
                    spectral_features['pitch_std'] = torch.tensor(np.std(valid_f0)).float()
                    spectral_features['voiced_ratio'] = torch.tensor(np.mean(voiced_flag)).float()
                else:
                    spectral_features['pitch_mean'] = torch.tensor(0.0).float()
                    spectral_features['pitch_std'] = torch.tensor(0.0).float()
                    spectral_features['voiced_ratio'] = torch.tensor(0.0).float()

            except Exception as e:
                logger.warning(f"Pitch extraction failed: {e}")
                spectral_features.update({
                    'pitch_mean': torch.tensor(0.0).float(),
                    'pitch_std': torch.tensor(0.0).float(),
                    'voiced_ratio': torch.tensor(0.0).float()
                })

        except ImportError:
            logger.warning("librosa not available for spectral feature extraction")
            # Provide dummy features
            spectral_features = {
                'spectral_centroid': torch.tensor(0.5).float(),
                'spectral_rolloff': torch.tensor(0.5).float(),
                'zero_crossing_rate': torch.tensor(0.1).float(),
                'rms_energy': torch.tensor(0.1).float(),
                'pitch_mean': torch.tensor(200.0).float(),
                'pitch_std': torch.tensor(50.0).float(),
                'voiced_ratio': torch.tensor(0.5).float()
            }

        return spectral_features

    def extract_multimodal_features(self, 
                                   image_input: Optional[Union[torch.Tensor, str, Path]] = None,
                                   video_input: Optional[Union[torch.Tensor, str, Path]] = None,
                                   audio_input: Optional[Union[torch.Tensor, str, Path]] = None) -> Dict[str, Any]:
        """
        Extract features from multiple modalities

        Args:
            image_input: Image input (optional)
            video_input: Video input (optional)
            audio_input: Audio input (optional)

        Returns:
            Dictionary with features from all provided modalities
        """
        multimodal_features = {
            'modalities_present': [],
            'feature_dimensions': {}
        }

        # Extract image features
        if image_input is not None:
            try:
                image_features = self.extract_image_features(image_input)
                multimodal_features['image'] = image_features
                multimodal_features['modalities_present'].append('image')
                multimodal_features['feature_dimensions']['image'] = image_features['feature_dim']
            except Exception as e:
                logger.warning(f"Image feature extraction failed: {e}")
                multimodal_features['image'] = {'error': str(e)}

        # Extract video features
        if video_input is not None:
            try:
                video_features = self.extract_video_features(video_input)
                multimodal_features['video'] = video_features
                multimodal_features['modalities_present'].append('video')
                multimodal_features['feature_dimensions']['video'] = video_features.get('temporal_dim', 256)
            except Exception as e:
                logger.warning(f"Video feature extraction failed: {e}")
                multimodal_features['video'] = {'error': str(e)}

        # Extract audio features
        if audio_input is not None:
            try:
                audio_features = self.extract_audio_features(audio_input)
                multimodal_features['audio'] = audio_features
                multimodal_features['modalities_present'].append('audio')
                multimodal_features['feature_dimensions']['audio'] = audio_features['feature_dim']
            except Exception as e:
                logger.warning(f"Audio feature extraction failed: {e}")
                multimodal_features['audio'] = {'error': str(e)}

        return multimodal_features

    def compute_cross_modal_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between features from different modalities

        Args:
            features1: Features from first modality
            features2: Features from second modality

        Returns:
            Cosine similarity score
        """
        # Normalize features
        features1_norm = nn.functional.normalize(features1, p=2, dim=-1)
        features2_norm = nn.functional.normalize(features2, p=2, dim=-1)

        # Compute cosine similarity
        similarity = torch.sum(features1_norm * features2_norm, dim=-1)

        return similarity

    def analyze_cross_modal_consistency(self, multimodal_features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Analyze consistency between different modalities

        Args:
            multimodal_features: Features from multiple modalities

        Returns:
            Cross-modal consistency metrics
        """
        consistency_metrics = {}
        modalities = multimodal_features['modalities_present']

        # Image-Video consistency
        if 'image' in modalities and 'video' in modalities:
            image_feats = multimodal_features['image']['deep_features']
            # Use mean frame features from video
            video_frame_feats = multimodal_features['video']['frame_features'].mean(dim=1)

            img_video_similarity = self.compute_cross_modal_similarity(image_feats, video_frame_feats)
            consistency_metrics['image_video_consistency'] = img_video_similarity

        # Audio-Visual consistency (simplified)
        if 'audio' in modalities and ('image' in modalities or 'video' in modalities):
            audio_feats = multimodal_features['audio']['deep_features']

            if 'image' in modalities:
                visual_feats = multimodal_features['image']['deep_features']
            else:
                visual_feats = multimodal_features['video']['frame_features'].mean(dim=1)

            # This is a simplified consistency check
            # In practice, you'd want more sophisticated audio-visual alignment
            av_consistency = self.compute_cross_modal_similarity(audio_feats, visual_feats)
            consistency_metrics['audio_visual_consistency'] = av_consistency

        return consistency_metrics

    def get_feature_summary(self, multimodal_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of extracted features

        Args:
            multimodal_features: Multi-modal features

        Returns:
            Feature summary
        """
        summary = {
            'modalities_processed': multimodal_features['modalities_present'],
            'total_modalities': len(multimodal_features['modalities_present']),
            'feature_dimensions': multimodal_features['feature_dimensions'],
            'extraction_status': {}
        }

        # Check extraction status for each modality
        for modality in ['image', 'video', 'audio']:
            if modality in multimodal_features:
                if 'error' in multimodal_features[modality]:
                    summary['extraction_status'][modality] = 'failed'
                else:
                    summary['extraction_status'][modality] = 'success'
            else:
                summary['extraction_status'][modality] = 'not_provided'

        # Compute cross-modal consistency if multiple modalities
        if len(multimodal_features['modalities_present']) > 1:
            consistency = self.analyze_cross_modal_consistency(multimodal_features)
            summary['cross_modal_consistency'] = consistency

        return summary

    def save_features(self, features: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save extracted features to file

        Args:
            features: Extracted features
            output_path: Path to save features
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to numpy for saving
        features_to_save = {}

        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj

        features_to_save = convert_tensors(features)

        # Save as compressed numpy file
        np.savez_compressed(output_path, **features_to_save)

        logger.info(f"Features saved to {output_path}")

    def load_features(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load features from file

        Args:
            input_path: Path to load features from

        Returns:
            Loaded features
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Features file not found: {input_path}")

        # Load numpy file
        loaded = np.load(input_path, allow_pickle=True)

        # Convert back to appropriate format
        features = {}
        for key, value in loaded.items():
            if isinstance(value, np.ndarray) and value.dtype == object:
                # Handle nested structures
                features[key] = value.item()
            else:
                features[key] = value

        logger.info(f"Features loaded from {input_path}")

        return features
