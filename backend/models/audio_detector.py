"""
Audio Deepfake Detector
Specialized model for detecting manipulated audio using ECAPA-TDNN and Wav2Vec 2.0
Supports voice cloning, pitch modulation, and spoofed audio detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
from typing import Dict, Tuple, Union, Optional, List
import logging
from pathlib import Path
import warnings

from .base_model import BaseDetectionModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

class AudioDetector(BaseDetectionModel):
    """
    Audio-based deepfake detection model
    Supports multiple audio analysis techniques and spectral features
    """

    def __init__(self,
                 backbone: str = 'ecapa_tdnn',
                 num_classes: int = 2,
                 device: str = 'cpu',
                 pretrained: bool = True,
                 sample_rate: int = 16000,
                 duration: int = 10):
        """
        Initialize Audio Detector

        Args:
            backbone: Backbone architecture ('ecapa_tdnn', 'wav2vec2', 'crnn')
            num_classes: Number of classes
            device: Device to run on
            pretrained: Use pretrained weights
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
        """
        super().__init__(f'audio_detector_{backbone}', num_classes, device, pretrained)

        self.backbone_name = backbone
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = sample_rate * duration
        self.input_shape = (1, self.n_samples)

        # Audio processing parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.n_mfcc = 40

        # Build model
        self.build_model()

        # Move to device
        self.to(self.device)

        # Initialize audio transforms
        self._init_transforms()

        logger.info(f"Initialized AudioDetector with {backbone} backbone")

    def build_model(self) -> None:
        """Build the audio detection model architecture"""

        if self.backbone_name == 'ecapa_tdnn':
            self.backbone = self._build_ecapa_tdnn()
            self.feature_dim = 512

        elif self.backbone_name == 'wav2vec2':
            self.backbone = self._build_wav2vec2_like()
            self.feature_dim = 768

        elif self.backbone_name == 'crnn':
            self.backbone = self._build_crnn()
            self.feature_dim = 512

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Spectral feature extractors
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'power': 2.0
            }
        )

        # Spectral analysis branch
        self.spectral_analyzer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )

    def _build_ecapa_tdnn(self) -> nn.Module:
        """Build ECAPA-TDNN like architecture"""
        return nn.Sequential(
            # Frame-level processing
            nn.Conv1d(1, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # TDNN blocks
            self._make_tdnn_block(512, 512, kernel_size=3, dilation=1),
            self._make_tdnn_block(512, 512, kernel_size=3, dilation=2),
            self._make_tdnn_block(512, 512, kernel_size=3, dilation=3),
            self._make_tdnn_block(512, 512, kernel_size=1, dilation=1),

            # Attention pooling
            self._make_attention_pooling(512),

            # Final projection
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

    def _make_tdnn_block(self, in_channels: int, out_channels: int, 
                        kernel_size: int, dilation: int) -> nn.Module:
        """Create TDNN block with residual connection"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                     dilation=dilation, padding=dilation * (kernel_size - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_attention_pooling(self, channels: int) -> nn.Module:
        """Create attention-based pooling layer"""
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def _build_wav2vec2_like(self) -> nn.Module:
        """Build Wav2Vec2-like architecture"""
        return nn.Sequential(
            # Feature encoder (CNN)
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=2),
            nn.GroupNorm(num_groups=1, num_channels=512),
            nn.GELU(),

            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=512),
            nn.GELU(),

            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=512),
            nn.GELU(),

            # Transformer-like blocks (simplified)
            self._make_transformer_block(512, num_heads=8),
            self._make_transformer_block(512, num_heads=8),
            self._make_transformer_block(512, num_heads=8),

            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # Final projection
            nn.Linear(512, 768),
            nn.GELU()
        )

    def _make_transformer_block(self, d_model: int, num_heads: int) -> nn.Module:
        """Create transformer block"""
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

    def _build_crnn(self) -> nn.Module:
        """Build CNN-RNN hybrid architecture"""
        return nn.Sequential(
            # CNN feature extraction from spectrogram
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Reshape for RNN
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(start_dim=1, end_dim=2),

            # BiLSTM
            nn.LSTM(256, 256, num_layers=2, batch_first=True, 
                   bidirectional=True, dropout=0.3),

            # Take last output
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 512)  # 256 * 2 for bidirectional
        )

    def _init_transforms(self) -> None:
        """Initialize audio preprocessing transforms"""
        # Resampling transform
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=None,  # Will be set dynamically
            new_freq=self.sample_rate
        )

        # Normalization
        self.normalizer = lambda x: (x - x.mean()) / (x.std() + 1e-8)

    def preprocess_input(self, input_data) -> torch.Tensor:
        """
        Preprocess input audio data

        Args:
            input_data: Input audio (file path, numpy array, or tensor)

        Returns:
            Preprocessed audio tensor
        """
        if isinstance(input_data, str):
            # Load audio from file
            waveform, orig_sr = torchaudio.load(input_data)
        elif isinstance(input_data, np.ndarray):
            # Convert numpy to tensor
            waveform = torch.from_numpy(input_data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            orig_sr = self.sample_rate  # Assume correct sample rate
        elif isinstance(input_data, torch.Tensor):
            waveform = input_data.float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            orig_sr = self.sample_rate
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Resample if needed
        if isinstance(input_data, str) and orig_sr != self.sample_rate:
            self.resampler.orig_freq = orig_sr
            waveform = self.resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or trim to desired length
        current_length = waveform.shape[1]
        if current_length < self.n_samples:
            # Pad with zeros
            pad_length = self.n_samples - current_length
            waveform = F.pad(waveform, (0, pad_length))
        elif current_length > self.n_samples:
            # Trim from center
            start = (current_length - self.n_samples) // 2
            waveform = waveform[:, start:start + self.n_samples]

        # Normalize
        waveform = self.normalizer(waveform)

        # Add batch dimension if needed
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.unsqueeze(0)  # [1, 1, n_samples]

        return waveform

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio input

        Args:
            x: Input audio tensor [batch_size, 1, n_samples]

        Returns:
            Extracted features
        """
        batch_size = x.shape[0]

        # Extract raw audio features using backbone
        if self.backbone_name == 'crnn':
            # For CRNN, we need spectrogram input
            mel_spec = self.mel_spectrogram(x.squeeze(1))  # [batch, n_mels, time]
            mel_spec = torch.log(mel_spec + 1e-8).unsqueeze(1)  # Add channel dim and log scale
            audio_features = self.backbone(mel_spec)
        else:
            # For ECAPA-TDNN and Wav2Vec2, use raw waveform
            if x.dim() == 3 and x.shape[1] == 1:
                x_input = x.squeeze(1)  # Remove channel dimension for 1D conv
            else:
                x_input = x

            # Handle different backbone architectures
            if self.backbone_name == 'wav2vec2':
                # For transformer-based models, we need to handle sequence dimension
                audio_features = self.backbone(x_input)
            else:
                # For ECAPA-TDNN
                audio_features = self._extract_ecapa_features(x_input)

        # Extract spectral features for fusion
        mel_spec = self.mel_spectrogram(x.squeeze(1))
        mel_spec = torch.log(mel_spec + 1e-8).unsqueeze(1)
        spectral_features = self.spectral_analyzer(mel_spec)

        # Combine features
        combined_features = torch.cat([audio_features, spectral_features], dim=1)

        # Final feature fusion
        fused_features = self.feature_fusion(combined_features)

        return fused_features

    def _extract_ecapa_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using ECAPA-TDNN backbone"""
        # Process through TDNN layers
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, nn.Sequential):
                x = layer(x)
            elif isinstance(layer, nn.Linear):
                # Global pooling before linear layer
                if x.dim() == 3:
                    x = torch.mean(x, dim=2)  # Average over time dimension
                x = layer(x)
            elif hasattr(layer, 'weight') and layer.weight.dim() == 3:  # Conv1d layer
                x = layer(x)
            else:
                x = layer(x)

        return x

    def get_feature_breakdown(self, input_data) -> Dict[str, float]:
        """
        Get detailed feature breakdown for audio analysis

        Args:
            input_data: Input audio

        Returns:
            Dictionary with feature scores
        """
        self.eval()

        with torch.no_grad():
            # Load and preprocess audio
            if isinstance(input_data, str):
                waveform, sr = torchaudio.load(input_data)
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
            else:
                waveform = self.preprocess_input(input_data).squeeze(0)

            # Convert to numpy for analysis
            audio_np = waveform.squeeze().numpy()

            scores = {}

            # Spectral analysis using librosa
            try:
                # Spectral centroid (brightness)
                spectral_centroids = librosa.feature.spectral_centroid(
                    y=audio_np, sr=self.sample_rate
                )[0]
                scores['spectral_brightness'] = float(np.mean(spectral_centroids) / (self.sample_rate / 2))

                # Spectral bandwidth
                spectral_bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio_np, sr=self.sample_rate
                )[0]
                scores['spectral_bandwidth'] = float(np.mean(spectral_bandwidth) / (self.sample_rate / 2))

                # Zero crossing rate (voice quality)
                zcr = librosa.feature.zero_crossing_rate(audio_np)[0]
                scores['voice_quality'] = float(np.mean(zcr))

                # Pitch analysis
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                    sr=self.sample_rate
                )

                # Pitch stability (less variation = more synthetic)
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    pitch_variance = np.var(valid_f0)
                    scores['pitch_modulation'] = float(min(pitch_variance / 1000.0, 1.0))
                else:
                    scores['pitch_modulation'] = 0.0

                # Voice activity detection
                voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0
                scores['voice_activity'] = float(voiced_ratio)

            except Exception as e:
                logger.warning(f"Error in spectral analysis: {e}")
                scores.update({
                    'spectral_brightness': 0.5,
                    'spectral_bandwidth': 0.5,
                    'voice_quality': 0.5,
                    'pitch_modulation': 0.5,
                    'voice_activity': 0.5
                })

            # Noise analysis
            # High-frequency energy ratio (artifacts detection)
            try:
                stft = librosa.stft(audio_np)
                magnitude = np.abs(stft)

                # High frequency energy
                high_freq_start = int(0.8 * magnitude.shape[0])
                high_freq_energy = np.sum(magnitude[high_freq_start:, :])
                total_energy = np.sum(magnitude)
                scores['noise_score'] = float(high_freq_energy / total_energy) if total_energy > 0 else 0.0

                # Compression artifacts (spectral flatness)
                spectral_flatness = librosa.feature.spectral_flatness(y=audio_np)[0]
                scores['compression_artifacts'] = float(1.0 - np.mean(spectral_flatness))

            except Exception as e:
                logger.warning(f"Error in noise analysis: {e}")
                scores['noise_score'] = 0.5
                scores['compression_artifacts'] = 0.5

            # Temporal analysis
            scores['temporal_inconsistency'] = self._analyze_temporal_consistency(audio_np)

            # Facial warping not applicable for audio
            scores['facial_warping'] = 0.0
            scores['blur_score'] = 0.0  # Not applicable

            return scores

    def _analyze_temporal_consistency(self, audio_np: np.ndarray) -> float:
        """Analyze temporal consistency in audio"""
        try:
            # Divide audio into segments and analyze consistency
            segment_length = len(audio_np) // 10
            segment_features = []

            for i in range(10):
                start = i * segment_length
                end = min(start + segment_length, len(audio_np))
                segment = audio_np[start:end]

                if len(segment) > 0:
                    # Extract features for each segment
                    rms = np.sqrt(np.mean(segment**2))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
                    segment_features.append([rms, zcr])

            if len(segment_features) > 1:
                segment_features = np.array(segment_features)
                # Calculate variance across segments
                feature_variance = np.mean(np.var(segment_features, axis=0))
                return float(min(feature_variance, 1.0))
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error in temporal consistency analysis: {e}")
            return 0.5

    def analyze_voice_characteristics(self, input_data) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze voice characteristics for spoofing detection

        Args:
            input_data: Input audio

        Returns:
            Dictionary with voice analysis results
        """
        if isinstance(input_data, str):
            waveform, sr = torchaudio.load(input_data)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
        else:
            waveform = self.preprocess_input(input_data).squeeze(0)

        audio_np = waveform.squeeze().numpy()

        try:
            # Fundamental frequency analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )

            # Voice quality metrics
            valid_f0 = f0[~np.isnan(f0)]

            results = {
                'mean_pitch': float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0,
                'pitch_range': float(np.max(valid_f0) - np.min(valid_f0)) if len(valid_f0) > 0 else 0.0,
                'pitch_stability': float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0,
                'voiced_ratio': float(np.mean(voiced_flag)),
                'voice_confidence': float(np.mean(voiced_probs)),
            }

            # Formant analysis (simplified)
            mfccs = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate, n_mfcc=13)
            results['formant_stability'] = float(np.mean(np.std(mfccs, axis=1)))

            # Energy distribution
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=self.sample_rate)[0]
            results['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            results['spectral_centroid_std'] = float(np.std(spectral_centroid))

            return results

        except Exception as e:
            logger.warning(f"Error in voice characteristics analysis: {e}")
            return {
                'mean_pitch': 0.0,
                'pitch_range': 0.0,
                'pitch_stability': 0.0,
                'voiced_ratio': 0.0,
                'voice_confidence': 0.0,
                'formant_stability': 0.0,
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0
            }
