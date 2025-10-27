"""
Audio Preprocessor for Multi-Modal Deepfake Detection
Handles audio loading, preprocessing, spectral analysis, and voice feature extraction
Compatible with dataset structure: dataset/audio/{train,validation,test}/{FAKE,REAL}/
"""

import os
import numpy as np
import logging
import time
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import torchaudio
import torchaudio.transforms as T
from scipy import signal
import tempfile

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Comprehensive audio preprocessor for deepfake detection
    Supports audio loading, resampling, noise reduction, and feature extraction
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 duration: float = 10.0,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 40,
                 normalize: bool = True,
                 device: str = 'cpu'):
        """
        Initialize Audio Preprocessor

        Args:
            sample_rate: Target sample rate for audio
            duration: Target duration in seconds
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks
            n_mfcc: Number of MFCC coefficients
            normalize: Whether to normalize audio
            device: Device for processing
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.normalize = normalize
        self.device = device

        # Initialize transforms
        self._init_transforms()

        # Platform-specific configurations
        self.platform_configs = {
            'whatsapp': {
                'sample_rate': 8000,
                'bitrate': 64,
                'compression': 'opus',
                'duration_limit': 16 * 60,  # 16 minutes
                'noise_gate': -40  # dB
            },
            'youtube': {
                'sample_rate': 48000,
                'bitrate': 128,
                'compression': 'aac',
                'duration_limit': None,
                'noise_gate': -50
            },
            'zoom': {
                'sample_rate': 16000,
                'bitrate': 64,
                'compression': 'opus',
                'duration_limit': 40 * 60,  # 40 minutes
                'noise_gate': -35
            },
            'generic': {
                'sample_rate': 44100,
                'bitrate': 256,
                'compression': 'mp3',
                'duration_limit': None,
                'noise_gate': -60
            }
        }

        logger.info(f"Initialized AudioPreprocessor with {sample_rate}Hz, {duration}s duration")

    def _init_transforms(self):
        """Initialize audio transformation pipelines"""
        # Mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )

        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'power': 2.0
            }
        )

        # Resampler (will be initialized dynamically)
        self.resampler = None

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file with multiple fallback methods

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, original_sample_rate)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Try different loading methods
        audio_data = None
        orig_sr = None

        # Method 1: librosa (most robust)
        try:
            audio_data, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)
            logger.debug(f"Loaded audio with librosa: {audio_path}")
        except Exception as e:
            logger.warning(f"librosa loading failed: {e}")

        # Method 2: soundfile
        if audio_data is None:
            try:
                audio_data, orig_sr = sf.read(str(audio_path))
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                logger.debug(f"Loaded audio with soundfile: {audio_path}")
            except Exception as e:
                logger.warning(f"soundfile loading failed: {e}")

        # Method 3: torchaudio
        if audio_data is None:
            try:
                waveform, orig_sr = torchaudio.load(str(audio_path))
                audio_data = waveform.mean(dim=0).numpy()  # Convert to mono
                logger.debug(f"Loaded audio with torchaudio: {audio_path}")
            except Exception as e:
                logger.warning(f"torchaudio loading failed: {e}")

        if audio_data is None:
            raise ValueError(f"Could not load audio file: {audio_path}")

        return audio_data, orig_sr

    def resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate

        Args:
            audio_data: Input audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio data
        """
        if orig_sr == target_sr:
            return audio_data

        try:
            # Use librosa for resampling
            resampled_audio = librosa.resample(
                audio_data, 
                orig_sr=orig_sr, 
                target_sr=target_sr,
                res_type='kaiser_best'
            )
            return resampled_audio

        except Exception as e:
            logger.warning(f"librosa resampling failed: {e}")

            # Fallback to scipy
            try:
                resampled_audio = signal.resample(
                    audio_data, 
                    int(len(audio_data) * target_sr / orig_sr)
                )
                return resampled_audio
            except Exception as e:
                logger.error(f"Resampling failed: {e}")
                raise

    def normalize_audio(self, audio_data: np.ndarray, method: str = 'peak') -> np.ndarray:
        """
        Normalize audio data

        Args:
            audio_data: Input audio data
            method: Normalization method ('peak', 'rms', 'lufs')

        Returns:
            Normalized audio data
        """
        if method == 'peak':
            # Peak normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                return audio_data / max_val
            return audio_data

        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms > 0:
                return audio_data / rms * 0.1  # Scale to reasonable level
            return audio_data

        elif method == 'lufs':
            # Simple LUFS-like normalization
            # This is a simplified version - real LUFS requires more complex filtering
            squared = audio_data ** 2
            mean_square = np.mean(squared)
            if mean_square > 0:
                lufs_estimate = -0.691 + 10 * np.log10(mean_square)
                target_lufs = -23.0  # Broadcast standard
                gain_db = target_lufs - lufs_estimate
                gain_linear = 10 ** (gain_db / 20)
                return audio_data * gain_linear
            return audio_data

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def apply_noise_gate(self, audio_data: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        Apply noise gate to remove low-level noise

        Args:
            audio_data: Input audio data
            threshold_db: Threshold in dB below which audio is gated

        Returns:
            Gated audio data
        """
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio_data) + 1e-10)

        # Create gate mask
        gate_mask = audio_db > threshold_db

        # Apply gate with smooth transitions
        smoothed_mask = signal.savgol_filter(gate_mask.astype(float), 11, 3)

        return audio_data * smoothed_mask

    def pad_or_trim_audio(self, audio_data: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or trim audio to target length

        Args:
            audio_data: Input audio data
            target_length: Target length in samples

        Returns:
            Processed audio data
        """
        current_length = len(audio_data)

        if current_length == target_length:
            return audio_data

        elif current_length < target_length:
            # Pad audio
            pad_length = target_length - current_length

            # Pad with repetition if audio is very short
            if current_length < target_length // 4:
                repetitions = (target_length // current_length) + 1
                repeated_audio = np.tile(audio_data, repetitions)
                return repeated_audio[:target_length]
            else:
                # Pad with zeros
                return np.pad(audio_data, (0, pad_length), mode='constant')

        else:
            # Trim audio - take from center
            start_idx = (current_length - target_length) // 2
            return audio_data[start_idx:start_idx + target_length]

    def apply_platform_preprocessing(self, audio_data: np.ndarray, platform: str = 'generic') -> np.ndarray:
        """
        Apply platform-specific preprocessing

        Args:
            audio_data: Input audio data
            platform: Platform type

        Returns:
            Platform-processed audio
        """
        config = self.platform_configs.get(platform, self.platform_configs['generic'])

        # Apply noise gate
        if config['noise_gate'] is not None:
            audio_data = self.apply_noise_gate(audio_data, config['noise_gate'])

        # Simulate compression artifacts (simplified)
        if config['bitrate'] < 128:
            # Simulate low bitrate compression by adding quantization noise
            noise_level = (128 - config['bitrate']) / 1000.0
            noise = np.random.normal(0, noise_level, audio_data.shape)
            audio_data = audio_data + noise

        # Simulate sample rate conversion artifacts
        if config['sample_rate'] != self.sample_rate:
            # Downsample and upsample to simulate artifacts
            temp_audio = self.resample_audio(audio_data, self.sample_rate, config['sample_rate'])
            audio_data = self.resample_audio(temp_audio, config['sample_rate'], self.sample_rate)

        return audio_data

    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from audio

        Args:
            audio_data: Input audio data

        Returns:
            Dictionary of spectral features
        """
        features = {}

        try:
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)

            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['mfcc'] = mfccs

            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['chroma'] = chroma

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            features['spectral_centroids'] = spectral_centroids

            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            features['spectral_rolloff'] = spectral_rolloff

            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate
            )[0]
            features['spectral_bandwidth'] = spectral_bandwidth

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zero_crossing_rate'] = zcr

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            # Return basic features if advanced extraction fails
            features = {
                'mel_spectrogram': np.zeros((self.n_mels, 100)),
                'mfcc': np.zeros((self.n_mfcc, 100))
            }

        return features

    def extract_pitch_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract pitch and voice quality features

        Args:
            audio_data: Input audio data

        Returns:
            Dictionary of pitch features
        """
        features = {}

        try:
            # Fundamental frequency estimation
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )

            # Remove NaN values
            valid_f0 = f0[~np.isnan(f0)]

            if len(valid_f0) > 0:
                features['f0_mean'] = np.mean(valid_f0)
                features['f0_std'] = np.std(valid_f0)
                features['f0_range'] = np.max(valid_f0) - np.min(valid_f0)
                features['f0_median'] = np.median(valid_f0)
                features['voiced_ratio'] = np.mean(voiced_flag)
                features['voice_confidence'] = np.mean(voiced_probs)
            else:
                features.update({
                    'f0_mean': 0, 'f0_std': 0, 'f0_range': 0,
                    'f0_median': 0, 'voiced_ratio': 0, 'voice_confidence': 0
                })

            # Voice activity detection (simplified)
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            energy_threshold = np.mean(rms_energy) * 0.1
            voice_activity = rms_energy > energy_threshold
            features['voice_activity_ratio'] = np.mean(voice_activity)

        except Exception as e:
            logger.warning(f"Pitch feature extraction failed: {e}")
            features = {
                'f0_mean': 0, 'f0_std': 0, 'f0_range': 0,
                'f0_median': 0, 'voiced_ratio': 0, 'voice_confidence': 0,
                'voice_activity_ratio': 0
            }

        return features

    def preprocess_single(self,
                         audio_path: Union[str, Path],
                         platform_config: Optional[Dict] = None,
                         extract_features: bool = False,
                         duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Preprocess a single audio file

        Args:
            audio_path: Path to audio file
            platform_config: Platform-specific configuration
            extract_features: Whether to extract additional features
            duration: Target duration (overrides instance default)

        Returns:
            Dictionary with processed audio and metadata
        """
        start_time = time.time()

        try:
            audio_path = Path(audio_path)
            target_duration = duration or self.duration
            target_samples = int(self.sample_rate * target_duration)

            # Load audio
            audio_data, orig_sr = self.load_audio(audio_path)
            original_duration = len(audio_data) / orig_sr

            # Resample if needed
            if orig_sr != self.sample_rate:
                audio_data = self.resample_audio(audio_data, orig_sr, self.sample_rate)

            # Apply platform preprocessing
            platform = platform_config.get('platform', 'generic') if platform_config else 'generic'
            if platform != 'generic':
                audio_data = self.apply_platform_preprocessing(audio_data, platform)

            # Normalize
            if self.normalize:
                audio_data = self.normalize_audio(audio_data, method='peak')

            # Pad or trim to target duration
            audio_data = self.pad_or_trim_audio(audio_data, target_samples)

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)  # Add batch dimension

            result = {
                'processed_audio': audio_tensor,
                'sample_rate': self.sample_rate,
                'duration': target_duration,
                'original_duration': original_duration,
                'original_sample_rate': orig_sr,
                'platform': platform,
                'processing_time': time.time() - start_time,
                'source_path': str(audio_path)
            }

            # Extract additional features if requested
            if extract_features:
                spectral_features = self.extract_spectral_features(audio_data)
                pitch_features = self.extract_pitch_features(audio_data)

                result['spectral_features'] = spectral_features
                result['pitch_features'] = pitch_features

            return result

        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise

    def preprocess(self,
                  audio_input: Union[str, Path],
                  duration: Optional[float] = None,
                  platform_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main preprocessing method (wrapper for backward compatibility)

        Args:
            audio_input: Input audio path
            duration: Target duration
            platform_config: Platform configuration

        Returns:
            Preprocessing results
        """
        return self.preprocess_single(
            audio_input,
            platform_config=platform_config,
            extract_features=False,
            duration=duration
        )

    def load_dataset_split(self,
                          dataset_path: Union[str, Path],
                          split: str = 'train',
                          limit: Optional[int] = None,
                          shuffle: bool = True) -> Dict[str, List]:
        """
        Load dataset split following the structure:
        dataset/audio/{train,validation,test}/{FAKE,REAL}/

        Args:
            dataset_path: Path to dataset root
            split: Dataset split ('train', 'validation', 'test')
            limit: Maximum number of samples per class
            shuffle: Whether to shuffle the samples

        Returns:
            Dictionary with file paths and labels
        """
        dataset_root = Path(dataset_path)
        split_path = dataset_root / 'audio' / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split path not found: {split_path}")

        # Load files from each class
        fake_files = []
        real_files = []

        # Audio extensions
        audio_extensions = ['.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg', '.wma']

        # Load FAKE samples
        fake_path = split_path / 'FAKE'
        if fake_path.exists():
            for ext in audio_extensions:
                fake_files.extend(list(fake_path.glob(f'*{ext}')))
                fake_files.extend(list(fake_path.glob(f'*{ext.upper()}')))

        # Load REAL samples
        real_path = split_path / 'REAL'
        if real_path.exists():
            for ext in audio_extensions:
                real_files.extend(list(real_path.glob(f'*{ext}')))
                real_files.extend(list(real_path.glob(f'*{ext.upper()}')))

        # Apply limit if specified
        if limit:
            fake_files = fake_files[:limit]
            real_files = real_files[:limit]

        # Combine and create labels
        file_paths = fake_files + real_files
        labels = [1] * len(fake_files) + [0] * len(real_files)  # 1 for FAKE, 0 for REAL

        # Shuffle if requested
        if shuffle:
            from random import shuffle as random_shuffle
            combined = list(zip(file_paths, labels))
            random_shuffle(combined)
            file_paths, labels = zip(*combined)
            file_paths, labels = list(file_paths), list(labels)

        logger.info(f"Loaded {len(file_paths)} audio files from {split} split "
                   f"({len(fake_files)} fake, {len(real_files)} real)")

        return {
            'file_paths': [str(p) for p in file_paths],
            'labels': labels,
            'class_counts': {'FAKE': len(fake_files), 'REAL': len(real_files)},
            'split': split,
            'total_samples': len(file_paths)
        }

    def create_data_loader(self,
                          dataset_path: Union[str, Path],
                          split: str = 'train',
                          batch_size: int = 16,
                          shuffle: bool = True,
                          num_workers: int = 2) -> torch.utils.data.DataLoader:
        """
        Create PyTorch DataLoader for audio processing

        Args:
            dataset_path: Path to dataset
            split: Dataset split
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            PyTorch DataLoader
        """
        dataset_info = self.load_dataset_split(dataset_path, split, shuffle=shuffle)

        class AudioDataset(torch.utils.data.Dataset):
            def __init__(self, file_paths, labels, preprocessor, platform_config=None):
                self.file_paths = file_paths
                self.labels = labels
                self.preprocessor = preprocessor
                self.platform_config = platform_config

            def __len__(self):
                return len(self.file_paths)

            def __getitem__(self, idx):
                try:
                    result = self.preprocessor.preprocess_single(
                        self.file_paths[idx],
                        platform_config=self.platform_config,
                        extract_features=False
                    )
                    return result['processed_audio'], self.labels[idx]
                except Exception as e:
                    logger.warning(f"Error loading audio {self.file_paths[idx]}: {e}")
                    # Return dummy tensor
                    dummy_audio = torch.zeros(1, self.preprocessor.n_samples)
                    return dummy_audio, self.labels[idx]

        dataset = AudioDataset(
            dataset_info['file_paths'],
            dataset_info['labels'],
            self
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        logger.info(f"Created Audio DataLoader for {split} split with {len(dataset)} samples")

        return dataloader

    def get_dataset_statistics(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the audio dataset

        Args:
            dataset_path: Path to dataset root

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'dataset_path': str(dataset_path),
            'splits': {},
            'overall': {'FAKE': 0, 'REAL': 0, 'total': 0}
        }

        dataset_root = Path(dataset_path)
        audio_root = dataset_root / 'audio'

        if not audio_root.exists():
            return {'error': f'Audio dataset path not found: {audio_root}'}

        splits = ['train', 'validation', 'test']

        for split in splits:
            split_path = audio_root / split
            if not split_path.exists():
                stats['splits'][split] = {'error': f'Split not found: {split}'}
                continue

            try:
                split_info = self.load_dataset_split(dataset_path, split, shuffle=False)

                stats['splits'][split] = {
                    'FAKE': split_info['class_counts']['FAKE'],
                    'REAL': split_info['class_counts']['REAL'],
                    'total': split_info['total_samples'],
                    'fake_ratio': split_info['class_counts']['FAKE'] / split_info['total_samples'] if split_info['total_samples'] > 0 else 0,
                    'balance_score': min(split_info['class_counts']['FAKE'], split_info['class_counts']['REAL']) / max(split_info['class_counts']['FAKE'], split_info['class_counts']['REAL']) if max(split_info['class_counts']['FAKE'], split_info['class_counts']['REAL']) > 0 else 0
                }

                # Add to overall stats
                stats['overall']['FAKE'] += split_info['class_counts']['FAKE']
                stats['overall']['REAL'] += split_info['class_counts']['REAL']
                stats['overall']['total'] += split_info['total_samples']

            except Exception as e:
                stats['splits'][split] = {'error': str(e)}

        # Calculate overall ratios
        if stats['overall']['total'] > 0:
            stats['overall']['fake_ratio'] = stats['overall']['FAKE'] / stats['overall']['total']
            stats['overall']['real_ratio'] = stats['overall']['REAL'] / stats['overall']['total']

        return stats
