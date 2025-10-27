"""
Audio Feature Extractor for Multi-Modal Deepfake Detection
Specialized extraction of voice quality, spectral, and prosodic features
Focuses on detecting voice synthesis artifacts and unnatural speech patterns
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    Advanced audio feature extractor for deepfake speech detection
    Analyzes voice quality, spectral characteristics, and prosodic features
    """

    def __init__(self, sample_rate: int = 16000, device: str = 'cpu'):
        """
        Initialize Audio Feature Extractor

        Args:
            sample_rate: Audio sample rate
            device: Device for computation ('cpu' or 'cuda')
        """
        self.sample_rate = sample_rate
        self.device = device

        # Initialize feature extraction parameters
        self._init_spectral_params()
        self._init_prosodic_params()
        self._init_voice_quality_params()

        logger.info(f"Initialized AudioFeatureExtractor with {sample_rate}Hz sample rate")

    def _init_spectral_params(self):
        """Initialize spectral analysis parameters"""

        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048

        # Mel-scale parameters
        self.n_mels = 128
        self.n_mfcc = 40
        self.fmin = 80
        self.fmax = 8000

        # Spectral feature parameters
        self.spectral_bands = [
            (80, 300),      # Low frequency
            (300, 1000),    # Mid-low frequency
            (1000, 3000),   # Mid frequency
            (3000, 8000)    # High frequency
        ]

    def _init_prosodic_params(self):
        """Initialize prosodic analysis parameters"""

        # Pitch analysis
        self.pitch_fmin = 50
        self.pitch_fmax = 500
        self.pitch_threshold = 0.3

        # Voice activity detection
        self.vad_frame_length = 0.025  # 25ms
        self.vad_hop_length = 0.01     # 10ms
        self.energy_threshold = 0.01

        # Rhythm analysis
        self.rhythm_window = 0.1  # 100ms windows

    def _init_voice_quality_params(self):
        """Initialize voice quality analysis parameters"""

        # Formant analysis
        self.formant_max_freq = 5000
        self.formant_frame_length = 0.025
        self.formant_hop_length = 0.01

        # Jitter and shimmer analysis
        self.jitter_period_range = (0.002, 0.02)  # 2-20ms
        self.shimmer_analysis_length = 0.1  # 100ms windows

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features from audio

        Args:
            audio: Audio signal

        Returns:
            Dictionary of spectral features
        """
        features = {}

        try:
            # Short-Time Fourier Transform
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            magnitude = np.abs(stft)
            power = magnitude ** 2

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax
            )

            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )

            # MFCC statistics
            features.update({
                'mfcc_mean': float(np.mean(mfccs)),
                'mfcc_std': float(np.std(mfccs)),
                'mfcc_skewness': float(stats.skew(mfccs.flatten())),
                'mfcc_kurtosis': float(stats.kurtosis(mfccs.flatten()))
            })

            # First and second derivatives (delta and delta-delta)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            features.update({
                'mfcc_delta_mean': float(np.mean(mfcc_delta)),
                'mfcc_delta_std': float(np.std(mfcc_delta)),
                'mfcc_delta2_mean': float(np.mean(mfcc_delta2)),
                'mfcc_delta2_std': float(np.std(mfcc_delta2))
            })

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]

            features.update({
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_contrast_mean': float(np.mean(spectral_contrast)),
                'spectral_contrast_std': float(np.std(spectral_contrast)),
                'spectral_flatness_mean': float(np.mean(spectral_flatness)),
                'spectral_flatness_std': float(np.std(spectral_flatness))
            })

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.update({
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr))
            })

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features.update({
                'chroma_mean': float(np.mean(chroma)),
                'chroma_std': float(np.std(chroma)),
                'chroma_energy': float(np.sum(chroma ** 2))
            })

            # Tonnetz (tonal centroid features)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            features.update({
                'tonnetz_mean': float(np.mean(tonnetz)),
                'tonnetz_std': float(np.std(tonnetz))
            })

            # Spectral energy distribution across bands
            for i, (fmin_band, fmax_band) in enumerate(self.spectral_bands):
                freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
                band_mask = (freq_bins >= fmin_band) & (freq_bins <= fmax_band)
                band_energy = np.mean(np.sum(power[band_mask, :], axis=0))
                features[f'spectral_energy_band_{i}'] = float(band_energy)

            # Spectral flux (measure of spectral change)
            spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
            features.update({
                'spectral_flux_mean': float(np.mean(spectral_flux)),
                'spectral_flux_std': float(np.std(spectral_flux))
            })

        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            # Return default values
            features = {
                'mfcc_mean': 0.0, 'mfcc_std': 1.0, 'mfcc_skewness': 0.0, 'mfcc_kurtosis': 3.0,
                'spectral_centroid_mean': 2000.0, 'spectral_centroid_std': 500.0,
                'spectral_bandwidth_mean': 1000.0, 'spectral_bandwidth_std': 200.0,
                'zcr_mean': 0.1, 'zcr_std': 0.05
            }

        return features

    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch and F0-related features

        Args:
            audio: Audio signal

        Returns:
            Dictionary of pitch features
        """
        features = {}

        try:
            # Fundamental frequency estimation using PYIN
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=self.pitch_fmin, fmax=self.pitch_fmax,
                sr=self.sample_rate, threshold=self.pitch_threshold
            )

            # Remove unvoiced segments
            valid_f0 = f0[voiced_flag]

            if len(valid_f0) > 0:
                # Basic pitch statistics
                features.update({
                    'f0_mean': float(np.mean(valid_f0)),
                    'f0_std': float(np.std(valid_f0)),
                    'f0_median': float(np.median(valid_f0)),
                    'f0_range': float(np.max(valid_f0) - np.min(valid_f0)),
                    'f0_q75_q25': float(np.percentile(valid_f0, 75) - np.percentile(valid_f0, 25))
                })

                # Pitch contour smoothness
                f0_diff = np.diff(valid_f0)
                features.update({
                    'f0_contour_smoothness': float(1.0 / (1.0 + np.std(f0_diff))),
                    'f0_jumps': float(np.sum(np.abs(f0_diff) > np.std(f0_diff) * 2) / len(f0_diff))
                })

                # Pitch modulation analysis
                # Apply smoothing to get slow pitch changes
                if len(valid_f0) > 10:
                    smoothed_f0 = signal.savgol_filter(valid_f0, min(11, len(valid_f0)//2*2+1), 3)
                    pitch_modulation = valid_f0 - smoothed_f0
                    features.update({
                        'pitch_modulation_depth': float(np.std(pitch_modulation)),
                        'pitch_tremor': float(np.mean(np.abs(pitch_modulation)))
                    })

            else:
                features.update({
                    'f0_mean': 0.0, 'f0_std': 0.0, 'f0_median': 0.0,
                    'f0_range': 0.0, 'f0_q75_q25': 0.0,
                    'f0_contour_smoothness': 0.0, 'f0_jumps': 0.0,
                    'pitch_modulation_depth': 0.0, 'pitch_tremor': 0.0
                })

            # Voice activity detection
            voiced_ratio = np.mean(voiced_flag)
            voice_confidence = np.mean(voiced_probs)

            features.update({
                'voiced_ratio': float(voiced_ratio),
                'voice_confidence': float(voice_confidence)
            })

            # Pitch stability analysis
            if len(valid_f0) > 5:
                # Local pitch variations
                local_variations = []
                window_size = min(5, len(valid_f0))

                for i in range(len(valid_f0) - window_size + 1):
                    window = valid_f0[i:i + window_size]
                    variation = np.std(window) / np.mean(window) if np.mean(window) > 0 else 0
                    local_variations.append(variation)

                features['pitch_local_stability'] = float(1.0 / (1.0 + np.mean(local_variations)))

        except Exception as e:
            logger.warning(f"Pitch feature extraction failed: {e}")
            features = {
                'f0_mean': 150.0, 'f0_std': 30.0, 'f0_median': 150.0,
                'f0_range': 100.0, 'voiced_ratio': 0.6, 'voice_confidence': 0.7,
                'pitch_local_stability': 0.8
            }

        return features

    def extract_voice_quality_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract voice quality features (jitter, shimmer, HNR, etc.)

        Args:
            audio: Audio signal

        Returns:
            Dictionary of voice quality features
        """
        features = {}

        try:
            # Harmonic-to-Noise Ratio (HNR)
            # Simplified HNR estimation
            # Apply autocorrelation to find periodicity
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find dominant period
            if len(autocorr) > 100:
                # Look for peak in reasonable pitch period range (2ms to 20ms)
                min_period = int(0.002 * self.sample_rate)
                max_period = int(0.02 * self.sample_rate)

                search_range = autocorr[min_period:min(max_period, len(autocorr))]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    hnr_estimate = autocorr[peak_idx] / (autocorr[0] + 1e-8)
                    features['hnr_estimate'] = float(max(0, min(hnr_estimate, 1.0)))
                else:
                    features['hnr_estimate'] = 0.5
            else:
                features['hnr_estimate'] = 0.5

            # Energy-based voice quality measures
            # RMS energy analysis
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.01 * self.sample_rate)     # 10ms hop

            rms_values = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_values.append(rms)

            if rms_values:
                rms_array = np.array(rms_values)

                # Shimmer (amplitude perturbation)
                shimmer_local = []
                for i in range(len(rms_array) - 1):
                    if rms_array[i] + rms_array[i+1] > 0:
                        shimmer_local.append(abs(rms_array[i] - rms_array[i+1]) / 
                                           (rms_array[i] + rms_array[i+1]) * 2)

                if shimmer_local:
                    features['shimmer_local'] = float(np.mean(shimmer_local))
                    features['shimmer_variability'] = float(np.std(shimmer_local))
                else:
                    features['shimmer_local'] = 0.05
                    features['shimmer_variability'] = 0.02

                # Voice stability measures
                features.update({
                    'energy_stability': float(1.0 / (1.0 + np.std(rms_array) / (np.mean(rms_array) + 1e-8))),
                    'energy_dynamic_range': float(np.max(rms_array) - np.min(rms_array)) if len(rms_array) > 0 else 0.0
                })

            # Spectral tilt (measure of voice quality)
            # Compute power spectrum
            freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)

            # Fit line to log spectrum in speech range (100Hz - 4kHz)
            speech_mask = (freqs >= 100) & (freqs <= 4000)
            if np.sum(speech_mask) > 10:
                log_freqs = np.log(freqs[speech_mask])
                log_psd = np.log(psd[speech_mask] + 1e-10)

                # Linear regression for spectral tilt
                slope, intercept = np.polyfit(log_freqs, log_psd, 1)
                features['spectral_tilt'] = float(slope)

                # High-frequency emphasis
                high_freq_mask = (freqs >= 2000) & (freqs <= 8000)
                low_freq_mask = (freqs >= 100) & (freqs <= 1000)

                if np.sum(high_freq_mask) > 0 and np.sum(low_freq_mask) > 0:
                    high_freq_energy = np.mean(psd[high_freq_mask])
                    low_freq_energy = np.mean(psd[low_freq_mask])
                    features['high_freq_emphasis'] = float(high_freq_energy / (low_freq_energy + 1e-8))
                else:
                    features['high_freq_emphasis'] = 1.0
            else:
                features.update({
                    'spectral_tilt': -1.0,
                    'high_freq_emphasis': 1.0
                })

            # Voice breaks and irregularities
            # Detect sudden energy drops (voice breaks)
            if rms_values and len(rms_values) > 5:
                energy_threshold = np.mean(rms_values) * 0.3
                voice_breaks = np.sum(np.array(rms_values) < energy_threshold)
                features['voice_break_ratio'] = float(voice_breaks / len(rms_values))

                # Energy variability
                energy_diff = np.diff(rms_values)
                features['energy_variability'] = float(np.std(energy_diff))
            else:
                features.update({
                    'voice_break_ratio': 0.05,
                    'energy_variability': 0.1
                })

        except Exception as e:
            logger.warning(f"Voice quality feature extraction failed: {e}")
            features = {
                'hnr_estimate': 0.7, 'shimmer_local': 0.05, 'shimmer_variability': 0.02,
                'energy_stability': 0.8, 'energy_dynamic_range': 0.3,
                'spectral_tilt': -1.0, 'high_freq_emphasis': 1.0,
                'voice_break_ratio': 0.05, 'energy_variability': 0.1
            }

        return features

    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features (rhythm, stress, intonation)

        Args:
            audio: Audio signal

        Returns:
            Dictionary of prosodic features
        """
        features = {}

        try:
            # Energy-based rhythm analysis
            frame_length = int(self.rhythm_window * self.sample_rate)
            hop_length = frame_length // 2

            energy_contour = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.sum(frame ** 2)
                energy_contour.append(energy)

            if len(energy_contour) > 10:
                energy_contour = np.array(energy_contour)

                # Rhythm regularity
                # Autocorrelation of energy contour
                energy_autocorr = np.correlate(energy_contour, energy_contour, mode='full')
                energy_autocorr = energy_autocorr[len(energy_autocorr)//2:]

                # Find rhythmic periods
                if len(energy_autocorr) > 10:
                    # Look for peaks indicating rhythmic patterns
                    peaks, _ = signal.find_peaks(energy_autocorr[1:], height=np.max(energy_autocorr) * 0.3)
                    if len(peaks) > 0:
                        dominant_period = peaks[0] + 1  # Add 1 for the offset
                        rhythm_strength = energy_autocorr[dominant_period] / energy_autocorr[0]
                        features['rhythm_strength'] = float(rhythm_strength)
                        features['rhythm_period'] = float(dominant_period * hop_length / self.sample_rate)
                    else:
                        features.update({'rhythm_strength': 0.3, 'rhythm_period': 0.5})

                # Speech rate estimation (very rough)
                # Count energy peaks as potential syllables
                energy_peaks, _ = signal.find_peaks(
                    energy_contour, 
                    height=np.mean(energy_contour) + 0.5 * np.std(energy_contour),
                    distance=int(0.1 * self.sample_rate / hop_length)  # Minimum 100ms apart
                )

                speech_duration = len(audio) / self.sample_rate
                estimated_speech_rate = len(energy_peaks) / speech_duration if speech_duration > 0 else 0
                features['estimated_speech_rate'] = float(estimated_speech_rate)

                # Energy dynamics
                features.update({
                    'energy_contour_range': float(np.max(energy_contour) - np.min(energy_contour)),
                    'energy_contour_std': float(np.std(energy_contour)),
                    'energy_contour_skewness': float(stats.skew(energy_contour))
                })

            # Pause analysis
            # Detect silent regions
            silent_threshold = np.mean(np.abs(audio)) * 0.1
            silent_frames = np.abs(audio) < silent_threshold

            # Find continuous silent regions
            silent_regions = []
            in_silence = False
            silence_start = 0

            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_silence:
                    silence_start = i
                    in_silence = True
                elif not is_silent and in_silence:
                    silence_duration = (i - silence_start) / self.sample_rate
                    if silence_duration > 0.05:  # Only count pauses > 50ms
                        silent_regions.append(silence_duration)
                    in_silence = False

            if silent_regions:
                features.update({
                    'pause_count': len(silent_regions),
                    'pause_duration_mean': float(np.mean(silent_regions)),
                    'pause_duration_std': float(np.std(silent_regions)),
                    'pause_ratio': float(sum(silent_regions) / (len(audio) / self.sample_rate))
                })
            else:
                features.update({
                    'pause_count': 0, 'pause_duration_mean': 0.0,
                    'pause_duration_std': 0.0, 'pause_ratio': 0.0
                })

        except Exception as e:
            logger.warning(f"Prosodic feature extraction failed: {e}")
            features = {
                'rhythm_strength': 0.5, 'rhythm_period': 0.5,
                'estimated_speech_rate': 3.0, 'energy_contour_range': 0.5,
                'pause_count': 5, 'pause_duration_mean': 0.2, 'pause_ratio': 0.1
            }

        return features

    def extract_synthesis_artifacts(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract features specific to synthesis artifacts

        Args:
            audio: Audio signal

        Returns:
            Dictionary of synthesis artifact features
        """
        features = {}

        try:
            # High-frequency artifact detection
            # Compute spectrogram
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)

            # Analyze very high frequencies (potential synthesis artifacts)
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)

            # Very high frequency energy (above 6kHz)
            high_freq_mask = freq_bins > 6000
            very_high_freq_energy = np.mean(np.sum(magnitude[high_freq_mask, :], axis=0))
            total_energy = np.mean(np.sum(magnitude, axis=0))

            features['very_high_freq_ratio'] = float(very_high_freq_energy / (total_energy + 1e-8))

            # Spectral regularity analysis
            # Look for unnatural spectral patterns
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=128
            )

            # Check for overly smooth spectral evolution
            mel_diff = np.diff(mel_spec, axis=1)
            spectral_smoothness = 1.0 / (1.0 + np.std(mel_diff))
            features['spectral_temporal_smoothness'] = float(spectral_smoothness)

            # Formant regularity (simplified)
            # Look for unnatural formant patterns
            if len(magnitude) > 100:
                # Focus on speech formant region (0-4kHz)
                speech_mask = freq_bins <= 4000
                speech_spectrum = magnitude[speech_mask, :]

                # Analyze spectral peaks consistency
                peak_positions = []
                for frame in speech_spectrum.T:
                    peaks, _ = signal.find_peaks(frame, height=np.max(frame) * 0.3)
                    peak_positions.append(len(peaks))

                if peak_positions:
                    formant_consistency = 1.0 / (1.0 + np.std(peak_positions))
                    features['formant_consistency'] = float(formant_consistency)
                else:
                    features['formant_consistency'] = 0.5

            # Phase coherence analysis
            # Check for unnatural phase patterns in synthesis
            phase = np.angle(stft)
            phase_diff = np.diff(phase, axis=1)

            # Instantaneous frequency deviation
            inst_freq_dev = np.std(phase_diff, axis=1)
            features['phase_coherence'] = float(1.0 / (1.0 + np.mean(inst_freq_dev)))

            # Periodicity artifacts
            # Check for unnatural periodicity in the signal
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Look for multiple strong periodicities (artifact of some synthesis)
            if len(autocorr) > 1000:
                peaks, _ = signal.find_peaks(
                    autocorr[100:1000],  # Skip the main peak at 0
                    height=np.max(autocorr) * 0.2
                )

                strong_periodicities = len(peaks)
                features['periodicity_artifacts'] = float(min(strong_periodicities / 5.0, 1.0))
            else:
                features['periodicity_artifacts'] = 0.2

        except Exception as e:
            logger.warning(f"Synthesis artifact extraction failed: {e}")
            features = {
                'very_high_freq_ratio': 0.05, 'spectral_temporal_smoothness': 0.7,
                'formant_consistency': 0.8, 'phase_coherence': 0.6,
                'periodicity_artifacts': 0.2
            }

        return features

    def extract_all_audio_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive audio features

        Args:
            audio: Audio signal

        Returns:
            Dictionary of all audio features
        """
        all_features = {}

        try:
            # Extract different feature categories
            spectral_features = self.extract_spectral_features(audio)
            pitch_features = self.extract_pitch_features(audio)
            voice_quality_features = self.extract_voice_quality_features(audio)
            prosodic_features = self.extract_prosodic_features(audio)
            synthesis_features = self.extract_synthesis_artifacts(audio)

            # Combine all features
            all_features.update(spectral_features)
            all_features.update(pitch_features)
            all_features.update(voice_quality_features)
            all_features.update(prosodic_features)
            all_features.update(synthesis_features)

            # Add summary statistics
            feature_values = [v for v in all_features.values() if isinstance(v, (int, float))]
            if feature_values:
                all_features.update({
                    'audio_feature_mean': float(np.mean(feature_values)),
                    'audio_feature_std': float(np.std(feature_values)),
                    'audio_feature_count': len(feature_values)
                })

        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            # Return minimal features
            all_features = {
                'spectral_score': 0.5, 'pitch_score': 0.5,
                'voice_quality_score': 0.5, 'prosodic_score': 0.5,
                'synthesis_artifact_score': 0.5, 'audio_feature_count': 5
            }

        return all_features

    def compute_audio_quality_score(self, features: Dict[str, float]) -> float:
        """
        Compute overall audio quality score from features

        Args:
            features: Extracted audio features

        Returns:
            Quality score (0-1, higher = better quality)
        """
        try:
            quality_indicators = []

            # Voice quality indicators
            if 'hnr_estimate' in features:
                quality_indicators.append(features['hnr_estimate'])

            if 'energy_stability' in features:
                quality_indicators.append(features['energy_stability'])

            # Spectral quality
            if 'spectral_centroid_mean' in features:
                # Normalize spectral centroid (typical speech: 1000-3000 Hz)
                centroid_quality = 1.0 - abs(features['spectral_centroid_mean'] - 2000) / 2000
                quality_indicators.append(max(0, min(centroid_quality, 1.0)))

            # Low synthesis artifacts
            if 'very_high_freq_ratio' in features:
                artifact_quality = max(0, 1.0 - features['very_high_freq_ratio'] * 10)
                quality_indicators.append(artifact_quality)

            # Natural prosody
            if 'rhythm_strength' in features:
                quality_indicators.append(features['rhythm_strength'])

            if quality_indicators:
                return float(np.mean(quality_indicators))
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"Audio quality score computation failed: {e}")
            return 0.5
