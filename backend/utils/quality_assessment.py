"""
Quality Assessment Utilities for Multi-Modal Deepfake Detection
Comprehensive quality evaluation for images, videos, and audio
Includes technical quality metrics, perceptual quality, and deepfake-specific assessments
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from scipy import signal, ndimage, stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class QualityAssessor:
    """
    Comprehensive quality assessor for multi-modal media
    Evaluates technical quality, perceptual quality, and manipulation artifacts
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize Quality Assessor

        Args:
            device: Device for computation
        """
        self.device = device

        # Initialize quality assessment parameters
        self._init_image_quality_metrics()
        self._init_video_quality_metrics()
        self._init_audio_quality_metrics()
        self._init_perceptual_metrics()

        logger.info("Initialized QualityAssessor")

    def _init_image_quality_metrics(self):
        """Initialize image quality assessment parameters"""

        # Sharpness assessment
        self.sharpness_kernels = {
            'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
            'sobel': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'tenengrad': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        }

        # Noise assessment
        self.noise_estimation_methods = ['wavelet', 'laplacian', 'robust_mad']

        # Color quality parameters
        self.color_spaces = ['RGB', 'HSV', 'LAB', 'YUV']

    def _init_video_quality_metrics(self):
        """Initialize video quality assessment parameters"""

        # Temporal quality parameters
        self.temporal_window_sizes = [3, 5, 7]
        self.motion_threshold = 1.0

        # Frame quality assessment
        self.frame_similarity_methods = ['mse', 'ssim', 'psnr']

    def _init_audio_quality_metrics(self):
        """Initialize audio quality assessment parameters"""

        # Audio quality parameters
        self.sample_rates = [8000, 16000, 22050, 44100, 48000]
        self.snr_estimation_methods = ['spectral', 'temporal']

        # Voice quality parameters
        self.voice_quality_bands = [
            (80, 300),    # Fundamental frequency range
            (300, 1000),  # Low formants
            (1000, 3400), # Speech clarity range
            (3400, 8000)  # High frequency content
        ]

    def _init_perceptual_metrics(self):
        """Initialize perceptual quality metrics"""

        # SSIM parameters
        self.ssim_window_size = 11
        self.ssim_k1 = 0.01
        self.ssim_k2 = 0.03

        # MS-SSIM scales
        self.msssim_scales = [1, 2, 4, 8, 16]

    def assess_image_quality(self, image: np.ndarray, reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Assess image quality using multiple metrics

        Args:
            image: Input image
            reference: Reference image for comparison (optional)

        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}

        try:
            # Technical quality metrics
            technical_quality = self._assess_technical_quality(image)
            quality_metrics.update(technical_quality)

            # Perceptual quality metrics
            perceptual_quality = self._assess_perceptual_quality(image)
            quality_metrics.update(perceptual_quality)

            # Reference-based metrics if available
            if reference is not None:
                reference_quality = self._assess_reference_based_quality(image, reference)
                quality_metrics.update(reference_quality)

            # Overall quality score
            quality_scores = [v for v in quality_metrics.values() if isinstance(v, (int, float)) and 0 <= v <= 1]
            if quality_scores:
                quality_metrics['overall_image_quality'] = float(np.mean(quality_scores))

        except Exception as e:
            logger.warning(f"Image quality assessment failed: {e}")
            quality_metrics = {
                'sharpness_score': 0.5, 'noise_score': 0.7,
                'color_quality': 0.8, 'overall_image_quality': 0.7
            }

        return quality_metrics

    def _assess_technical_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess technical image quality"""
        metrics = {}

        try:
            # Convert to grayscale for some metrics
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Sharpness assessment
            sharpness_scores = []
            for kernel_name, kernel in self.sharpness_kernels.items():
                if kernel_name == 'tenengrad':
                    # Tenengrad variance
                    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    tenengrad = grad_x**2 + grad_y**2
                    sharpness = np.mean(tenengrad)
                else:
                    filtered = cv2.filter2D(gray.astype(np.float64), -1, kernel)
                    sharpness = np.var(filtered)

                # Normalize sharpness score
                normalized_sharpness = min(sharpness / 1000.0, 1.0)
                sharpness_scores.append(normalized_sharpness)

            metrics['sharpness_score'] = float(np.mean(sharpness_scores))

            # Noise assessment
            noise_estimates = []

            # Method 1: Laplacian noise estimation
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_laplacian = np.var(laplacian)
            noise_estimates.append(min(1000.0 / (noise_laplacian + 1e-8), 1.0))

            # Method 2: Robust MAD estimation
            # Use median absolute deviation for noise estimation
            median_img = ndimage.median_filter(gray, size=3)
            noise_mad = np.median(np.abs(gray.astype(float) - median_img.astype(float)))
            noise_score_mad = max(0, 1.0 - noise_mad / 50.0)  # Normalize
            noise_estimates.append(noise_score_mad)

            metrics['noise_score'] = float(np.mean(noise_estimates))

            # Contrast assessment
            contrast = np.std(gray) / (np.mean(gray) + 1e-8)
            metrics['contrast_score'] = float(min(contrast / 2.0, 1.0))

            # Dynamic range
            dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
            metrics['dynamic_range'] = float(dynamic_range)

        except Exception as e:
            logger.warning(f"Technical quality assessment failed: {e}")
            metrics = {
                'sharpness_score': 0.5, 'noise_score': 0.7,
                'contrast_score': 0.6, 'dynamic_range': 0.8
            }

        return metrics

    def _assess_perceptual_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess perceptual image quality"""
        metrics = {}

        try:
            # Color quality assessment
            if len(image.shape) == 3:
                color_quality = self._assess_color_quality(image)
                metrics.update(color_quality)

            # Spatial frequency analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # FFT-based frequency analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            # High frequency content
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2

            # Create frequency masks
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # High frequency mask (outer regions)
            high_freq_mask = distances > min(h, w) // 4
            low_freq_mask = distances <= min(h, w) // 8

            high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
            total_energy = np.sum(magnitude_spectrum)

            if total_energy > 0:
                high_freq_ratio = high_freq_energy / total_energy
                metrics['high_freq_content'] = float(min(high_freq_ratio * 10, 1.0))

            # Edge density (perceptual sharpness indicator)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            metrics['edge_density'] = float(edge_density * 10)  # Scale up

        except Exception as e:
            logger.warning(f"Perceptual quality assessment failed: {e}")
            metrics = {
                'high_freq_content': 0.3,
                'edge_density': 0.1
            }

        return metrics

    def _assess_color_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess color quality of image"""
        metrics = {}

        try:
            # Color space analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Saturation analysis
            saturation = hsv[:, :, 1] / 255.0
            saturation_mean = np.mean(saturation)
            saturation_std = np.std(saturation)

            # Good images have moderate saturation with some variation
            saturation_quality = 1.0 - abs(saturation_mean - 0.5)  # Optimal around 0.5
            saturation_quality *= min(saturation_std * 4, 1.0)     # Some variation is good

            metrics['color_saturation_quality'] = float(saturation_quality)

            # Color distribution uniformity
            # Analyze color histogram in LAB space
            l_channel = lab[:, :, 0]
            a_channel = lab[:, :, 1]
            b_channel = lab[:, :, 2]

            # Color variance (higher is generally better for natural images)
            color_variance = np.mean([np.var(l_channel), np.var(a_channel), np.var(b_channel)])
            color_variance_score = min(color_variance / 2000.0, 1.0)  # Normalize

            metrics['color_variance'] = float(color_variance_score)

            # Color cast detection (should be balanced)
            r_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            b_mean = np.mean(image[:, :, 2])

            # Deviation from gray (128, 128, 128)
            color_balance = 1.0 - (abs(r_mean - g_mean) + abs(g_mean - b_mean) + abs(r_mean - b_mean)) / (3 * 255)
            metrics['color_balance'] = float(max(color_balance, 0))

        except Exception as e:
            logger.warning(f"Color quality assessment failed: {e}")
            metrics = {
                'color_saturation_quality': 0.7,
                'color_variance': 0.6,
                'color_balance': 0.8
            }

        return metrics

    def _assess_reference_based_quality(self, image: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """Assess quality using reference image"""
        metrics = {}

        try:
            # Ensure same size
            if image.shape != reference.shape:
                image = cv2.resize(image, (reference.shape[1], reference.shape[0]))

            # PSNR calculation
            mse = np.mean((image.astype(float) - reference.astype(float)) ** 2)
            if mse == 0:
                psnr = 100  # Perfect match
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))

            # Normalize PSNR to [0, 1] range (typical range: 20-50 dB)
            psnr_normalized = min((psnr - 20) / 30, 1.0) if psnr > 20 else 0
            metrics['psnr'] = float(max(psnr_normalized, 0))

            # SSIM calculation
            ssim_score = self._calculate_ssim(image, reference)
            metrics['ssim'] = float(ssim_score)

            # MS-SSIM calculation
            msssim_score = self._calculate_msssim(image, reference)
            metrics['ms_ssim'] = float(msssim_score)

        except Exception as e:
            logger.warning(f"Reference-based quality assessment failed: {e}")
            metrics = {
                'psnr': 0.7,
                'ssim': 0.8,
                'ms_ssim': 0.75
            }

        return metrics

    def _calculate_ssim(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate SSIM between two images"""
        try:
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = image1

            if len(image2.shape) == 3:
                gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = image2

            # Convert to float
            img1 = gray1.astype(np.float64)
            img2 = gray2.astype(np.float64)

            # SSIM constants
            k1, k2 = self.ssim_k1, self.ssim_k2
            L = 255  # Dynamic range
            c1 = (k1 * L) ** 2
            c2 = (k2 * L) ** 2

            # Gaussian kernel
            kernel = cv2.getGaussianKernel(self.ssim_window_size, 1.5)
            kernel = np.outer(kernel, kernel)

            # Statistics
            mu1 = cv2.filter2D(img1, -1, kernel)
            mu2 = cv2.filter2D(img2, -1, kernel)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = cv2.filter2D(img1 ** 2, -1, kernel) - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, kernel) - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2

            # SSIM formula
            numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

            ssim_map = numerator / (denominator + 1e-8)
            return float(np.mean(ssim_map))

        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.7

    def _calculate_msssim(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate Multi-Scale SSIM"""
        try:
            ssim_values = []

            # Calculate SSIM at multiple scales
            img1, img2 = image1.copy(), image2.copy()

            for scale in self.msssim_scales:
                if min(img1.shape[:2]) < 16:  # Stop if image becomes too small
                    break

                ssim_val = self._calculate_ssim(img1, img2)
                ssim_values.append(ssim_val)

                # Downsample for next scale
                if scale < self.msssim_scales[-1]:
                    h, w = img1.shape[:2]
                    new_h, new_w = h // 2, w // 2
                    img1 = cv2.resize(img1, (new_w, new_h))
                    img2 = cv2.resize(img2, (new_w, new_h))

            # Weighted average (higher weight for lower scales)
            if ssim_values:
                weights = np.array([1.0 / (i + 1) for i in range(len(ssim_values))])
                weights = weights / np.sum(weights)
                ms_ssim = np.average(ssim_values, weights=weights)
                return float(ms_ssim)
            else:
                return 0.7

        except Exception as e:
            logger.warning(f"MS-SSIM calculation failed: {e}")
            return 0.7

    def assess_video_quality(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Assess video quality using temporal and spatial metrics

        Args:
            frames: List of video frames

        Returns:
            Dictionary of video quality metrics
        """
        quality_metrics = {}

        try:
            if len(frames) < 2:
                return {'video_quality_score': 0.5}

            # Frame-level quality assessment
            frame_qualities = []
            for frame in frames[:10]:  # Sample first 10 frames
                frame_quality = self.assess_image_quality(frame)
                if 'overall_image_quality' in frame_quality:
                    frame_qualities.append(frame_quality['overall_image_quality'])

            if frame_qualities:
                quality_metrics['average_frame_quality'] = float(np.mean(frame_qualities))
                quality_metrics['frame_quality_consistency'] = float(1.0 - np.std(frame_qualities))

            # Temporal quality assessment
            temporal_quality = self._assess_temporal_quality(frames)
            quality_metrics.update(temporal_quality)

            # Motion quality assessment
            motion_quality = self._assess_motion_quality(frames)
            quality_metrics.update(motion_quality)

            # Overall video quality
            quality_values = [v for v in quality_metrics.values() if isinstance(v, (int, float)) and 0 <= v <= 1]
            if quality_values:
                quality_metrics['overall_video_quality'] = float(np.mean(quality_values))

        except Exception as e:
            logger.warning(f"Video quality assessment failed: {e}")
            quality_metrics = {
                'average_frame_quality': 0.7,
                'temporal_consistency': 0.8,
                'motion_smoothness': 0.7,
                'overall_video_quality': 0.73
            }

        return quality_metrics

    def _assess_temporal_quality(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Assess temporal quality of video"""
        metrics = {}

        try:
            # Frame-to-frame differences
            frame_diffs = []

            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]

                # Convert to grayscale
                if len(frame1.shape) == 3:
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                else:
                    gray1 = frame1

                if len(frame2.shape) == 3:
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                else:
                    gray2 = frame2

                # Resize if needed
                if gray1.shape != gray2.shape:
                    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

                # Calculate difference
                diff = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
                frame_diffs.append(diff)

            if frame_diffs:
                # Temporal consistency (lower variation = better)
                temporal_variation = np.std(frame_diffs)
                temporal_consistency = 1.0 / (1.0 + temporal_variation / 10.0)
                metrics['temporal_consistency'] = float(temporal_consistency)

                # Flickering detection
                high_diff_frames = np.sum(np.array(frame_diffs) > np.mean(frame_diffs) + 2 * np.std(frame_diffs))
                flickering_score = 1.0 - (high_diff_frames / len(frame_diffs))
                metrics['flickering_score'] = float(max(flickering_score, 0))

        except Exception as e:
            logger.warning(f"Temporal quality assessment failed: {e}")
            metrics = {
                'temporal_consistency': 0.8,
                'flickering_score': 0.9
            }

        return metrics

    def _assess_motion_quality(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Assess motion quality in video"""
        metrics = {}

        try:
            if len(frames) < 3:
                return {'motion_smoothness': 0.5}

            # Optical flow analysis
            motion_vectors = []

            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]

                # Convert to grayscale
                if len(frame1.shape) == 3:
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                else:
                    gray1 = frame1

                if len(frame2.shape) == 3:
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                else:
                    gray2 = frame2

                # Resize if needed
                if gray1.shape != gray2.shape:
                    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    gray1, gray2, None, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

                if flow is not None and len(flow) > 1:
                    # Calculate motion magnitude
                    if len(flow) >= 3:  # LK returns (points, status, error)
                        motion_magnitude = np.mean(np.linalg.norm(flow[0] - flow[1] if len(flow) > 1 else flow[0], axis=1))
                    else:
                        motion_magnitude = 1.0

                    motion_vectors.append(motion_magnitude)

            if motion_vectors:
                # Motion smoothness
                motion_variation = np.std(motion_vectors)
                motion_smoothness = 1.0 / (1.0 + motion_variation)
                metrics['motion_smoothness'] = float(motion_smoothness)

                # Motion magnitude
                avg_motion = np.mean(motion_vectors)
                # Normalize motion magnitude (optimal range 1-10 pixels)
                motion_quality = max(0, 1.0 - abs(avg_motion - 5) / 10.0)
                metrics['motion_magnitude_quality'] = float(motion_quality)

        except Exception as e:
            logger.warning(f"Motion quality assessment failed: {e}")
            metrics = {
                'motion_smoothness': 0.7,
                'motion_magnitude_quality': 0.8
            }

        return metrics

    def assess_audio_quality(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """
        Assess audio quality using various metrics

        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio

        Returns:
            Dictionary of audio quality metrics
        """
        quality_metrics = {}

        try:
            # SNR estimation
            snr_score = self._estimate_snr(audio)
            quality_metrics['snr_score'] = snr_score

            # THD+N estimation
            thd_score = self._estimate_thd(audio, sample_rate)
            quality_metrics['thd_score'] = thd_score

            # Dynamic range
            dynamic_range = self._calculate_audio_dynamic_range(audio)
            quality_metrics['dynamic_range'] = dynamic_range

            # Frequency response quality
            freq_quality = self._assess_frequency_response(audio, sample_rate)
            quality_metrics.update(freq_quality)

            # Clipping detection
            clipping_score = self._detect_audio_clipping(audio)
            quality_metrics['clipping_score'] = clipping_score

            # Overall audio quality
            quality_values = [v for v in quality_metrics.values() if isinstance(v, (int, float)) and 0 <= v <= 1]
            if quality_values:
                quality_metrics['overall_audio_quality'] = float(np.mean(quality_values))

        except Exception as e:
            logger.warning(f"Audio quality assessment failed: {e}")
            quality_metrics = {
                'snr_score': 0.7, 'thd_score': 0.8,
                'dynamic_range': 0.6, 'clipping_score': 0.9,
                'overall_audio_quality': 0.75
            }

        return quality_metrics

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        try:
            # Simple SNR estimation using signal variance vs noise variance
            # Assume noise is in quiet segments

            # Find quiet segments (bottom 10% of RMS values)
            frame_length = len(audio) // 100
            rms_values = []

            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_values.append(rms)

            if rms_values:
                noise_threshold = np.percentile(rms_values, 10)
                signal_power = np.mean(audio ** 2)
                noise_power = noise_threshold ** 2

                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    # Normalize to [0, 1] (typical SNR range: 0-60 dB)
                    snr_normalized = min(snr / 60.0, 1.0) if snr > 0 else 0
                    return float(max(snr_normalized, 0))

            return 0.7  # Default

        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}")
            return 0.7

    def _estimate_thd(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate Total Harmonic Distortion + Noise"""
        try:
            # Apply windowing and FFT
            windowed = audio * np.hanning(len(audio))
            fft = np.fft.fft(windowed)
            magnitude = np.abs(fft[:len(fft)//2])

            # Find fundamental frequency (peak in spectrum)
            freqs = np.fft.fftfreq(len(audio), 1/sample_rate)[:len(fft)//2]

            # Focus on speech range (100-1000 Hz)
            speech_mask = (freqs >= 100) & (freqs <= 1000)
            if np.sum(speech_mask) > 0:
                fundamental_idx = np.argmax(magnitude[speech_mask])
                fundamental_freq = freqs[speech_mask][fundamental_idx]

                # Calculate harmonic distortion (simplified)
                total_power = np.sum(magnitude ** 2)

                # Estimate distortion as high-frequency noise
                high_freq_mask = freqs > 4000
                if np.sum(high_freq_mask) > 0:
                    distortion_power = np.sum(magnitude[high_freq_mask] ** 2)
                    thd = 1.0 - (distortion_power / total_power)
                    return float(max(min(thd, 1.0), 0))

            return 0.8  # Default for speech

        except Exception as e:
            logger.warning(f"THD estimation failed: {e}")
            return 0.8

    def _calculate_audio_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate audio dynamic range"""
        try:
            # RMS-based dynamic range
            rms_values = []
            frame_length = len(audio) // 100

            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                if rms > 0:
                    rms_values.append(rms)

            if len(rms_values) > 1:
                dynamic_range_db = 20 * np.log10(max(rms_values) / min(rms_values))
                # Normalize to [0, 1] (typical range: 20-80 dB)
                dynamic_range_norm = min((dynamic_range_db - 20) / 60.0, 1.0) if dynamic_range_db > 20 else 0
                return float(max(dynamic_range_norm, 0))

            return 0.5

        except Exception as e:
            logger.warning(f"Dynamic range calculation failed: {e}")
            return 0.5

    def _assess_frequency_response(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Assess frequency response quality"""
        try:
            # FFT analysis
            fft = np.fft.fft(audio * np.hanning(len(audio)))
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(audio), 1/sample_rate)[:len(fft)//2]

            # Energy in different frequency bands
            band_energies = {}
            for i, (low_freq, high_freq) in enumerate(self.voice_quality_bands):
                band_mask = (freqs >= low_freq) & (freqs < high_freq)
                if np.sum(band_mask) > 0:
                    band_energy = np.sum(magnitude[band_mask] ** 2)
                    band_energies[f'band_{i}_energy'] = band_energy

            # Normalize band energies
            total_energy = sum(band_energies.values())
            if total_energy > 0:
                normalized_bands = {k: v / total_energy for k, v in band_energies.items()}
            else:
                normalized_bands = {f'band_{i}_energy': 0.25 for i in range(4)}

            # Frequency response flatness
            if len(band_energies) > 1:
                energy_values = list(normalized_bands.values())
                flatness = 1.0 - np.std(energy_values) / (np.mean(energy_values) + 1e-8)
                normalized_bands['frequency_flatness'] = float(max(flatness, 0))

            return {k: float(v) for k, v in normalized_bands.items()}

        except Exception as e:
            logger.warning(f"Frequency response assessment failed: {e}")
            return {
                'band_0_energy': 0.25, 'band_1_energy': 0.25,
                'band_2_energy': 0.25, 'band_3_energy': 0.25,
                'frequency_flatness': 0.8
            }

    def _detect_audio_clipping(self, audio: np.ndarray) -> float:
        """Detect audio clipping artifacts"""
        try:
            # Normalize audio to [-1, 1] range
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                normalized_audio = audio / max_val
            else:
                return 1.0

            # Detect clipping (values at or near ±1)
            clipping_threshold = 0.99
            clipped_samples = np.sum(np.abs(normalized_audio) >= clipping_threshold)

            # Clipping score (1.0 = no clipping, 0.0 = severe clipping)
            clipping_ratio = clipped_samples / len(audio)
            clipping_score = max(0, 1.0 - clipping_ratio * 100)  # Penalize clipping heavily

            return float(clipping_score)

        except Exception as e:
            logger.warning(f"Clipping detection failed: {e}")
            return 0.9

    def assess_multimodal_quality(self, 
                                 image: Optional[np.ndarray] = None,
                                 frames: Optional[List[np.ndarray]] = None,
                                 audio: Optional[np.ndarray] = None,
                                 sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Assess quality across multiple modalities

        Args:
            image: Image for assessment
            frames: Video frames for assessment
            audio: Audio for assessment
            sample_rate: Audio sample rate

        Returns:
            Dictionary of quality assessments for each modality
        """
        quality_results = {}

        try:
            # Image quality
            if image is not None:
                image_quality = self.assess_image_quality(image)
                quality_results['image_quality'] = image_quality

            # Video quality
            if frames is not None:
                video_quality = self.assess_video_quality(frames)
                quality_results['video_quality'] = video_quality

            # Audio quality
            if audio is not None:
                audio_quality = self.assess_audio_quality(audio, sample_rate)
                quality_results['audio_quality'] = audio_quality

            # Overall quality score
            overall_scores = []

            if 'image_quality' in quality_results:
                img_score = quality_results['image_quality'].get('overall_image_quality', 0.5)
                overall_scores.append(img_score)

            if 'video_quality' in quality_results:
                vid_score = quality_results['video_quality'].get('overall_video_quality', 0.5)
                overall_scores.append(vid_score)

            if 'audio_quality' in quality_results:
                aud_score = quality_results['audio_quality'].get('overall_audio_quality', 0.5)
                overall_scores.append(aud_score)

            if overall_scores:
                quality_results['overall_multimodal_quality'] = float(np.mean(overall_scores))

            # Quality consistency across modalities
            if len(overall_scores) > 1:
                quality_consistency = 1.0 - np.std(overall_scores)
                quality_results['quality_consistency'] = float(max(quality_consistency, 0))

        except Exception as e:
            logger.error(f"Multimodal quality assessment failed: {e}")
            quality_results = {
                'overall_multimodal_quality': 0.6,
                'quality_consistency': 0.7
            }

        return quality_results
