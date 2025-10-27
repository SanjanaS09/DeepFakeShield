"""
Advanced Artifact Analysis for Multi-Modal Deepfake Detection
Specialized detection of compression artifacts, manipulation traces, and synthetic patterns
Supports JPEG artifacts, GAN artifacts, warping distortions, and temporal inconsistencies
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from scipy import signal, ndimage, fft
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class ArtifactAnalyzer:
    """
    Advanced artifact analyzer for deepfake detection
    Detects various manipulation artifacts including compression, blending, and synthesis traces
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize Artifact Analyzer

        Args:
            device: Device for computation
        """
        self.device = device

        # Initialize analysis parameters
        self._init_compression_analysis()
        self._init_blending_analysis()
        self._init_synthesis_analysis()
        self._init_geometric_analysis()

        logger.info("Initialized ArtifactAnalyzer")

    def _init_compression_analysis(self):
        """Initialize compression artifact analysis parameters"""

        # JPEG compression parameters
        self.jpeg_block_size = 8
        self.jpeg_quality_levels = [10, 20, 30, 50, 70, 90, 95]

        # DCT analysis parameters
        self.dct_threshold = 0.1
        self.dct_zigzag_order = self._create_zigzag_order()

        # Quantization table analysis
        self.standard_qt_luma = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)

    def _init_blending_analysis(self):
        """Initialize blending artifact analysis parameters"""

        # Edge transition analysis
        self.edge_kernels = {
            'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
            'canny': None  # Will use cv2.Canny
        }

        # Color blending parameters
        self.color_spaces = ['RGB', 'HSV', 'LAB', 'YUV']
        self.blending_window_sizes = [3, 5, 7, 11]

    def _init_synthesis_analysis(self):
        """Initialize synthesis artifact analysis parameters"""

        # GAN artifact detection
        self.gan_frequency_bands = [
            (0, 50),      # Very low frequency
            (50, 100),    # Low frequency  
            (100, 200),   # Medium frequency
            (200, 400),   # High frequency
            (400, 1000)   # Very high frequency
        ]

        # Checkerboard pattern detection
        self.checkerboard_sizes = [2, 4, 8, 16]

        # Upsampling artifact detection
        self.upsampling_patterns = ['bilinear', 'nearest', 'bicubic']

    def _init_geometric_analysis(self):
        """Initialize geometric distortion analysis parameters"""

        # Warping detection
        self.grid_size = 16
        self.warp_threshold = 0.1

        # Perspective distortion
        self.perspective_grid_points = 100

    def _create_zigzag_order(self) -> np.ndarray:
        """Create zigzag order for DCT coefficient analysis"""
        zigzag = np.array([
            0,  1,  8, 16,  9,  2,  3, 10,
            17, 24, 32, 25, 18, 11,  4,  5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13,  6,  7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
        ])
        return zigzag

    def detect_jpeg_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect JPEG compression artifacts

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Dictionary of JPEG artifact measures
        """
        artifacts = {}

        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Blocking artifact detection
            blocking_score = self._detect_blocking_artifacts(gray)
            artifacts['blocking_artifacts'] = blocking_score

            # DCT coefficient analysis
            dct_artifacts = self._analyze_dct_artifacts(gray)
            artifacts.update(dct_artifacts)

            # Ringing artifacts around edges
            ringing_score = self._detect_ringing_artifacts(gray)
            artifacts['ringing_artifacts'] = ringing_score

            # Mosquito noise detection
            mosquito_score = self._detect_mosquito_noise(gray)
            artifacts['mosquito_noise'] = mosquito_score

            # Overall JPEG quality estimation
            quality_estimate = self._estimate_jpeg_quality(gray)
            artifacts['estimated_jpeg_quality'] = quality_estimate

        except Exception as e:
            logger.warning(f"JPEG artifact detection failed: {e}")
            artifacts = {
                'blocking_artifacts': 0.1, 'ringing_artifacts': 0.1,
                'mosquito_noise': 0.1, 'estimated_jpeg_quality': 80.0
            }

        return artifacts

    def _detect_blocking_artifacts(self, gray_image: np.ndarray) -> float:
        """Detect JPEG blocking artifacts"""
        try:
            h, w = gray_image.shape
            block_size = self.jpeg_block_size

            # Analyze 8x8 block boundaries
            vertical_diffs = []
            horizontal_diffs = []

            # Vertical boundaries
            for x in range(block_size, w, block_size):
                if x < w - 1:
                    diff = np.mean(np.abs(
                        gray_image[:, x].astype(float) - 
                        gray_image[:, x-1].astype(float)
                    ))
                    vertical_diffs.append(diff)

            # Horizontal boundaries
            for y in range(block_size, h, block_size):
                if y < h - 1:
                    diff = np.mean(np.abs(
                        gray_image[y, :].astype(float) - 
                        gray_image[y-1, :].astype(float)
                    ))
                    horizontal_diffs.append(diff)

            # Random boundaries for comparison
            random_diffs = []
            num_random = len(vertical_diffs) + len(horizontal_diffs)

            for _ in range(num_random):
                if np.random.choice([True, False]):  # Vertical
                    x = np.random.randint(1, w-1)
                    diff = np.mean(np.abs(
                        gray_image[:, x].astype(float) - 
                        gray_image[:, x-1].astype(float)
                    ))
                else:  # Horizontal
                    y = np.random.randint(1, h-1)
                    diff = np.mean(np.abs(
                        gray_image[y, :].astype(float) - 
                        gray_image[y-1, :].astype(float)
                    ))
                random_diffs.append(diff)

            # Blocking score
            if vertical_diffs or horizontal_diffs:
                block_diffs = vertical_diffs + horizontal_diffs
                block_mean = np.mean(block_diffs)
                random_mean = np.mean(random_diffs) if random_diffs else block_mean

                blocking_score = max(0, (block_mean - random_mean) / (block_mean + 1e-8))
                return float(min(blocking_score, 1.0))

            return 0.0

        except Exception as e:
            logger.warning(f"Blocking artifact detection failed: {e}")
            return 0.1

    def _analyze_dct_artifacts(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Analyze DCT coefficient artifacts"""
        try:
            h, w = gray_image.shape
            block_size = self.jpeg_block_size

            dct_scores = []
            high_freq_ratios = []
            quantization_indicators = []

            # Process 8x8 blocks
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = gray_image[y:y+block_size, x:x+block_size].astype(np.float32)

                    # Apply DCT
                    dct_block = cv2.dct(block)

                    # High frequency energy
                    high_freq_mask = np.zeros_like(dct_block)
                    high_freq_mask[4:, 4:] = 1
                    high_freq_energy = np.sum(dct_block * high_freq_mask) ** 2
                    total_energy = np.sum(dct_block ** 2)

                    if total_energy > 0:
                        high_freq_ratio = high_freq_energy / total_energy
                        high_freq_ratios.append(high_freq_ratio)

                    # Quantization artifacts
                    # Count near-zero coefficients
                    near_zero = np.sum(np.abs(dct_block) < self.dct_threshold)
                    quantization_indicators.append(near_zero / 64.0)  # Normalize by block size

            artifacts = {}
            if high_freq_ratios:
                artifacts['dct_high_freq_ratio'] = float(np.mean(high_freq_ratios))
                artifacts['dct_high_freq_variance'] = float(np.var(high_freq_ratios))

            if quantization_indicators:
                artifacts['dct_quantization_ratio'] = float(np.mean(quantization_indicators))

            return artifacts

        except Exception as e:
            logger.warning(f"DCT artifact analysis failed: {e}")
            return {
                'dct_high_freq_ratio': 0.1,
                'dct_high_freq_variance': 0.05,
                'dct_quantization_ratio': 0.7
            }

    def _detect_ringing_artifacts(self, gray_image: np.ndarray) -> float:
        """Detect ringing artifacts around edges"""
        try:
            # Detect edges
            edges = cv2.Canny(gray_image, 50, 150)

            # Dilate edges to get edge regions
            kernel = np.ones((5, 5), np.uint8)
            edge_regions = cv2.dilate(edges, kernel, iterations=1)

            # Apply Laplacian to detect oscillations
            laplacian = cv2.Laplacian(gray_image.astype(np.float32), cv2.CV_32F)

            # Measure ringing in edge regions
            edge_pixels = edge_regions > 0
            if np.sum(edge_pixels) > 0:
                edge_laplacian = laplacian[edge_pixels]
                ringing_score = np.std(edge_laplacian) / (np.mean(np.abs(edge_laplacian)) + 1e-8)
                return float(min(ringing_score / 10.0, 1.0))  # Normalize

            return 0.0

        except Exception as e:
            logger.warning(f"Ringing artifact detection failed: {e}")
            return 0.1

    def _detect_mosquito_noise(self, gray_image: np.ndarray) -> float:
        """Detect mosquito noise artifacts"""
        try:
            # Apply high-pass filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_pass = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)

            # Detect high-frequency noise in smooth regions
            # Smooth regions have low gradient
            grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Find smooth regions (low gradient)
            smooth_threshold = np.percentile(gradient_magnitude, 25)  # Bottom 25%
            smooth_regions = gradient_magnitude < smooth_threshold

            # Measure noise in smooth regions
            if np.sum(smooth_regions) > 0:
                noise_in_smooth = high_pass[smooth_regions]
                mosquito_score = np.std(noise_in_smooth)
                return float(min(mosquito_score / 50.0, 1.0))  # Normalize

            return 0.0

        except Exception as e:
            logger.warning(f"Mosquito noise detection failed: {e}")
            return 0.1

    def _estimate_jpeg_quality(self, gray_image: np.ndarray) -> float:
        """Estimate JPEG quality level"""
        try:
            # Analyze blocking artifacts and DCT characteristics
            blocking = self._detect_blocking_artifacts(gray_image)
            dct_artifacts = self._analyze_dct_artifacts(gray_image)

            # Combine metrics to estimate quality
            high_freq_ratio = dct_artifacts.get('dct_high_freq_ratio', 0.1)
            quantization_ratio = dct_artifacts.get('dct_quantization_ratio', 0.7)

            # Quality estimation (inverse relationship with artifacts)
            quality_indicators = [
                (1.0 - blocking) * 100,           # Less blocking = higher quality
                high_freq_ratio * 200,            # More high freq = higher quality
                (1.0 - quantization_ratio) * 100  # Less quantization = higher quality
            ]

            estimated_quality = np.mean(quality_indicators)
            return float(np.clip(estimated_quality, 10, 100))

        except Exception as e:
            logger.warning(f"JPEG quality estimation failed: {e}")
            return 75.0

    def detect_blending_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect blending and composition artifacts

        Args:
            image: Input image (RGB)

        Returns:
            Dictionary of blending artifact measures
        """
        artifacts = {}

        try:
            # Edge inconsistency analysis
            edge_artifacts = self._analyze_edge_inconsistencies(image)
            artifacts.update(edge_artifacts)

            # Color inconsistency analysis
            color_artifacts = self._analyze_color_inconsistencies(image)
            artifacts.update(color_artifacts)

            # Illumination inconsistency
            illumination_artifacts = self._analyze_illumination_inconsistencies(image)
            artifacts.update(illumination_artifacts)

            # Boundary detection
            boundary_artifacts = self._detect_composition_boundaries(image)
            artifacts.update(boundary_artifacts)

        except Exception as e:
            logger.warning(f"Blending artifact detection failed: {e}")
            artifacts = {
                'edge_inconsistency': 0.1, 'color_inconsistency': 0.1,
                'illumination_inconsistency': 0.1, 'boundary_artifacts': 0.1
            }

        return artifacts

    def _analyze_edge_inconsistencies(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze edge consistency across the image"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Multi-scale edge analysis
            edge_responses = []

            for kernel_name, kernel in self.edge_kernels.items():
                if kernel is not None:
                    edges = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                else:  # Canny
                    edges = cv2.Canny(gray, 50, 150).astype(np.float32)

                edge_responses.append(edges)

            # Analyze edge strength variations
            edge_variations = []
            for edges in edge_responses:
                # Divide image into regions
                h, w = edges.shape
                region_size = min(h, w) // 8

                regional_strengths = []
                for y in range(0, h - region_size + 1, region_size):
                    for x in range(0, w - region_size + 1, region_size):
                        region = edges[y:y+region_size, x:x+region_size]
                        strength = np.mean(np.abs(region))
                        regional_strengths.append(strength)

                if regional_strengths:
                    variation = np.std(regional_strengths) / (np.mean(regional_strengths) + 1e-8)
                    edge_variations.append(variation)

            artifacts = {}
            if edge_variations:
                artifacts['edge_inconsistency'] = float(np.mean(edge_variations))

            return artifacts

        except Exception as e:
            logger.warning(f"Edge inconsistency analysis failed: {e}")
            return {'edge_inconsistency': 0.1}

    def _analyze_color_inconsistencies(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze color consistency across the image"""
        try:
            if len(image.shape) != 3:
                return {'color_inconsistency': 0.0}

            color_inconsistencies = []

            # Analyze different color spaces
            color_spaces = {
                'RGB': image,
                'HSV': cv2.cvtColor(image, cv2.COLOR_RGB2HSV),
                'LAB': cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            }

            for space_name, color_img in color_spaces.items():
                # Analyze each channel
                for channel in range(3):
                    channel_img = color_img[:, :, channel]

                    # Regional analysis
                    h, w = channel_img.shape
                    region_size = min(h, w) // 6

                    regional_stats = []
                    for y in range(0, h - region_size + 1, region_size):
                        for x in range(0, w - region_size + 1, region_size):
                            region = channel_img[y:y+region_size, x:x+region_size]
                            mean_val = np.mean(region)
                            std_val = np.std(region)
                            regional_stats.append((mean_val, std_val))

                    if regional_stats:
                        means = [stat[0] for stat in regional_stats]
                        stds = [stat[1] for stat in regional_stats]

                        # Inconsistency in means and standard deviations
                        mean_inconsistency = np.std(means) / (np.mean(means) + 1e-8)
                        std_inconsistency = np.std(stds) / (np.mean(stds) + 1e-8)

                        channel_inconsistency = (mean_inconsistency + std_inconsistency) / 2
                        color_inconsistencies.append(channel_inconsistency)

            artifacts = {}
            if color_inconsistencies:
                artifacts['color_inconsistency'] = float(np.mean(color_inconsistencies))

            return artifacts

        except Exception as e:
            logger.warning(f"Color inconsistency analysis failed: {e}")
            return {'color_inconsistency': 0.1}

    def _analyze_illumination_inconsistencies(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze illumination consistency"""
        try:
            if len(image.shape) == 3:
                # Use luminance channel
                yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                luminance = yuv[:, :, 0]
            else:
                luminance = image

            # Analyze illumination gradients
            grad_x = cv2.Sobel(luminance.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(luminance.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)

            # Analyze gradient directions
            gradient_angles = np.arctan2(grad_y, grad_x)
            gradient_magnitudes = np.sqrt(grad_x**2 + grad_y**2)

            # Regional gradient analysis
            h, w = luminance.shape
            region_size = min(h, w) // 8

            regional_illumination_patterns = []
            for y in range(0, h - region_size + 1, region_size):
                for x in range(0, w - region_size + 1, region_size):
                    region_angles = gradient_angles[y:y+region_size, x:x+region_size]
                    region_magnitudes = gradient_magnitudes[y:y+region_size, x:x+region_size]

                    # Weighted average angle (weighted by gradient magnitude)
                    if np.sum(region_magnitudes) > 0:
                        avg_angle = np.average(region_angles, weights=region_magnitudes)
                        regional_illumination_patterns.append(avg_angle)

            # Measure consistency of illumination patterns
            if len(regional_illumination_patterns) > 1:
                # Circular statistics for angles
                angle_consistency = np.std(regional_illumination_patterns)
                illumination_inconsistency = angle_consistency / np.pi  # Normalize to [0, 1]
            else:
                illumination_inconsistency = 0.0

            return {'illumination_inconsistency': float(illumination_inconsistency)}

        except Exception as e:
            logger.warning(f"Illumination inconsistency analysis failed: {e}")
            return {'illumination_inconsistency': 0.1}

    def _detect_composition_boundaries(self, image: np.ndarray) -> Dict[str, float]:
        """Detect artificial composition boundaries"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Edge detection with multiple scales
            edges_fine = cv2.Canny(gray, 50, 150)
            edges_coarse = cv2.Canny(gray, 100, 200)

            # Find strong continuous edges (potential boundaries)
            # Dilate and erode to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            connected_edges = cv2.morphologyEx(edges_fine, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(connected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze contour characteristics
            suspicious_boundaries = 0
            total_contour_length = 0

            for contour in contours:
                # Calculate contour properties
                length = cv2.arcLength(contour, False)
                area = cv2.contourArea(contour)

                if length > 50:  # Only consider significant contours
                    total_contour_length += length

                    # Check if contour is suspiciously straight or geometric
                    epsilon = 0.02 * length
                    approx = cv2.approxPolyDP(contour, epsilon, False)

                    # Very few vertices suggest artificial boundaries
                    if len(approx) < length / 20:  # Fewer vertices than expected
                        suspicious_boundaries += 1

            # Boundary artifact score
            if len(contours) > 0:
                boundary_score = suspicious_boundaries / len(contours)
            else:
                boundary_score = 0.0

            return {'boundary_artifacts': float(boundary_score)}

        except Exception as e:
            logger.warning(f"Composition boundary detection failed: {e}")
            return {'boundary_artifacts': 0.1}

    def detect_gan_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect GAN synthesis artifacts

        Args:
            image: Input image

        Returns:
            Dictionary of GAN artifact measures
        """
        artifacts = {}

        try:
            # Spectral analysis for GAN artifacts
            spectral_artifacts = self._analyze_spectral_artifacts(image)
            artifacts.update(spectral_artifacts)

            # Checkerboard artifact detection
            checkerboard_artifacts = self._detect_checkerboard_artifacts(image)
            artifacts.update(checkerboard_artifacts)

            # Upsampling artifact detection
            upsampling_artifacts = self._detect_upsampling_artifacts(image)
            artifacts.update(upsampling_artifacts)

            # Texture inconsistency
            texture_artifacts = self._analyze_texture_inconsistencies(image)
            artifacts.update(texture_artifacts)

        except Exception as e:
            logger.warning(f"GAN artifact detection failed: {e}")
            artifacts = {
                'spectral_artifacts': 0.1, 'checkerboard_artifacts': 0.1,
                'upsampling_artifacts': 0.1, 'texture_inconsistencies': 0.1
            }

        return artifacts

    def _analyze_spectral_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze frequency domain artifacts typical of GANs"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            # Log scale for better analysis
            log_spectrum = np.log(magnitude_spectrum + 1)

            # Analyze frequency bands
            h, w = log_spectrum.shape
            center_y, center_x = h // 2, w // 2

            band_energies = []
            for low_freq, high_freq in self.gan_frequency_bands:
                # Create annular mask for frequency band
                y, x = np.ogrid[:h, :w]
                distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

                band_mask = (distances >= low_freq) & (distances < high_freq)
                if np.sum(band_mask) > 0:
                    band_energy = np.mean(log_spectrum[band_mask])
                    band_energies.append(band_energy)

            # Analyze spectral characteristics
            artifacts = {}
            if band_energies:
                # High frequency suppression (common in GANs)
                hf_suppression = band_energies[0] / (band_energies[-1] + 1e-8)
                artifacts['high_freq_suppression'] = float(min(hf_suppression / 5.0, 1.0))

                # Spectral irregularities
                energy_variation = np.std(band_energies) / (np.mean(band_energies) + 1e-8)
                artifacts['spectral_irregularity'] = float(min(energy_variation, 1.0))

            return artifacts

        except Exception as e:
            logger.warning(f"Spectral artifact analysis failed: {e}")
            return {'high_freq_suppression': 0.1, 'spectral_irregularity': 0.1}

    def _detect_checkerboard_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """Detect checkerboard patterns from transpose convolution"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            checkerboard_scores = []

            for size in self.checkerboard_sizes:
                # Create checkerboard kernel
                kernel = np.zeros((size*2, size*2))
                kernel[::2, ::2] = 1
                kernel[1::2, 1::2] = 1
                kernel = kernel * 2 - 1  # Convert to [-1, 1]

                # Convolve with image
                correlation = cv2.filter2D(gray.astype(np.float32), -1, kernel)

                # Measure correlation strength
                correlation_strength = np.mean(np.abs(correlation))
                checkerboard_scores.append(correlation_strength)

            # Overall checkerboard artifact score
            if checkerboard_scores:
                max_score = max(checkerboard_scores)
                artifacts = {'checkerboard_artifacts': float(min(max_score / 100.0, 1.0))}
            else:
                artifacts = {'checkerboard_artifacts': 0.1}

            return artifacts

        except Exception as e:
            logger.warning(f"Checkerboard artifact detection failed: {e}")
            return {'checkerboard_artifacts': 0.1}

    def _detect_upsampling_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """Detect upsampling artifacts"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Downsampling followed by different upsampling methods
            h, w = gray.shape
            small_size = (w // 4, h // 4)

            # Downsample
            small_img = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)

            # Upsample with different methods
            upsampled_imgs = {}
            upsampled_imgs['nearest'] = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)
            upsampled_imgs['bilinear'] = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)
            upsampled_imgs['bicubic'] = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_CUBIC)

            # Measure similarity to upsampled versions
            similarities = {}
            for method, upsampled in upsampled_imgs.items():
                # Compute normalized cross-correlation
                correlation = cv2.matchTemplate(gray.astype(np.float32), upsampled.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                max_correlation = np.max(correlation)
                similarities[method] = max_correlation

            # High similarity suggests upsampling artifacts
            max_similarity = max(similarities.values())
            upsampling_score = max(0, (max_similarity - 0.7) / 0.3)  # Threshold at 0.7

            return {'upsampling_artifacts': float(min(upsampling_score, 1.0))}

        except Exception as e:
            logger.warning(f"Upsampling artifact detection failed: {e}")
            return {'upsampling_artifacts': 0.1}

    def _analyze_texture_inconsistencies(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture consistency typical of GAN outputs"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Local Binary Pattern analysis
            from skimage.feature import local_binary_pattern

            # Multiple scales
            texture_inconsistencies = []

            for radius in [1, 2, 3]:
                n_points = 8 * radius
                lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

                # Regional texture analysis
                h, w = lbp.shape
                region_size = min(h, w) // 8

                regional_textures = []
                for y in range(0, h - region_size + 1, region_size):
                    for x in range(0, w - region_size + 1, region_size):
                        region = lbp[y:y+region_size, x:x+region_size]

                        # Texture histogram
                        hist, _ = np.histogram(region, bins=n_points+2, range=(0, n_points+2))
                        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-8)

                        # Texture entropy
                        entropy = -np.sum(hist * np.log2(hist + 1e-10))
                        regional_textures.append(entropy)

                # Texture inconsistency
                if regional_textures:
                    texture_variation = np.std(regional_textures) / (np.mean(regional_textures) + 1e-8)
                    texture_inconsistencies.append(texture_variation)

            # Overall texture inconsistency
            if texture_inconsistencies:
                overall_inconsistency = np.mean(texture_inconsistencies)
                return {'texture_inconsistencies': float(min(overall_inconsistency, 1.0))}
            else:
                return {'texture_inconsistencies': 0.1}

        except Exception as e:
            logger.warning(f"Texture inconsistency analysis failed: {e}")
            return {'texture_inconsistencies': 0.1}

    def analyze_all_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive artifact analysis

        Args:
            image: Input image

        Returns:
            Dictionary of all artifact measures
        """
        all_artifacts = {}

        try:
            # JPEG artifacts
            jpeg_artifacts = self.detect_jpeg_artifacts(image)
            all_artifacts.update(jpeg_artifacts)

            # Blending artifacts
            blending_artifacts = self.detect_blending_artifacts(image)
            all_artifacts.update(blending_artifacts)

            # GAN artifacts
            gan_artifacts = self.detect_gan_artifacts(image)
            all_artifacts.update(gan_artifacts)

            # Summary scores
            artifact_categories = {
                'jpeg': [k for k in jpeg_artifacts.keys()],
                'blending': [k for k in blending_artifacts.keys()],
                'gan': [k for k in gan_artifacts.keys()]
            }

            category_scores = {}
            for category, keys in artifact_categories.items():
                if keys:
                    category_values = [all_artifacts[k] for k in keys if k in all_artifacts]
                    if category_values:
                        category_scores[f'{category}_artifact_score'] = float(np.mean(category_values))

            all_artifacts.update(category_scores)

            # Overall artifact score
            all_values = [v for v in all_artifacts.values() if isinstance(v, (int, float))]
            if all_values:
                all_artifacts['overall_artifact_score'] = float(np.mean(all_values))

        except Exception as e:
            logger.error(f"Comprehensive artifact analysis failed: {e}")
            all_artifacts = {
                'jpeg_artifact_score': 0.1,
                'blending_artifact_score': 0.1,
                'gan_artifact_score': 0.1,
                'overall_artifact_score': 0.1
            }

        return all_artifacts
