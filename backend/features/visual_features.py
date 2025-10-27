"""
Visual Feature Extractor for Multi-Modal Deepfake Detection
Specialized extraction of visual artifacts, texture patterns, and spatial inconsistencies
Supports advanced analysis of facial manipulation artifacts and compression traces
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from scipy import ndimage, signal
from skimage import feature, measure, filters
from PIL import Image, ImageFilter, ImageEnhance
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class VisualFeatureExtractor:
    """
    Advanced visual feature extractor for deepfake detection
    Focuses on manipulation artifacts, texture inconsistencies, and spatial anomalies
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize Visual Feature Extractor

        Args:
            device: Device for computation ('cpu' or 'cuda')
        """
        self.device = device

        # Initialize feature extraction components
        self._init_filters()
        self._init_detectors()

        logger.info("Initialized VisualFeatureExtractor")

    def _init_filters(self):
        """Initialize various filters for feature extraction"""

        # Sobel operators for edge detection
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Prewitt operators
        self.prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        self.prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        # Laplacian kernel for blob detection
        self.laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

        # Gabor filter parameters
        self.gabor_frequencies = [0.1, 0.3, 0.5, 0.7]
        self.gabor_orientations = [0, 45, 90, 135]

    def _init_detectors(self):
        """Initialize specialized detectors"""

        # JPEG compression detection patterns
        self.jpeg_block_size = 8

        # Noise detection parameters
        self.noise_kernel_sizes = [3, 5, 7]

        # Face landmark detection (if available)
        try:
            import dlib
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if Path(predictor_path).exists():
                self.face_predictor = dlib.shape_predictor(predictor_path)
                self.face_detector = dlib.get_frontal_face_detector()
                self.dlib_available = True
            else:
                self.dlib_available = False
                logger.info("dlib predictor not found, using alternative methods")
        except ImportError:
            self.dlib_available = False
            logger.info("dlib not available")

    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture-based features using Local Binary Patterns and Gabor filters

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Dictionary of texture features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        features = {}

        try:
            # Local Binary Patterns
            radius = 3
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float) / np.sum(lbp_hist)

            features.update({
                'lbp_uniformity': np.sum(lbp_hist[:-1]),  # Exclude non-uniform patterns
                'lbp_entropy': -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)),
                'lbp_energy': np.sum(lbp_hist ** 2),
                'lbp_contrast': np.var(lbp_hist)
            })

        except Exception as e:
            logger.warning(f"LBP extraction failed: {e}")
            features.update({
                'lbp_uniformity': 0.5, 'lbp_entropy': 3.0,
                'lbp_energy': 0.1, 'lbp_contrast': 0.1
            })

        try:
            # Gabor filter responses
            gabor_responses = []
            for freq in self.gabor_frequencies:
                for theta in np.radians(self.gabor_orientations):
                    real, _ = filters.gabor(gray, frequency=freq, theta=theta)
                    gabor_responses.append(real)

            if gabor_responses:
                gabor_mean = np.mean([np.mean(resp) for resp in gabor_responses])
                gabor_std = np.mean([np.std(resp) for resp in gabor_responses])
                gabor_energy = np.mean([np.sum(resp ** 2) for resp in gabor_responses])

                features.update({
                    'gabor_mean_response': float(gabor_mean),
                    'gabor_std_response': float(gabor_std),
                    'gabor_energy': float(gabor_energy / 1e6)  # Normalize
                })

        except Exception as e:
            logger.warning(f"Gabor filter extraction failed: {e}")
            features.update({
                'gabor_mean_response': 0.0,
                'gabor_std_response': 0.1,
                'gabor_energy': 0.1
            })

        # GLCM (Gray-Level Co-occurrence Matrix) features
        try:
            from skimage.feature import graycomatrix, graycoprops

            # Compute GLCM
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]

            glcm = graycomatrix(
                (gray / 4).astype(np.uint8),  # Reduce levels for efficiency
                distances=distances,
                angles=np.radians(angles),
                levels=64,
                symmetric=True,
                normed=True
            )

            # Extract GLCM properties
            contrast = np.mean(graycoprops(glcm, 'contrast'))
            dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            energy = np.mean(graycoprops(glcm, 'energy'))
            correlation = np.mean(graycoprops(glcm, 'correlation'))

            features.update({
                'glcm_contrast': float(contrast),
                'glcm_dissimilarity': float(dissimilarity),
                'glcm_homogeneity': float(homogeneity),
                'glcm_energy': float(energy),
                'glcm_correlation': float(correlation)
            })

        except Exception as e:
            logger.warning(f"GLCM extraction failed: {e}")
            features.update({
                'glcm_contrast': 0.5, 'glcm_dissimilarity': 0.3,
                'glcm_homogeneity': 0.4, 'glcm_energy': 0.2,
                'glcm_correlation': 0.1
            })

        return features

    def extract_compression_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract features related to compression artifacts

        Args:
            image: Input image

        Returns:
            Dictionary of compression-related features
        """
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # JPEG blocking artifacts detection
        block_size = self.jpeg_block_size
        h, w = gray.shape

        # Analyze 8x8 block boundaries
        vertical_diffs = []
        horizontal_diffs = []

        # Vertical block boundaries
        for x in range(block_size, w, block_size):
            if x < w - 1:
                diff = np.mean(np.abs(gray[:, x] - gray[:, x-1]))
                vertical_diffs.append(diff)

        # Horizontal block boundaries
        for y in range(block_size, h, block_size):
            if y < h - 1:
                diff = np.mean(np.abs(gray[y, :] - gray[y-1, :]))
                horizontal_diffs.append(diff)

        # Compare with random boundaries
        random_diffs = []
        for _ in range(len(vertical_diffs) + len(horizontal_diffs)):
            if np.random.choice([True, False]):  # Vertical
                x = np.random.randint(1, w-1)
                diff = np.mean(np.abs(gray[:, x] - gray[:, x-1]))
            else:  # Horizontal
                y = np.random.randint(1, h-1)
                diff = np.mean(np.abs(gray[y, :] - gray[y-1, :]))
            random_diffs.append(diff)

        # Blocking artifact score
        if vertical_diffs or horizontal_diffs:
            block_diffs = vertical_diffs + horizontal_diffs
            block_mean = np.mean(block_diffs)
            random_mean = np.mean(random_diffs) if random_diffs else block_mean

            blocking_score = max(0, (block_mean - random_mean) / (block_mean + 1e-8))
            features['jpeg_blocking_score'] = float(min(blocking_score, 1.0))
        else:
            features['jpeg_blocking_score'] = 0.0

        # DCT coefficient analysis
        try:
            # Divide image into 8x8 blocks and analyze DCT
            dct_scores = []

            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = gray[y:y+block_size, x:x+block_size].astype(np.float32)

                    # Apply DCT
                    dct_block = cv2.dct(block)

                    # Analyze high-frequency coefficients
                    high_freq_mask = np.zeros_like(dct_block)
                    high_freq_mask[4:, 4:] = 1
                    high_freq_energy = np.sum(dct_block * high_freq_mask) ** 2
                    total_energy = np.sum(dct_block ** 2)

                    if total_energy > 0:
                        dct_scores.append(high_freq_energy / total_energy)

            if dct_scores:
                features.update({
                    'dct_high_freq_ratio': float(np.mean(dct_scores)),
                    'dct_coefficient_variance': float(np.var(dct_scores)),
                    'dct_sparsity': float(np.mean([np.sum(score < 0.01) for score in dct_scores]))
                })

        except Exception as e:
            logger.warning(f"DCT analysis failed: {e}")
            features.update({
                'dct_high_freq_ratio': 0.1,
                'dct_coefficient_variance': 0.05,
                'dct_sparsity': 0.7
            })

        # Quantization artifacts
        try:
            # Look for quantization steps in histogram
            hist, bins = np.histogram(gray, bins=256, range=(0, 256))

            # Find peaks in histogram that might indicate quantization
            peaks = signal.find_peaks(hist, height=np.max(hist) * 0.1)[0]

            if len(peaks) > 1:
                peak_distances = np.diff(peaks)
                uniform_spacing = np.std(peak_distances) / np.mean(peak_distances) if np.mean(peak_distances) > 0 else 1
                features['quantization_uniformity'] = float(1.0 / (1.0 + uniform_spacing))
            else:
                features['quantization_uniformity'] = 0.0

        except Exception as e:
            logger.warning(f"Quantization analysis failed: {e}")
            features['quantization_uniformity'] = 0.5

        return features

    def extract_noise_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract noise-related features

        Args:
            image: Input image

        Returns:
            Dictionary of noise features
        """
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # High-pass filtering for noise detection
        try:
            # Apply Gaussian blur and subtract from original
            blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)
            high_pass = gray.astype(np.float32) - blurred

            # Noise variance estimation
            noise_variance = np.var(high_pass)
            features['noise_variance'] = float(noise_variance / 100.0)  # Normalize

            # Noise distribution analysis
            noise_hist, _ = np.histogram(high_pass, bins=50)
            noise_hist = noise_hist / np.sum(noise_hist)

            # Measure deviation from Gaussian
            x = np.linspace(-3, 3, 50)
            gaussian = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            gaussian = gaussian / np.sum(gaussian)

            gaussian_deviation = np.sum(np.abs(noise_hist - gaussian))
            features['noise_non_gaussianity'] = float(gaussian_deviation)

        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            features.update({
                'noise_variance': 0.1,
                'noise_non_gaussianity': 0.2
            })

        # Multi-scale noise analysis
        try:
            noise_scales = []

            for kernel_size in self.noise_kernel_sizes:
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
                smoothed = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                noise = gray.astype(np.float32) - smoothed
                noise_energy = np.mean(noise ** 2)
                noise_scales.append(noise_energy)

            features.update({
                'multi_scale_noise_mean': float(np.mean(noise_scales)),
                'multi_scale_noise_std': float(np.std(noise_scales)),
                'noise_scale_ratio': float(noise_scales[0] / (noise_scales[-1] + 1e-8))
            })

        except Exception as e:
            logger.warning(f"Multi-scale noise analysis failed: {e}")
            features.update({
                'multi_scale_noise_mean': 0.05,
                'multi_scale_noise_std': 0.02,
                'noise_scale_ratio': 1.0
            })

        return features

    def extract_face_manipulation_features(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Extract features specific to facial manipulation detection

        Args:
            image: Input image
            face_bbox: Face bounding box [x, y, width, height] (optional)

        Returns:
            Dictionary of face manipulation features
        """
        features = {}

        # Auto-detect face if bbox not provided
        if face_bbox is None:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                face_bbox = faces[0]  # Use first detected face
            else:
                # Use center region if no face detected
                h, w = gray.shape
                face_bbox = [w//4, h//4, w//2, h//2]

        if face_bbox is not None:
            x, y, w, h = face_bbox

            # Extract face region
            if len(image.shape) == 3:
                face_region = image[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            else:
                face_gray = image[y:y+h, x:x+w]

            if face_gray.size == 0:
                return {'face_manipulation_score': 0.0}

            # Skin texture analysis
            try:
                # Analyze texture smoothness (over-smoothing in deepfakes)
                laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                features['face_texture_variance'] = float(laplacian_var / 1000.0)

                # Edge density (manipulated faces often have fewer natural edges)
                edges = cv2.Canny(face_gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                features['face_edge_density'] = float(edge_density)

                # Frequency domain analysis
                f_transform = np.fft.fft2(face_gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)

                # High frequency energy (often reduced in manipulated faces)
                fh, fw = magnitude_spectrum.shape
                high_freq_mask = np.zeros_like(magnitude_spectrum)
                center_y, center_x = fh // 2, fw // 2
                radius = min(fh, fw) // 4

                y, x = np.ogrid[:fh, :fw]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2
                high_freq_mask[mask] = 1

                high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
                total_energy = np.sum(magnitude_spectrum)

                features['face_high_freq_ratio'] = float(high_freq_energy / total_energy)

            except Exception as e:
                logger.warning(f"Face texture analysis failed: {e}")
                features.update({
                    'face_texture_variance': 0.5,
                    'face_edge_density': 0.1,
                    'face_high_freq_ratio': 0.2
                })

            # Asymmetry analysis
            try:
                # Split face vertically and compare halves
                face_left = face_gray[:, :w//2]
                face_right = np.fliplr(face_gray[:, w//2:])

                # Resize to same size
                if face_right.shape[1] != face_left.shape[1]:
                    min_width = min(face_left.shape[1], face_right.shape[1])
                    face_left = face_left[:, :min_width]
                    face_right = face_right[:, :min_width]

                # Calculate asymmetry
                if face_left.shape == face_right.shape:
                    asymmetry = np.mean(np.abs(face_left.astype(float) - face_right.astype(float)))
                    features['face_asymmetry'] = float(asymmetry / 255.0)
                else:
                    features['face_asymmetry'] = 0.1

            except Exception as e:
                logger.warning(f"Face asymmetry analysis failed: {e}")
                features['face_asymmetry'] = 0.1

        else:
            # No face detected
            features = {
                'face_texture_variance': 0.0,
                'face_edge_density': 0.0,
                'face_high_freq_ratio': 0.0,
                'face_asymmetry': 0.0
            }

        return features

    def extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract edge-based features for manipulation detection

        Args:
            image: Input image

        Returns:
            Dictionary of edge features
        """
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        try:
            # Multiple edge detection methods
            # Sobel edges
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # Canny edges
            canny_edges = cv2.Canny(gray, 50, 150)

            # Laplacian edges
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)

            # Edge statistics
            features.update({
                'sobel_edge_mean': float(np.mean(sobel_magnitude)),
                'sobel_edge_std': float(np.std(sobel_magnitude)),
                'canny_edge_density': float(np.sum(canny_edges > 0) / canny_edges.size),
                'laplacian_variance': float(laplacian.var()),
            })

            # Edge orientation analysis
            sobel_orientation = np.arctan2(sobel_y, sobel_x + 1e-8)

            # Histogram of orientations
            orientation_hist, _ = np.histogram(sobel_orientation, bins=8, range=(-np.pi, np.pi))
            orientation_hist = orientation_hist / np.sum(orientation_hist)

            # Orientation entropy (more uniform in manipulated images)
            orientation_entropy = -np.sum(orientation_hist * np.log2(orientation_hist + 1e-10))
            features['edge_orientation_entropy'] = float(orientation_entropy)

            # Edge continuity analysis
            edge_binary = (sobel_magnitude > np.mean(sobel_magnitude)).astype(np.uint8)

            # Find connected components
            num_labels, labels = cv2.connectedComponents(edge_binary)

            # Analyze component sizes
            if num_labels > 1:
                component_sizes = [np.sum(labels == i) for i in range(1, num_labels)]
                avg_component_size = np.mean(component_sizes)
                features['edge_component_avg_size'] = float(avg_component_size / edge_binary.size)
                features['edge_fragmentation'] = float((num_labels - 1) / edge_binary.size * 1000)
            else:
                features['edge_component_avg_size'] = 0.0
                features['edge_fragmentation'] = 0.0

        except Exception as e:
            logger.warning(f"Edge feature extraction failed: {e}")
            features = {
                'sobel_edge_mean': 10.0, 'sobel_edge_std': 5.0,
                'canny_edge_density': 0.1, 'laplacian_variance': 100.0,
                'edge_orientation_entropy': 2.5, 'edge_component_avg_size': 0.001,
                'edge_fragmentation': 1.0
            }

        return features

    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-based features

        Args:
            image: Input image (RGB)

        Returns:
            Dictionary of color features
        """
        features = {}

        if len(image.shape) != 3:
            # Grayscale image
            features = {
                'color_variance': 0.0, 'color_saturation_mean': 0.0,
                'color_hue_entropy': 0.0, 'color_channel_correlation': 0.0
            }
            return features

        try:
            # RGB statistics
            rgb_means = [np.mean(image[:, :, i]) for i in range(3)]
            rgb_stds = [np.std(image[:, :, i]) for i in range(3)]

            features.update({
                'rgb_mean_r': float(rgb_means[0] / 255.0),
                'rgb_mean_g': float(rgb_means[1] / 255.0),
                'rgb_mean_b': float(rgb_means[2] / 255.0),
                'rgb_std_r': float(rgb_stds[0] / 255.0),
                'rgb_std_g': float(rgb_stds[1] / 255.0),
                'rgb_std_b': float(rgb_stds[2] / 255.0)
            })

            # Channel correlations (unnatural correlations may indicate manipulation)
            r_channel = image[:, :, 0].flatten()
            g_channel = image[:, :, 1].flatten()
            b_channel = image[:, :, 2].flatten()

            rg_corr = np.corrcoef(r_channel, g_channel)[0, 1]
            rb_corr = np.corrcoef(r_channel, b_channel)[0, 1]
            gb_corr = np.corrcoef(g_channel, b_channel)[0, 1]

            features.update({
                'color_rg_correlation': float(rg_corr if not np.isnan(rg_corr) else 0.0),
                'color_rb_correlation': float(rb_corr if not np.isnan(rb_corr) else 0.0),
                'color_gb_correlation': float(gb_corr if not np.isnan(gb_corr) else 0.0)
            })

            # HSV analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]

            # Hue entropy
            hue_hist, _ = np.histogram(hue, bins=180, range=(0, 180))
            hue_hist = hue_hist / np.sum(hue_hist)
            hue_entropy = -np.sum(hue_hist * np.log2(hue_hist + 1e-10))

            features.update({
                'color_hue_entropy': float(hue_entropy),
                'color_saturation_mean': float(np.mean(saturation) / 255.0),
                'color_saturation_std': float(np.std(saturation) / 255.0),
                'color_value_mean': float(np.mean(value) / 255.0)
            })

        except Exception as e:
            logger.warning(f"Color feature extraction failed: {e}")
            features = {
                'rgb_mean_r': 0.5, 'rgb_mean_g': 0.5, 'rgb_mean_b': 0.5,
                'rgb_std_r': 0.1, 'rgb_std_g': 0.1, 'rgb_std_b': 0.1,
                'color_rg_correlation': 0.3, 'color_rb_correlation': 0.3,
                'color_gb_correlation': 0.3, 'color_hue_entropy': 5.0,
                'color_saturation_mean': 0.5, 'color_saturation_std': 0.2,
                'color_value_mean': 0.5
            }

        return features

    def extract_all_visual_features(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Extract comprehensive visual features

        Args:
            image: Input image
            face_bbox: Optional face bounding box

        Returns:
            Dictionary of all visual features
        """
        all_features = {}

        try:
            # Extract different feature categories
            texture_features = self.extract_texture_features(image)
            compression_features = self.extract_compression_artifacts(image)
            noise_features = self.extract_noise_features(image)
            face_features = self.extract_face_manipulation_features(image, face_bbox)
            edge_features = self.extract_edge_features(image)
            color_features = self.extract_color_features(image)

            # Combine all features
            all_features.update(texture_features)
            all_features.update(compression_features)
            all_features.update(noise_features)
            all_features.update(face_features)
            all_features.update(edge_features)
            all_features.update(color_features)

            # Add summary statistics
            feature_values = [v for v in all_features.values() if isinstance(v, (int, float))]
            if feature_values:
                all_features.update({
                    'visual_feature_mean': float(np.mean(feature_values)),
                    'visual_feature_std': float(np.std(feature_values)),
                    'visual_feature_count': len(feature_values)
                })

        except Exception as e:
            logger.error(f"Visual feature extraction failed: {e}")
            # Return minimal features
            all_features = {
                'texture_score': 0.5, 'compression_score': 0.5,
                'noise_score': 0.5, 'manipulation_score': 0.5,
                'visual_feature_count': 4
            }

        return all_features

    def compute_visual_quality_score(self, features: Dict[str, float]) -> float:
        """
        Compute overall visual quality score from features

        Args:
            features: Extracted visual features

        Returns:
            Quality score (0-1, higher = better quality)
        """
        try:
            quality_indicators = []

            # Texture quality
            if 'face_texture_variance' in features:
                texture_quality = min(features['face_texture_variance'] / 0.5, 1.0)
                quality_indicators.append(texture_quality)

            # Edge quality
            if 'canny_edge_density' in features:
                edge_quality = min(features['canny_edge_density'] / 0.2, 1.0)
                quality_indicators.append(edge_quality)

            # Noise quality (inverse - less noise is better)
            if 'noise_variance' in features:
                noise_quality = max(0, 1.0 - features['noise_variance'] / 0.1)
                quality_indicators.append(noise_quality)

            # Compression quality (inverse - less compression artifacts is better)
            if 'jpeg_blocking_score' in features:
                compression_quality = max(0, 1.0 - features['jpeg_blocking_score'])
                quality_indicators.append(compression_quality)

            if quality_indicators:
                return float(np.mean(quality_indicators))
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"Quality score computation failed: {e}")
            return 0.5
