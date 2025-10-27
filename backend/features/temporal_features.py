"""
Temporal Feature Extractor for Multi-Modal Deepfake Detection
Specialized analysis of temporal inconsistencies, motion patterns, and frame-to-frame changes
Focuses on detecting unnatural temporal artifacts in manipulated video sequences
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from scipy import signal, ndimage
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class TemporalFeatureExtractor:
    """
    Advanced temporal feature extractor for video deepfake detection
    Analyzes motion consistency, temporal smoothness, and frame-level changes
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize Temporal Feature Extractor

        Args:
            device: Device for computation ('cpu' or 'cuda')
        """
        self.device = device

        # Initialize optical flow parameters
        self._init_optical_flow()

        # Initialize temporal analysis parameters
        self._init_temporal_params()

        logger.info("Initialized TemporalFeatureExtractor")

    def _init_optical_flow(self):
        """Initialize optical flow computation parameters"""

        # Lucas-Kanade parameters
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        # Dense optical flow parameters (Farneback)
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

        # Feature detection parameters
        self.feature_params = {
            'maxCorners': 100,
            'qualityLevel': 0.01,
            'minDistance': 10,
            'blockSize': 7
        }

    def _init_temporal_params(self):
        """Initialize temporal analysis parameters"""

        # Smoothness analysis window
        self.smoothness_window = 5

        # Motion magnitude thresholds
        self.motion_thresholds = {
            'low': 0.5,
            'medium': 2.0,
            'high': 5.0
        }

        # Frequency analysis parameters
        self.temporal_freq_bands = [
            (0.1, 1.0),    # Very low frequency
            (1.0, 5.0),    # Low frequency  
            (5.0, 15.0),   # Medium frequency
            (15.0, 30.0)   # High frequency
        ]

    def compute_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray, 
                           method: str = 'farneback') -> np.ndarray:
        """
        Compute optical flow between consecutive frames

        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            method: Optical flow method ('farneback', 'lk')

        Returns:
            Optical flow field
        """
        # Convert to grayscale if needed
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev_frame

        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr_frame

        if method == 'farneback':
            # Dense optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, None, **self.farneback_params
            )
            if flow is None:
                flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2))

        elif method == 'lk':
            # Lucas-Kanade sparse optical flow
            # Detect feature points
            p0 = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)

            if p0 is not None:
                # Track features
                p1, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, p0, None, **self.lk_params
                )

                # Create dense flow field from sparse features
                flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2))

                if p1 is not None and status is not None:
                    good_old = p0[status == 1]
                    good_new = p1[status == 1]

                    if len(good_old) > 0:
                        for i, (old, new) in enumerate(zip(good_old, good_new)):
                            x, y = old.ravel().astype(int)
                            dx, dy = (new - old).ravel()
                            if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                                flow[y, x, 0] = dx
                                flow[y, x, 1] = dy
            else:
                flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2))

        else:
            raise ValueError(f"Unknown optical flow method: {method}")

        return flow

    def extract_motion_features(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Extract motion-based features from frame sequence

        Args:
            frames: List of video frames

        Returns:
            Dictionary of motion features
        """
        features = {}

        if len(frames) < 2:
            return {'motion_magnitude_mean': 0.0, 'motion_consistency': 0.0}

        try:
            # Compute optical flow for consecutive frames
            flows = []
            motion_magnitudes = []
            motion_angles = []

            for i in range(len(frames) - 1):
                flow = self.compute_optical_flow(frames[i], frames[i + 1])
                flows.append(flow)

                # Motion magnitude and angle
                magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])

                motion_magnitudes.append(magnitude)
                motion_angles.append(angle)

            # Motion statistics
            all_magnitudes = np.concatenate([m.flatten() for m in motion_magnitudes])
            all_angles = np.concatenate([a.flatten() for a in motion_angles])

            features.update({
                'motion_magnitude_mean': float(np.mean(all_magnitudes)),
                'motion_magnitude_std': float(np.std(all_magnitudes)),
                'motion_magnitude_max': float(np.max(all_magnitudes)),
                'motion_magnitude_median': float(np.median(all_magnitudes))
            })

            # Motion consistency analysis
            if len(motion_magnitudes) > 1:
                # Frame-to-frame motion consistency
                consistency_scores = []

                for i in range(len(motion_magnitudes) - 1):
                    mag1 = motion_magnitudes[i]
                    mag2 = motion_magnitudes[i + 1]

                    # Compute correlation
                    if mag1.size > 0 and mag2.size > 0:
                        corr = np.corrcoef(mag1.flatten(), mag2.flatten())[0, 1]
                        if not np.isnan(corr):
                            consistency_scores.append(corr)

                if consistency_scores:
                    features['motion_consistency'] = float(np.mean(consistency_scores))
                else:
                    features['motion_consistency'] = 0.0

            # Motion direction analysis
            angle_hist, _ = np.histogram(all_angles, bins=8, range=(-np.pi, np.pi))
            angle_hist = angle_hist / np.sum(angle_hist)

            # Direction entropy
            angle_entropy = -np.sum(angle_hist * np.log2(angle_hist + 1e-10))
            features['motion_direction_entropy'] = float(angle_entropy)

            # Dominant motion direction
            dominant_direction = np.argmax(angle_hist)
            features['motion_dominant_direction'] = float(dominant_direction / 8.0)

            # Motion smoothness (temporal derivative of motion)
            if len(motion_magnitudes) > 2:
                motion_accelerations = []

                for i in range(len(motion_magnitudes) - 2):
                    mag_curr = motion_magnitudes[i + 1]
                    mag_prev = motion_magnitudes[i]
                    mag_next = motion_magnitudes[i + 2]

                    # Second derivative approximation
                    acceleration = mag_next - 2 * mag_curr + mag_prev
                    motion_accelerations.append(np.mean(np.abs(acceleration)))

                if motion_accelerations:
                    features['motion_smoothness'] = float(1.0 / (1.0 + np.mean(motion_accelerations)))
                else:
                    features['motion_smoothness'] = 0.5

        except Exception as e:
            logger.warning(f"Motion feature extraction failed: {e}")
            features = {
                'motion_magnitude_mean': 1.0, 'motion_magnitude_std': 0.5,
                'motion_magnitude_max': 2.0, 'motion_magnitude_median': 0.8,
                'motion_consistency': 0.5, 'motion_direction_entropy': 2.5,
                'motion_dominant_direction': 0.5, 'motion_smoothness': 0.5
            }

        return features

    def extract_temporal_consistency_features(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Extract temporal consistency features

        Args:
            frames: List of video frames

        Returns:
            Dictionary of temporal consistency features
        """
        features = {}

        if len(frames) < 3:
            return {'temporal_consistency_score': 0.5}

        try:
            # Convert frames to grayscale for analysis
            gray_frames = []
            for frame in frames:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                gray_frames.append(gray)

            # Frame-to-frame differences
            frame_diffs = []
            for i in range(len(gray_frames) - 1):
                diff = np.mean(np.abs(gray_frames[i+1].astype(float) - gray_frames[i].astype(float)))
                frame_diffs.append(diff)

            # Temporal smoothness
            diff_std = np.std(frame_diffs)
            diff_mean = np.mean(frame_diffs)
            temporal_smoothness = 1.0 / (1.0 + diff_std / (diff_mean + 1e-8))
            features['temporal_smoothness'] = float(temporal_smoothness)

            # Sudden change detection
            diff_threshold = diff_mean + 2 * diff_std
            sudden_changes = np.sum(np.array(frame_diffs) > diff_threshold)
            features['sudden_change_ratio'] = float(sudden_changes / len(frame_diffs))

            # Autocorrelation analysis
            if len(frame_diffs) > 10:
                # Compute autocorrelation of frame differences
                autocorr = np.correlate(frame_diffs, frame_diffs, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                # Normalize
                autocorr = autocorr / autocorr[0]

                # Find first minimum (indicates temporal period)
                if len(autocorr) > 3:
                    min_idx = np.argmin(autocorr[1:5]) + 1
                    features['temporal_autocorr_min'] = float(autocorr[min_idx])
                    features['temporal_period_estimate'] = float(min_idx)
                else:
                    features['temporal_autocorr_min'] = 0.8
                    features['temporal_period_estimate'] = 2.0

            # Flickering detection
            # Analyze brightness variations
            brightness_values = [np.mean(frame) for frame in gray_frames]
            brightness_diff = np.diff(brightness_values)

            # Detect high-frequency flickering
            flicker_score = np.std(brightness_diff) / (np.mean(brightness_values) + 1e-8)
            features['flickering_score'] = float(flicker_score)

            # Color consistency (if color frames)
            if len(frames[0].shape) == 3:
                color_consistency_scores = []

                for channel in range(3):
                    channel_values = [np.mean(frame[:, :, channel]) for frame in frames]
                    channel_std = np.std(channel_values)
                    channel_mean = np.mean(channel_values)
                    consistency = 1.0 / (1.0 + channel_std / (channel_mean + 1e-8))
                    color_consistency_scores.append(consistency)

                features['color_temporal_consistency'] = float(np.mean(color_consistency_scores))

        except Exception as e:
            logger.warning(f"Temporal consistency extraction failed: {e}")
            features = {
                'temporal_smoothness': 0.5, 'sudden_change_ratio': 0.1,
                'temporal_autocorr_min': 0.8, 'temporal_period_estimate': 2.0,
                'flickering_score': 0.1, 'color_temporal_consistency': 0.7
            }

        return features

    def extract_face_temporal_features(self, frames: List[np.ndarray], face_bboxes: List[Optional[List[int]]] = None) -> Dict[str, float]:
        """
        Extract temporal features specific to facial regions

        Args:
            frames: List of video frames
            face_bboxes: List of face bounding boxes for each frame

        Returns:
            Dictionary of face temporal features
        """
        features = {}

        if len(frames) < 2:
            return {'face_temporal_consistency': 0.5}

        try:
            # Auto-detect faces if bboxes not provided
            if face_bboxes is None:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                face_bboxes = []

                for frame in frames:
                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = frame

                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    if len(faces) > 0:
                        face_bboxes.append(faces[0])  # Use first detected face
                    else:
                        face_bboxes.append(None)

            # Extract face regions
            face_regions = []
            valid_indices = []

            for i, (frame, bbox) in enumerate(zip(frames, face_bboxes)):
                if bbox is not None:
                    x, y, w, h = bbox
                    if len(frame.shape) == 3:
                        face_region = frame[y:y+h, x:x+w]
                    else:
                        face_region = frame[y:y+h, x:x+w]

                    if face_region.size > 0:
                        # Resize to standard size for comparison
                        face_region = cv2.resize(face_region, (64, 64))
                        face_regions.append(face_region)
                        valid_indices.append(i)

            if len(face_regions) < 2:
                return {'face_temporal_consistency': 0.0}

            # Face consistency analysis
            face_similarities = []

            for i in range(len(face_regions) - 1):
                face1 = face_regions[i]
                face2 = face_regions[i + 1]

                # Convert to grayscale for comparison
                if len(face1.shape) == 3:
                    face1_gray = cv2.cvtColor(face1, cv2.COLOR_RGB2GRAY)
                else:
                    face1_gray = face1

                if len(face2.shape) == 3:
                    face2_gray = cv2.cvtColor(face2, cv2.COLOR_RGB2GRAY)
                else:
                    face2_gray = face2

                # Compute similarity
                similarity = np.corrcoef(face1_gray.flatten(), face2_gray.flatten())[0, 1]
                if not np.isnan(similarity):
                    face_similarities.append(similarity)

            if face_similarities:
                features['face_temporal_consistency'] = float(np.mean(face_similarities))
                features['face_temporal_variance'] = float(np.std(face_similarities))

            # Face position stability
            if len(valid_indices) > 1:
                face_centers = []
                face_sizes = []

                for i, bbox in enumerate([face_bboxes[j] for j in valid_indices]):
                    if bbox is not None:
                        x, y, w, h = bbox
                        center_x = x + w // 2
                        center_y = y + h // 2
                        face_centers.append((center_x, center_y))
                        face_sizes.append(w * h)

                if len(face_centers) > 1:
                    # Position stability
                    center_movements = []
                    for i in range(len(face_centers) - 1):
                        dx = face_centers[i+1][0] - face_centers[i][0]
                        dy = face_centers[i+1][1] - face_centers[i][1]
                        movement = np.sqrt(dx**2 + dy**2)
                        center_movements.append(movement)

                    features['face_position_stability'] = float(1.0 / (1.0 + np.mean(center_movements) / 10.0))

                    # Size stability
                    size_changes = np.diff(face_sizes)
                    features['face_size_stability'] = float(1.0 / (1.0 + np.std(size_changes) / np.mean(face_sizes)))

            # Facial landmark consistency (simplified)
            # This would require more advanced face analysis
            # For now, use edge-based consistency
            face_edge_consistency = []

            for face_region in face_regions:
                if len(face_region.shape) == 3:
                    gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
                else:
                    gray = face_region

                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                face_edge_consistency.append(edge_density)

            if len(face_edge_consistency) > 1:
                edge_consistency_std = np.std(face_edge_consistency)
                edge_consistency_mean = np.mean(face_edge_consistency)
                features['face_edge_temporal_consistency'] = float(1.0 / (1.0 + edge_consistency_std / (edge_consistency_mean + 1e-8)))

        except Exception as e:
            logger.warning(f"Face temporal feature extraction failed: {e}")
            features = {
                'face_temporal_consistency': 0.5, 'face_temporal_variance': 0.1,
                'face_position_stability': 0.8, 'face_size_stability': 0.8,
                'face_edge_temporal_consistency': 0.7
            }

        return features

    def extract_frequency_domain_features(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Extract temporal frequency domain features

        Args:
            frames: List of video frames

        Returns:
            Dictionary of frequency domain features
        """
        features = {}

        if len(frames) < 8:  # Need minimum frames for frequency analysis
            return {'temporal_freq_energy_ratio': 0.5}

        try:
            # Convert frames to temporal signals
            # Sample points from frames
            h, w = frames[0].shape[:2]
            sample_points = [
                (h//4, w//4), (h//4, 3*w//4),
                (3*h//4, w//4), (3*h//4, 3*w//4),
                (h//2, w//2)
            ]

            temporal_signals = []

            for point in sample_points:
                y, x = point
                signal_values = []

                for frame in frames:
                    if len(frame.shape) == 3:
                        # Use luminance
                        value = 0.299 * frame[y, x, 0] + 0.587 * frame[y, x, 1] + 0.114 * frame[y, x, 2]
                    else:
                        value = frame[y, x]
                    signal_values.append(value)

                temporal_signals.append(signal_values)

            # Analyze frequency content
            freq_features = []

            for signal_values in temporal_signals:
                if len(signal_values) > 4:
                    # FFT analysis
                    fft = np.fft.fft(signal_values)
                    freqs = np.fft.fftfreq(len(signal_values))
                    magnitude = np.abs(fft)

                    # Energy in different frequency bands
                    total_energy = np.sum(magnitude**2)

                    if total_energy > 0:
                        band_energies = []

                        for low_freq, high_freq in self.temporal_freq_bands:
                            mask = (freqs >= low_freq) & (freqs <= high_freq)
                            band_energy = np.sum(magnitude[mask]**2)
                            band_energies.append(band_energy / total_energy)

                        freq_features.append(band_energies)

            if freq_features:
                # Average across sample points
                avg_band_energies = np.mean(freq_features, axis=0)

                features.update({
                    'temporal_low_freq_energy': float(avg_band_energies[0]),
                    'temporal_mid_freq_energy': float(avg_band_energies[1]),
                    'temporal_high_freq_energy': float(avg_band_energies[2]),
                    'temporal_very_high_freq_energy': float(avg_band_energies[3])
                })

                # Frequency distribution metrics
                features['temporal_freq_entropy'] = float(-np.sum(avg_band_energies * np.log2(avg_band_energies + 1e-10)))
                features['temporal_freq_centroid'] = float(np.sum(avg_band_energies * np.arange(len(avg_band_energies))) / np.sum(avg_band_energies))

        except Exception as e:
            logger.warning(f"Frequency domain feature extraction failed: {e}")
            features = {
                'temporal_low_freq_energy': 0.4, 'temporal_mid_freq_energy': 0.3,
                'temporal_high_freq_energy': 0.2, 'temporal_very_high_freq_energy': 0.1,
                'temporal_freq_entropy': 1.8, 'temporal_freq_centroid': 1.5
            }

        return features

    def extract_all_temporal_features(self, frames: List[np.ndarray], face_bboxes: List[Optional[List[int]]] = None) -> Dict[str, float]:
        """
        Extract comprehensive temporal features

        Args:
            frames: List of video frames
            face_bboxes: Optional list of face bounding boxes

        Returns:
            Dictionary of all temporal features
        """
        all_features = {}

        try:
            # Extract different feature categories
            motion_features = self.extract_motion_features(frames)
            consistency_features = self.extract_temporal_consistency_features(frames)
            face_temporal_features = self.extract_face_temporal_features(frames, face_bboxes)
            frequency_features = self.extract_frequency_domain_features(frames)

            # Combine all features
            all_features.update(motion_features)
            all_features.update(consistency_features)
            all_features.update(face_temporal_features)
            all_features.update(frequency_features)

            # Add summary statistics
            feature_values = [v for v in all_features.values() if isinstance(v, (int, float))]
            if feature_values:
                all_features.update({
                    'temporal_feature_mean': float(np.mean(feature_values)),
                    'temporal_feature_std': float(np.std(feature_values)),
                    'temporal_feature_count': len(feature_values)
                })

        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {e}")
            # Return minimal features
            all_features = {
                'motion_score': 0.5, 'consistency_score': 0.5,
                'face_temporal_score': 0.5, 'frequency_score': 0.5,
                'temporal_feature_count': 4
            }

        return all_features

    def compute_temporal_quality_score(self, features: Dict[str, float]) -> float:
        """
        Compute overall temporal quality score from features

        Args:
            features: Extracted temporal features

        Returns:
            Quality score (0-1, higher = better quality)
        """
        try:
            quality_indicators = []

            # Motion smoothness
            if 'motion_smoothness' in features:
                quality_indicators.append(features['motion_smoothness'])

            # Temporal consistency
            if 'temporal_smoothness' in features:
                quality_indicators.append(features['temporal_smoothness'])

            # Face consistency
            if 'face_temporal_consistency' in features:
                quality_indicators.append(features['face_temporal_consistency'])

            # Low flickering (inverse)
            if 'flickering_score' in features:
                flicker_quality = max(0, 1.0 - features['flickering_score'])
                quality_indicators.append(flicker_quality)

            if quality_indicators:
                return float(np.mean(quality_indicators))
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"Temporal quality score computation failed: {e}")
            return 0.5
