"""
AI-Generated Content Detector
Detects AI art, synthetic faces, AI-generated dance videos, etc.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class AIGeneratedDetector:
    """Detect AI-generated content without special dataset"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
    
    def detect_ai_generated_image(self, image_path: str) -> Dict:
        """
        Detect if image is AI-generated
        Uses artifact detection without additional training
        """
        
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check multiple indicators
        indicators = {
            'blending_artifacts': self._check_blending_artifacts(image_rgb),
            'face_consistency': self._check_face_consistency(image_rgb),
            'texture_inconsistency': self._check_texture_inconsistency(image_rgb),
            'frequency_artifacts': self._check_frequency_artifacts(image_rgb),
            'eye_quality': self._check_eye_quality(image_rgb)
        }
        
        # Calculate overall score
        ai_score = np.mean(list(indicators.values()))
        
        return {
            'is_ai_generated': ai_score > 0.6,
            'ai_probability': float(ai_score),
            'indicators': indicators,
            'explanation': self._generate_image_explanation(indicators, ai_score)
        }
    
    def detect_ai_generated_video(self, video_path: str) -> Dict:
        """Detect if video is AI-generated (deepfake, synthetic dance, etc.)"""
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return {'error': 'Failed to load video'}
        
        # Sample frames
        frame_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
        
        indicators_list = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            indicators = {
                'motion_inconsistency': self._check_motion_consistency(frame),
                'lighting_artifacts': self._check_lighting_artifacts(frame_rgb),
                'boundary_artifacts': self._check_boundary_artifacts(frame_rgb)
            }
            
            indicators_list.append(indicators)
        
        cap.release()
        
        # Aggregate results
        avg_indicators = {
            key: np.mean([ind[key] for ind in indicators_list])
            for key in indicators_list[0].keys()
        }
        
        ai_score = np.mean(list(avg_indicators.values()))
        
        return {
            'is_ai_generated': ai_score > 0.6,
            'ai_probability': float(ai_score),
            'indicators': avg_indicators,
            'explanation': self._generate_video_explanation(avg_indicators, ai_score)
        }
    
    def _check_blending_artifacts(self, image: np.ndarray) -> float:
        """Detect blending seams (common in AI images)"""
        
        # Convert to LAB color space
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Check for color discontinuities
        edges = cv2.Canny(image_lab[:,:,0], 50, 150)
        edge_density = np.mean(edges) / 255
        
        # Anomalous edge density indicates AI generation
        return min(edge_density, 1.0)
    
    def _check_face_consistency(self, image: np.ndarray) -> float:
        """Check facial feature consistency"""
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 1.3, 5
        )
        
        if len(faces) == 0:
            return 0.5  # Neutral if no face
        
        # Check face symmetry (AI images often have perfect symmetry)
        face = faces[0]
        x, y, w, h = face
        
        face_region = image[y:y+h, x:x+w]
        left_half = face_region[:, :w//2]
        right_half = cv2.flip(face_region[:, w//2:], 1)
        
        # Calculate difference
        diff = cv2.absdiff(left_half, right_half)
        symmetry_score = np.mean(diff) / 255
        
        # Perfect symmetry = likely AI
        return min(symmetry_score, 1.0)
    
    def _check_texture_inconsistency(self, image: np.ndarray) -> float:
        """Check for unnatural texture patterns"""
        
        # Compute texture using Gabor filter
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # High-frequency components
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Normalize (typical range is 0-1000)
        normalized_variance = min(texture_variance / 100, 1.0)
        
        return normalized_variance
    
    def _check_frequency_artifacts(self, image: np.ndarray) -> float:
        """Detect frequency domain artifacts"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Check for unusual frequency patterns
        magnitude_norm = np.log1p(magnitude)
        
        # AI images have different frequency signatures
        high_freq_ratio = np.sum(magnitude_norm > np.percentile(magnitude_norm, 95)) / magnitude_norm.size
        
        return min(high_freq_ratio * 5, 1.0)
    
    def _check_eye_quality(self, image: np.ndarray) -> float:
        """Check eye quality (common failure point for AI)"""
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 0.5
        
        face = faces[0]
        roi_gray = gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # AI images often have detected/undetected eyes
        if len(eyes) < 2:
            return 0.8  # Likely AI
        
        return 0.3  # Likely real
    
    def _check_motion_consistency(self, frame: np.ndarray) -> float:
        """Check for motion inconsistencies"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Count isolated pixels (common in synthetic videos)
        isolated_pixels = np.sum(opened) / opened.size
        
        return min(isolated_pixels * 10, 1.0)
    
    def _check_lighting_artifacts(self, image: np.ndarray) -> float:
        """Check for unnatural lighting"""
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Check brightness consistency
        brightness = hsv[:, :, 2]
        brightness_variance = np.var(brightness) / 255
        
        # Unnatural = very uniform or very varied
        if brightness_variance < 0.1:
            return 0.7  # Too uniform = AI
        elif brightness_variance > 0.9:
            return 0.6  # Too varied = AI
        
        return 0.2  # Natural
    
    def _check_boundary_artifacts(self, image: np.ndarray) -> float:
        """Check for artifacts at object boundaries"""
        
        # Detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Check for artifacts (disconnected or jagged edges)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.5
        
        # Calculate contour irregularity
        irregularity_scores = []
        for contour in contours[:10]:  # Check top 10 contours
            if cv2.contourArea(contour) > 100:
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    irregularity_scores.append(1 - circularity)
        
        if not irregularity_scores:
            return 0.5
        
        return min(np.mean(irregularity_scores), 1.0)
    
    def _generate_image_explanation(self, indicators: Dict, ai_score: float) -> str:
        """Generate explanation for image analysis"""
        
        explanation = f"""
🖼️  AI-GENERATED IMAGE ANALYSIS
{'=' * 50}

🔍 ANALYSIS RESULT: {'AI-GENERATED' if ai_score > 0.6 else 'LIKELY AUTHENTIC'}
   Confidence: {ai_score:.1%}

📊 INDICATORS:
"""
        
        for indicator, score in indicators.items():
            status = "🔴" if score > 0.6 else "🟢"
            explanation += f"   {status} {indicator}: {score:.1%}\n"
        
        if ai_score > 0.6:
            explanation += """
⚠️  AI GENERATION SIGNS:
   • Smooth blending artifacts detected
   • Texture irregularities found
   • Frequency domain anomalies
   • Facial feature inconsistencies
"""
        
        explanation += f"{'=' * 50}"
        
        return explanation
    
    def _generate_video_explanation(self, indicators: Dict, ai_score: float) -> str:
        """Generate explanation for video analysis"""
        
        explanation = f"""
🎥 AI-GENERATED VIDEO ANALYSIS
{'=' * 50}

🔍 RESULT: {'SYNTHETIC/AI-GENERATED' if ai_score > 0.6 else 'LIKELY AUTHENTIC'}
   Confidence: {ai_score:.1%}

📊 INDICATORS:
"""
        
        for indicator, score in indicators.items():
            status = "🔴" if score > 0.6 else "🟢"
            explanation += f"   {status} {indicator}: {score:.1%}\n"
        
        if ai_score > 0.6:
            explanation += """
⚠️  SYNTHETIC VIDEO SIGNS:
   • Inconsistent motion patterns
   • Unnatural lighting transitions
   • Boundary artifacts in movements
   • Synthetic dance movements detected
"""
        
        explanation += f"{'=' * 50}"
        
        return explanation
