"""
Lightweight Video Preprocessor for Real-Time Deepfake Detection
Optimized for laptops - extracts key frames, detects faces, prepares tensor input
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Lightweight video preprocessing for deepfake detection"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 frames_per_clip: int = 8,
                 device: str = 'cpu'):
        """Initialize preprocessor"""
        self.target_size = target_size
        self.frames_per_clip = frames_per_clip
        self.device = device
        
        # Initialize face detector (Haar Cascade - lightweight)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Normalization transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"VideoPreprocessor initialized: {frames_per_clip} frames @ {target_size}")
    
    def extract_frames(self, 
                      video_path: str,
                      max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Extract frames from video using uniform sampling"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames is None:
            max_frames = self.frames_per_clip * 3
        
        sample_interval = max(1, total_frames // max_frames)
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_idx += 1
        
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from {total_frames} total")
        return frames
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(50, 50)
        )
        return faces  # Returns array of shape (n_faces, 4)
    
    def crop_face(self, frame: np.ndarray, 
                  bbox: Tuple[int, int, int, int],
                  padding: float = 0.2) -> np.ndarray:
        """Crop face region with padding"""
        x, y, w, h = bbox
        
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        return frame[y1:y2, x1:x2]
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess single frame to tensor"""
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        from PIL import Image
        frame_pil = Image.fromarray(frame_resized)
        frame_tensor = self.normalize(frame_pil)
        
        return frame_tensor
    
    def sample_frames_uniform(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Uniformly sample frames to exact count"""
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        if len(frames) <= self.frames_per_clip:
            sampled = frames.copy()
            while len(sampled) < self.frames_per_clip:
                sampled.append(frames[-1])
            return sampled
        
        indices = np.linspace(0, len(frames)-1, self.frames_per_clip, dtype=int)
        return [frames[i] for i in indices]
    
    def preprocess_video(self,
                        video_path: str,
                        detect_faces: bool = True,
                        emit_callback = None) -> Dict[str, any]:
        """Complete preprocessing pipeline for video"""
        try:
            if emit_callback:
                emit_callback("Video Loading", f"Reading: {Path(video_path).name}")
            
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if emit_callback:
                emit_callback("Frame Extraction", f"Extracted {len(frames)} frames")
            
            # Face detection and cropping
            if detect_faces:
                processed_frames = []
                
                for i, frame in enumerate(frames):
                    faces = self.detect_faces(frame)
                    
                    # ✅ FIX: Check length of array instead of truthiness
                    if len(faces) > 0:
                        # Crop largest face
                        largest_face = max(faces, key=lambda b: b[2]*b[3])
                        cropped = self.crop_face(frame, largest_face)
                    else:
                        # Center crop if no face detected
                        h, w = frame.shape[:2]
                        size = min(h, w)
                        y1 = (h - size) // 2
                        x1 = (w - size) // 2
                        cropped = frame[y1:y1+size, x1:x1+size]
                    
                    processed_frames.append(cropped)
                
                frames = processed_frames
                
                if emit_callback:
                    emit_callback("Face Detection", f"Detected faces in frames")
            
            # Sample uniform frames
            sampled_frames = self.sample_frames_uniform(frames)
            
            if emit_callback:
                emit_callback("Frame Sampling", f"Sampled {len(sampled_frames)} frames")
            
            # Convert to tensors and stack
            frame_tensors = []
            for frame in sampled_frames:
                tensor = self.preprocess_frame(frame)
                frame_tensors.append(tensor)
            
            video_tensor = torch.stack(frame_tensors)
            
            if emit_callback:
                emit_callback("Preprocessing Complete", "Ready for inference")
            
            return {
                'video_tensor': video_tensor,
                'frames_extracted': len(frames),
                'frames_sampled': len(sampled_frames),
                'shape': tuple(video_tensor.shape),
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
