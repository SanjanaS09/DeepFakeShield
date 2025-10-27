"""
Video Deepfake Detector
Specialized model for detecting manipulated videos using temporal analysis
Supports I3D, SlowFast, X3D backbones with temporal consistency checking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Dict, Tuple, Union, List, Optional
import logging
from pathlib import Path
import tempfile
import subprocess

from .base_model import BaseDetectionModel

logger = logging.getLogger(__name__)

class VideoDetector(BaseDetectionModel):
    """
    Video-based deepfake detection model with temporal analysis
    Detects inconsistencies across frames and temporal artifacts
    """

    def __init__(self,
                 backbone: str = 'i3d',
                 num_classes: int = 2,
                 device: str = 'cpu',
                 pretrained: bool = True,
                 frames_per_clip: int = 16,
                 frame_sampling_rate: int = 2,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize Video Detector

        Args:
            backbone: Backbone architecture ('i3d', 'slowfast', 'x3d')
            num_classes: Number of classes
            device: Device to run on
            pretrained: Use pretrained weights
            frames_per_clip: Number of frames per video clip
            frame_sampling_rate: Frame sampling rate
            input_size: Input frame size
        """
        super().__init__(f'video_detector_{backbone}', num_classes, device, pretrained)

        self.backbone_name = backbone
        self.frames_per_clip = frames_per_clip
        self.frame_sampling_rate = frame_sampling_rate
        self.input_size = input_size
        self.input_shape = (3, frames_per_clip, input_size[0], input_size[1])

        # Build model
        self.build_model()

        # Move to device
        self.to(self.device)

        # Initialize transforms
        self._init_transforms()

        logger.info(f"Initialized VideoDetector with {backbone} backbone")

    def build_model(self) -> None:
        """Build the video detection model architecture"""

        if self.backbone_name == 'i3d':
            # Simplified I3D-like architecture
            self.backbone = self._build_i3d_backbone()
            self.feature_dim = 1024

        elif self.backbone_name == 'slowfast':
            # Simplified SlowFast-like architecture
            self.backbone = self._build_slowfast_backbone()
            self.feature_dim = 1536  # 1024 + 512 for slow and fast pathways

        elif self.backbone_name == 'x3d':
            # Simplified X3D-like architecture
            self.backbone = self._build_x3d_backbone()
            self.feature_dim = 832

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Temporal consistency module
        self.temporal_analyzer = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Frame-wise feature extractor for temporal analysis
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 256)
        )

        # Attention mechanism for temporal fusion
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim + 256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def _build_i3d_backbone(self) -> nn.Module:
        """Build I3D-like backbone"""
        return nn.Sequential(
            # 3D convolutions for spatiotemporal feature extraction
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

    def _build_slowfast_backbone(self) -> nn.Module:
        """Build SlowFast-like backbone with slow and fast pathways"""
        # Slow pathway (high spatial, low temporal resolution)
        slow_pathway = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

        # Fast pathway (low spatial, high temporal resolution)
        fast_pathway = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(5, 7, 7), stride=(1, 4, 4), padding=(2, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

        return nn.ModuleDict({
            'slow_pathway': slow_pathway,
            'fast_pathway': fast_pathway
        })

    def _build_x3d_backbone(self) -> nn.Module:
        """Build X3D-like backbone"""
        return nn.Sequential(
            # Efficient 3D convolutions
            nn.Conv3d(3, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),

            # Bottleneck blocks
            self._make_x3d_block(24, 54, stride=(1, 2, 2)),
            self._make_x3d_block(54, 108, stride=(2, 2, 2)),
            self._make_x3d_block(108, 216, stride=(2, 2, 2)),
            self._make_x3d_block(216, 432, stride=(2, 2, 2)),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

    def _make_x3d_block(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int]) -> nn.Module:
        """Create X3D bottleneck block"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels // 4, out_channels // 4, 
                     kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_transforms(self) -> None:
        """Initialize video preprocessing transforms"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def preprocess_input(self, input_data) -> torch.Tensor:
        """
        Preprocess input video data

        Args:
            input_data: Input video (path string or numpy array)

        Returns:
            Preprocessed tensor [batch_size, channels, frames, height, width]
        """
        if isinstance(input_data, str):
            # Load video from path
            frames = self._load_video_frames(input_data)
        elif isinstance(input_data, np.ndarray):
            # Assume input is already frame sequence
            frames = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Sample frames
        sampled_frames = self._sample_frames(frames)

        # Convert frames to tensor
        frame_tensors = []
        for frame in sampled_frames:
            if isinstance(frame, np.ndarray):
                # Convert numpy to PIL Image
                from PIL import Image
                frame_pil = Image.fromarray(frame.astype(np.uint8))
            else:
                frame_pil = frame

            frame_tensor = self.transform(frame_pil)
            frame_tensors.append(frame_tensor)

        # Stack frames: [frames, channels, height, width]
        video_tensor = torch.stack(frame_tensors, dim=0)

        # Reorder to [channels, frames, height, width] and add batch dimension
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        return video_tensor

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            raise ValueError(f"Could not load frames from {video_path}")

        return frames

    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample frames from video sequence"""
        num_frames = len(frames)

        if num_frames < self.frames_per_clip:
            # Repeat frames if video is too short
            indices = np.linspace(0, num_frames - 1, self.frames_per_clip, dtype=int)
        else:
            # Sample frames uniformly
            start_idx = max(0, (num_frames - self.frames_per_clip * self.frame_sampling_rate) // 2)
            indices = np.arange(start_idx, 
                              start_idx + self.frames_per_clip * self.frame_sampling_rate, 
                              self.frame_sampling_rate)

            # Ensure we don't exceed frame count
            indices = np.clip(indices, 0, num_frames - 1)

        return [frames[i] for i in indices]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatiotemporal features from video input

        Args:
            x: Input tensor [batch_size, channels, frames, height, width]

        Returns:
            Extracted features
        """
        batch_size, channels, frames, height, width = x.shape

        # Extract spatiotemporal features
        if self.backbone_name == 'slowfast':
            # Process slow and fast pathways separately
            slow_features = self.backbone['slow_pathway'](x)

            # Downsample for fast pathway (temporal dimension)
            x_fast = x[:, :, ::4, :, :]  # Sample every 4th frame
            fast_features = self.backbone['fast_pathway'](x_fast)

            # Concatenate features
            spatiotemporal_features = torch.cat([slow_features, fast_features], dim=1)
        else:
            spatiotemporal_features = self.backbone(x)

        # Extract frame-wise features for temporal analysis
        x_frames = x.permute(0, 2, 1, 3, 4)  # [batch, frames, channels, height, width]
        x_frames = x_frames.reshape(batch_size * frames, channels, height, width)

        frame_features = self.frame_encoder(x_frames)  # [batch*frames, 256]
        frame_features = frame_features.view(batch_size, frames, -1)  # [batch, frames, 256]

        # Temporal consistency analysis
        temporal_features, _ = self.temporal_analyzer(frame_features)
        temporal_features = temporal_features[:, -1, :]  # Use last hidden state

        # Combine features
        combined_features = torch.cat([spatiotemporal_features, temporal_features], dim=1)

        # Apply attention
        combined_features = combined_features.unsqueeze(1)
        attended_features, _ = self.temporal_attention(
            combined_features, combined_features, combined_features
        )
        attended_features = attended_features.squeeze(1)

        return attended_features

    def get_feature_breakdown(self, input_data) -> Dict[str, float]:
        """
        Get detailed feature breakdown for video analysis

        Args:
            input_data: Input video

        Returns:
            Dictionary with feature scores
        """
        self.eval()

        with torch.no_grad():
            # Load video frames for analysis
            if isinstance(input_data, str):
                frames = self._load_video_frames(input_data)
            else:
                frames = input_data

            # Calculate various metrics
            scores = {}

            # Temporal consistency analysis
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                frame_diffs.append(diff)

            if frame_diffs:
                # High variance in frame differences indicates inconsistency
                temporal_variance = np.var(frame_diffs)
                scores['temporal_inconsistency'] = float(min(temporal_variance / 1000.0, 1.0))
            else:
                scores['temporal_inconsistency'] = 0.0

            # Blur analysis across frames
            blur_scores = []
            for frame in frames[::5]:  # Sample every 5th frame
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(blur_score)

            if blur_scores:
                scores['blur_score'] = float(1.0 - min(np.mean(blur_scores) / 1000.0, 1.0))
            else:
                scores['blur_score'] = 0.0

            # Facial warping detection using optical flow
            if len(frames) > 1:
                warping_scores = []
                for i in range(1, min(len(frames), 10)):
                    prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                    curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, None, None
                    )

                    if flow is not None and len(flow) > 0:
                        # Analyze flow magnitude
                        flow_magnitude = np.mean(np.sqrt(flow[0]**2 + flow[1]**2)) if len(flow) > 1 else 0
                        warping_scores.append(flow_magnitude)

                if warping_scores:
                    scores['facial_warping'] = float(min(np.mean(warping_scores) / 10.0, 1.0))
                else:
                    scores['facial_warping'] = 0.0
            else:
                scores['facial_warping'] = 0.0

            # Noise analysis
            noise_scores = []
            for frame in frames[::10]:  # Sample every 10th frame
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                noise_level = np.std(gray)
                noise_scores.append(noise_level)

            if noise_scores:
                scores['noise_score'] = float(min(np.mean(noise_scores) / 255.0, 1.0))
            else:
                scores['noise_score'] = 0.0

            # Compression artifacts (simplified)
            scores['compression_artifacts'] = scores['blur_score'] * 0.5 + scores['noise_score'] * 0.3

            return scores

    def analyze_temporal_consistency(self, input_data) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze temporal consistency across video frames

        Args:
            input_data: Input video

        Returns:
            Dictionary with temporal analysis results
        """
        if isinstance(input_data, str):
            frames = self._load_video_frames(input_data)
        else:
            frames = input_data

        if len(frames) < 2:
            return {
                'consistency_score': 1.0,
                'frame_differences': [],
                'motion_consistency': 1.0,
                'lighting_consistency': 1.0
            }

        # Frame-to-frame differences
        frame_diffs = []
        motion_vectors = []
        lighting_changes = []

        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

            # Frame difference
            diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
            frame_diffs.append(diff)

            # Motion analysis using optical flow
            try:
                flow = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, None, None)
                if flow is not None and len(flow) > 0:
                    motion_mag = np.mean(np.sqrt(flow[0]**2 + flow[1]**2)) if len(flow) > 1 else 0
                    motion_vectors.append(motion_mag)
            except:
                motion_vectors.append(0.0)

            # Lighting consistency
            prev_brightness = np.mean(prev_frame)
            curr_brightness = np.mean(curr_frame)
            lighting_change = abs(curr_brightness - prev_brightness)
            lighting_changes.append(lighting_change)

        # Calculate consistency scores
        consistency_score = 1.0 - min(np.var(frame_diffs) / 10000.0, 1.0)
        motion_consistency = 1.0 - min(np.var(motion_vectors) / 100.0, 1.0) if motion_vectors else 1.0
        lighting_consistency = 1.0 - min(np.var(lighting_changes) / 1000.0, 1.0)

        return {
            'consistency_score': float(consistency_score),
            'frame_differences': frame_diffs,
            'motion_consistency': float(motion_consistency),
            'lighting_consistency': float(lighting_consistency),
            'average_motion': float(np.mean(motion_vectors)) if motion_vectors else 0.0,
            'num_frames_analyzed': len(frames)
        }
