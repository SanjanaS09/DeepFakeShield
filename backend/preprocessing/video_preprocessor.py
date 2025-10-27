"""
Video Preprocessor for Multi-Modal Deepfake Detection
Handles video loading, frame extraction, temporal analysis, and preprocessing
Compatible with dataset structure: dataset/video/{train,validation,test}/{FAKE,REAL}/
"""

import os
import cv2
import numpy as np
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """
    Comprehensive video preprocessor for deepfake detection
    Supports frame extraction, temporal sampling, face detection, and platform-specific processing
    """

    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 frames_per_clip: int = 16,
                 sampling_rate: int = 2,
                 target_fps: int = 25,
                 normalize: bool = True,
                 device: str = 'cpu'):
        """
        Initialize Video Preprocessor

        Args:
            target_size: Target frame size (height, width)
            frames_per_clip: Number of frames per video clip
            sampling_rate: Frame sampling rate (1 = every frame, 2 = every other frame)
            target_fps: Target FPS for processing
            normalize: Whether to normalize pixel values
            device: Device for preprocessing
        """
        self.target_size = target_size
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.target_fps = target_fps
        self.normalize = normalize
        self.device = device

        # Initialize face detection
        self._init_face_detector()

        # Initialize transforms
        self._init_transforms()

        # Platform-specific configurations
        self.platform_configs = {
            'whatsapp': {
                'max_resolution': (480, 360),
                'max_bitrate': '500k',
                'compression_crf': 28,
                'fps': 15
            },
            'youtube': {
                'max_resolution': (1920, 1080),
                'max_bitrate': '5000k',
                'compression_crf': 23,
                'fps': 30
            },
            'zoom': {
                'max_resolution': (640, 480),
                'max_bitrate': '800k',
                'compression_crf': 30,
                'fps': 15
            },
            'generic': {
                'max_resolution': (1920, 1080),
                'max_bitrate': '8000k',
                'compression_crf': 18,
                'fps': 30
            }
        }

        logger.info(f"Initialized VideoPreprocessor with {frames_per_clip} frames per clip")

    def _init_face_detector(self):
        """Initialize face detection for video frames"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            # Try MediaPipe if available
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detector_mp = self.mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                logger.info("MediaPipe face detection available for video")
            except ImportError:
                self.face_detector_mp = None

        except Exception as e:
            logger.warning(f"Face detection initialization failed: {e}")
            self.face_cascade = None
            self.face_detector_mp = None

    def _init_transforms(self):
        """Initialize video frame transformation pipelines"""
        # Standard normalization
        self.normalize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        # Basic transforms without normalization
        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

        # Augmentation for training
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

    def load_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get video metadata using OpenCV and ffprobe

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get basic info with OpenCV
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        # Try to get more detailed info with ffprobe
        detailed_info = {}
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)

                # Extract video stream info
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        detailed_info = {
                            'codec': stream.get('codec_name'),
                            'bitrate': int(stream.get('bit_rate', 0)),
                            'pixel_format': stream.get('pix_fmt'),
                            'color_space': stream.get('color_space'),
                            'level': stream.get('level')
                        }
                        break
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            logger.warning("ffprobe not available or failed, using basic info only")

        video_info = {
            'path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'resolution': f"{width}x{height}",
            'file_size_mb': video_path.stat().st_size / (1024 * 1024),
            **detailed_info
        }

        return video_info

    def extract_frames(self, 
                      video_path: Union[str, Path],
                      start_frame: int = 0,
                      max_frames: Optional[int] = None,
                      sampling_rate: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            max_frames: Maximum number of frames to extract
            sampling_rate: Frame sampling rate (overrides instance default)

        Returns:
            List of frame arrays in RGB format
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        sampling_rate = sampling_rate or self.sampling_rate
        frame_idx = 0
        extracted_count = 0

        try:
            # Seek to start frame if needed
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Sample frames based on sampling rate
                if (frame_idx - start_frame) % sampling_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1

                    # Check if we've reached max frames
                    if max_frames and extracted_count >= max_frames:
                        break

                frame_idx += 1

        finally:
            cap.release()

        logger.debug(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def detect_faces_in_frames(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect faces in video frames

        Args:
            frames: List of frame arrays

        Returns:
            List of face detection results for each frame
        """
        all_faces = []

        for frame in frames:
            frame_faces = []

            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                )

                for (x, y, w, h) in detected_faces:
                    frame_faces.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'center': [int(x + w//2), int(y + h//2)],
                        'area': int(w * h),
                        'confidence': 1.0
                    })

            all_faces.append(frame_faces)

        return all_faces

    def crop_face_regions(self, frames: List[np.ndarray], face_detections: List[List[Dict]]) -> List[np.ndarray]:
        """
        Crop face regions from frames based on detections

        Args:
            frames: List of frame arrays
            face_detections: Face detection results for each frame

        Returns:
            List of cropped frames focused on faces
        """
        cropped_frames = []

        for frame, faces in zip(frames, face_detections):
            if faces:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x['area'])
                x, y, w, h = largest_face['bbox']

                # Add padding
                padding = 0.2
                pad_w = int(w * padding)
                pad_h = int(h * padding)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame.shape[1], x + w + pad_w)
                y2 = min(frame.shape[0], y + h + pad_h)

                cropped_frame = frame[y1:y2, x1:x2]
            else:
                # Use center crop if no face detected
                h, w = frame.shape[:2]
                crop_size = min(h, w)
                y1 = (h - crop_size) // 2
                x1 = (w - crop_size) // 2
                cropped_frame = frame[y1:y1+crop_size, x1:x1+crop_size]

            cropped_frames.append(cropped_frame)

        return cropped_frames

    def sample_frames_temporal(self, frames: List[np.ndarray], method: str = 'uniform') -> List[np.ndarray]:
        """
        Sample frames for temporal analysis

        Args:
            frames: List of frame arrays
            method: Sampling method ('uniform', 'random', 'keyframes')

        Returns:
            Sampled frames
        """
        if len(frames) <= self.frames_per_clip:
            # Pad with repeated frames if not enough frames
            sampled = frames.copy()
            while len(sampled) < self.frames_per_clip:
                sampled.extend(frames[:self.frames_per_clip - len(sampled)])
            return sampled[:self.frames_per_clip]

        if method == 'uniform':
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, self.frames_per_clip, dtype=int)
            return [frames[i] for i in indices]

        elif method == 'random':
            # Random sampling
            indices = sorted(np.random.choice(len(frames), self.frames_per_clip, replace=False))
            return [frames[i] for i in indices]

        elif method == 'keyframes':
            # Simple keyframe detection based on frame differences
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                frame_diffs.append((i, diff))

            # Sort by difference and take top frames
            keyframes = sorted(frame_diffs, key=lambda x: x[1], reverse=True)
            selected_indices = sorted([0] + [kf[0] for kf in keyframes[:self.frames_per_clip-1]])

            return [frames[i] for i in selected_indices[:self.frames_per_clip]]

        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def apply_platform_compression(self, video_path: Union[str, Path], platform: str = 'generic') -> Path:
        """
        Apply platform-specific compression to video

        Args:
            video_path: Input video path
            platform: Platform type

        Returns:
            Path to compressed video
        """
        config = self.platform_configs.get(platform, self.platform_configs['generic'])

        # Create temporary file for compressed video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = Path(tmp_file.name)

        try:
            # Build ffmpeg command
            max_width, max_height = config['max_resolution']

            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', f'scale=\'min({max_width},iw)\':\'min({max_height},ih)\':force_original_aspect_ratio=decrease',
                '-r', str(config['fps']),
                '-c:v', 'libx264',
                '-crf', str(config['compression_crf']),
                '-maxrate', config['max_bitrate'],
                '-bufsize', str(int(config['max_bitrate'].replace('k', '')) * 2) + 'k',
                '-y',  # Overwrite output
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.warning(f"FFmpeg compression failed: {result.stderr}")
                return Path(video_path)  # Return original if compression fails

            return output_path

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Video compression failed: {e}")
            return Path(video_path)

    def preprocess_frames_batch(self, frames: List[np.ndarray], augment: bool = False) -> torch.Tensor:
        """
        Preprocess a batch of frames into tensor format

        Args:
            frames: List of frame arrays
            augment: Whether to apply augmentation

        Returns:
            Tensor of shape [frames, channels, height, width]
        """
        processed_frames = []

        transform = self.augment_transform if augment else (
            self.normalize_transform if self.normalize else self.basic_transform
        )

        for frame in frames:
            # Convert numpy array to tensor
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            processed_frame = transform(frame)
            processed_frames.append(processed_frame)

        # Stack into tensor [T, C, H, W]
        return torch.stack(processed_frames)

    def preprocess_single(self,
                         video_path: Union[str, Path],
                         platform_config: Optional[Dict] = None,
                         detect_faces: bool = True,
                         sampling_method: str = 'uniform',
                         augment: bool = False) -> Dict[str, Any]:
        """
        Preprocess a single video file

        Args:
            video_path: Path to video file
            platform_config: Platform-specific configuration
            detect_faces: Whether to detect and crop faces
            sampling_method: Frame sampling method
            augment: Whether to apply augmentation

        Returns:
            Dictionary with processed video data and metadata
        """
        start_time = time.time()

        try:
            video_path = Path(video_path)

            # Get video info
            video_info = self.load_video_info(video_path)

            # Apply platform compression if specified
            platform = platform_config.get('platform', 'generic') if platform_config else 'generic'
            processed_video_path = video_path

            if platform != 'generic':
                processed_video_path = self.apply_platform_compression(video_path, platform)

            # Extract frames
            frames = self.extract_frames(
                processed_video_path,
                max_frames=self.frames_per_clip * self.sampling_rate * 2  # Extract more for better sampling
            )

            if not frames:
                raise ValueError(f"No frames extracted from {video_path}")

            # Detect faces if requested
            face_info = []
            if detect_faces:
                face_detections = self.detect_faces_in_frames(frames)
                face_info = face_detections

                # Check if any faces detected
                total_faces = sum(len(faces) for faces in face_detections)
                if total_faces > 0:
                    frames = self.crop_face_regions(frames, face_detections)

            # Sample frames for temporal analysis
            sampled_frames = self.sample_frames_temporal(frames, method=sampling_method)

            # Preprocess frames to tensors
            processed_tensor = self.preprocess_frames_batch(sampled_frames, augment=augment)

            # Calculate frame timestamps
            fps = video_info['fps']
            frame_timestamps = [i / fps for i in range(len(sampled_frames))] if fps > 0 else []

            processing_time = time.time() - start_time

            # Clean up temporary files
            if processed_video_path != video_path:
                try:
                    processed_video_path.unlink()
                except:
                    pass

            return {
                'processed_frames': processed_tensor,
                'original_video_info': video_info,
                'frames_extracted': len(frames),
                'frames_sampled': len(sampled_frames),
                'face_info': face_info,
                'frame_timestamps': frame_timestamps,
                'sampling_method': sampling_method,
                'platform': platform,
                'processing_time': processing_time,
                'source_path': str(video_path),
                'augmented': augment
            }

        except Exception as e:
            logger.error(f"Error preprocessing video {video_path}: {e}")
            raise

    def preprocess(self,
                  video_input: Union[str, Path],
                  frames_per_clip: Optional[int] = None,
                  platform_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main preprocessing method (wrapper for backward compatibility)

        Args:
            video_input: Input video path
            frames_per_clip: Number of frames per clip (overrides instance default)
            platform_config: Platform configuration

        Returns:
            Preprocessing results
        """
        if frames_per_clip:
            original_frames_per_clip = self.frames_per_clip
            self.frames_per_clip = frames_per_clip

        try:
            result = self.preprocess_single(
                video_input,
                platform_config=platform_config,
                detect_faces=True,
                sampling_method='uniform',
                augment=False
            )
            return result
        finally:
            if frames_per_clip:
                self.frames_per_clip = original_frames_per_clip

    def load_dataset_split(self,
                          dataset_path: Union[str, Path],
                          split: str = 'train',
                          limit: Optional[int] = None,
                          shuffle: bool = True) -> Dict[str, List]:
        """
        Load dataset split following the structure:
        dataset/video/{train,validation,test}/{FAKE,REAL}/

        Args:
            dataset_path: Path to dataset root
            split: Dataset split ('train', 'validation', 'test')
            limit: Maximum number of samples per class
            shuffle: Whether to shuffle the samples

        Returns:
            Dictionary with file paths and labels
        """
        dataset_root = Path(dataset_path)
        split_path = dataset_root / 'video' / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split path not found: {split_path}")

        # Load files from each class
        fake_files = []
        real_files = []

        # Video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']

        # Load FAKE samples
        fake_path = split_path / 'FAKE'
        if fake_path.exists():
            for ext in video_extensions:
                fake_files.extend(list(fake_path.glob(f'*{ext}')))
                fake_files.extend(list(fake_path.glob(f'*{ext.upper()}')))

        # Load REAL samples
        real_path = split_path / 'REAL'
        if real_path.exists():
            for ext in video_extensions:
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

        logger.info(f"Loaded {len(file_paths)} videos from {split} split "
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
                          batch_size: int = 8,
                          shuffle: bool = True,
                          augment: bool = None,
                          num_workers: int = 2) -> torch.utils.data.DataLoader:
        """
        Create PyTorch DataLoader for video processing

        Args:
            dataset_path: Path to dataset
            split: Dataset split
            batch_size: Batch size (typically smaller for videos)
            shuffle: Whether to shuffle data
            augment: Whether to augment
            num_workers: Number of worker processes

        Returns:
            PyTorch DataLoader
        """
        if augment is None:
            augment = (split == 'train')

        dataset_info = self.load_dataset_split(dataset_path, split, shuffle=shuffle)

        class VideoDataset(torch.utils.data.Dataset):
            def __init__(self, file_paths, labels, preprocessor, augment=False, platform_config=None):
                self.file_paths = file_paths
                self.labels = labels
                self.preprocessor = preprocessor
                self.augment = augment
                self.platform_config = platform_config

            def __len__(self):
                return len(self.file_paths)

            def __getitem__(self, idx):
                try:
                    result = self.preprocessor.preprocess_single(
                        self.file_paths[idx],
                        platform_config=self.platform_config,
                        detect_faces=True,
                        sampling_method='uniform',
                        augment=self.augment
                    )
                    return result['processed_frames'], self.labels[idx]
                except Exception as e:
                    logger.warning(f"Error loading video {self.file_paths[idx]}: {e}")
                    # Return dummy tensor
                    dummy_video = torch.zeros(self.preprocessor.frames_per_clip, 3, *self.preprocessor.target_size)
                    return dummy_video, self.labels[idx]

        dataset = VideoDataset(
            dataset_info['file_paths'],
            dataset_info['labels'],
            self,
            augment=augment
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        logger.info(f"Created Video DataLoader for {split} split with {len(dataset)} samples")

        return dataloader

    def get_dataset_statistics(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the video dataset

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
        video_root = dataset_root / 'video'

        if not video_root.exists():
            return {'error': f'Video dataset path not found: {video_root}'}

        splits = ['train', 'validation', 'test']

        for split in splits:
            split_path = video_root / split
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
