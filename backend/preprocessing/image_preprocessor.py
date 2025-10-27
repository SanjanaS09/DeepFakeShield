"""
Image Preprocessor for Multi-Modal Deepfake Detection
Handles image loading, preprocessing, face detection, and normalization
Compatible with dataset structure: dataset/image/{train,validation,test}/{FAKE,REAL}/
"""

import os
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Comprehensive image preprocessor for deepfake detection
    Supports face detection, alignment, augmentation, and platform-specific preprocessing
    """

    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 device: str = 'cpu'):
        """
        Initialize Image Preprocessor

        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            device: Device for preprocessing ('cpu', 'cuda')
        """
        self.target_size = target_size
        self.normalize = normalize
        self.device = device

        # Initialize face detection
        self._init_face_detector()

        # Initialize transforms
        self._init_transforms()

        # Platform-specific settings
        self.platform_configs = {
            'whatsapp': {
                'max_size': (480, 360),
                'compression_quality': 80,
                'format': 'JPEG'
            },
            'youtube': {
                'max_size': (1920, 1080),
                'compression_quality': 90,
                'format': 'JPEG'
            },
            'zoom': {
                'max_size': (640, 480),
                'compression_quality': 70,
                'format': 'JPEG'
            },
            'generic': {
                'max_size': (1920, 1080),
                'compression_quality': 95,
                'format': 'JPEG'
            }
        }

        logger.info(f"Initialized ImagePreprocessor with target size {target_size}")

    def _init_face_detector(self):
        """Initialize face detection models"""
        try:
            # OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            # Try to initialize more advanced face detectors
            try:
                # MediaPipe Face Detection (if available)
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detector_mp = self.mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                self.mp_draw = mp.solutions.drawing_utils
                logger.info("MediaPipe face detection initialized")
            except ImportError:
                self.face_detector_mp = None
                logger.info("MediaPipe not available, using OpenCV only")

        except Exception as e:
            logger.warning(f"Face detection initialization failed: {e}")
            self.face_cascade = None
            self.face_detector_mp = None

    def _init_transforms(self):
        """Initialize image transformation pipelines"""
        # Standard ImageNet normalization
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        # Basic transforms without normalization
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Augmentation transforms for training
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array in RGB format
        """
        try:
            image_path = Path(image_path)

            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load with PIL for better format support
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                image_array = np.array(img)

            return image_array

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def detect_faces(self, image: np.ndarray, method: str = 'opencv') -> List[Dict[str, Any]]:
        """
        Detect faces in image using specified method

        Args:
            image: Input image as numpy array
            method: Detection method ('opencv', 'mediapipe', 'both')

        Returns:
            List of face detection results
        """
        faces = []

        if method in ['opencv', 'both'] and self.face_cascade is not None:
            # OpenCV detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, w, h) in detected_faces:
                faces.append({
                    'method': 'opencv',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 1.0,  # OpenCV doesn't provide confidence
                    'center': [int(x + w//2), int(y + h//2)],
                    'area': int(w * h)
                })

        if method in ['mediapipe', 'both'] and self.face_detector_mp is not None:
            # MediaPipe detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = self.face_detector_mp.process(rgb_image)

            if results.detections:
                h, w, _ = image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    faces.append({
                        'method': 'mediapipe',
                        'bbox': [x, y, width, height],
                        'confidence': detection.score[0],
                        'center': [int(x + width//2), int(y + height//2)],
                        'area': int(width * height)
                    })

        return faces

    def align_face(self, image: np.ndarray, face_bbox: List[int]) -> np.ndarray:
        """
        Align face region using detected bounding box

        Args:
            image: Input image
            face_bbox: Face bounding box [x, y, width, height]

        Returns:
            Aligned face region
        """
        try:
            x, y, w, h = face_bbox

            # Add padding around face
            padding = 0.2
            pad_w = int(w * padding)
            pad_h = int(h * padding)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)

            # Extract face region
            face_region = image[y1:y2, x1:x2]

            return face_region

        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return image

    def apply_platform_preprocessing(self, image: np.ndarray, platform: str = 'generic') -> np.ndarray:
        """
        Apply platform-specific preprocessing

        Args:
            image: Input image
            platform: Platform type (whatsapp, youtube, zoom, generic)

        Returns:
            Platform-processed image
        """
        config = self.platform_configs.get(platform, self.platform_configs['generic'])

        # Convert to PIL for processing
        pil_image = Image.fromarray(image)

        # Resize to platform limits
        max_size = config['max_size']
        if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Apply compression simulation
        import io
        compression_quality = config['compression_quality']

        # Save and reload with compression
        buffer = io.BytesIO()
        pil_image.save(buffer, format=config['format'], quality=compression_quality)
        buffer.seek(0)

        compressed_image = Image.open(buffer)
        processed_array = np.array(compressed_image)

        return processed_array

    def augment_image(self, image: np.ndarray, augmentation_type: str = 'light') -> np.ndarray:
        """
        Apply data augmentation to image

        Args:
            image: Input image
            augmentation_type: Augmentation level ('light', 'medium', 'heavy')

        Returns:
            Augmented image
        """
        pil_image = Image.fromarray(image)

        if augmentation_type == 'light':
            # Light augmentations
            if np.random.rand() > 0.5:
                pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            if np.random.rand() > 0.7:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(np.random.uniform(0.9, 1.1))

        elif augmentation_type == 'medium':
            # Medium augmentations
            if np.random.rand() > 0.5:
                pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            if np.random.rand() > 0.6:
                angle = np.random.uniform(-10, 10)
                pil_image = pil_image.rotate(angle, fillcolor=(128, 128, 128))

            if np.random.rand() > 0.6:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(np.random.uniform(0.8, 1.2))

        elif augmentation_type == 'heavy':
            # Heavy augmentations
            transformations = self.augment_transform
            tensor_image = transformations(pil_image)
            pil_image = transforms.ToPILImage()(tensor_image)

        return np.array(pil_image)

    def preprocess_single(self, 
                         image: Union[np.ndarray, str, Path],
                         platform_config: Optional[Dict] = None,
                         detect_faces: bool = True,
                         augment: bool = False) -> Dict[str, Any]:
        """
        Preprocess a single image

        Args:
            image: Input image (array, path, or PIL Image)
            platform_config: Platform-specific configuration
            detect_faces: Whether to detect faces
            augment: Whether to apply augmentation

        Returns:
            Dictionary with processed image and metadata
        """
        start_time = time.time()

        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image_array = self.load_image(image)
                source_path = str(image)
            else:
                image_array = image
                source_path = None

            original_shape = image_array.shape

            # Face detection
            faces = []
            if detect_faces:
                faces = self.detect_faces(image_array, method='opencv')

            # Focus on largest face if detected
            if faces:
                largest_face = max(faces, key=lambda x: x['area'])
                face_region = self.align_face(image_array, largest_face['bbox'])
                processed_image = cv2.resize(face_region, self.target_size)
            else:
                # Use full image if no faces detected
                processed_image = cv2.resize(image_array, self.target_size)

            # Apply platform preprocessing if specified
            platform = platform_config.get('platform', 'generic') if platform_config else 'generic'
            if platform != 'generic':
                processed_image = self.apply_platform_preprocessing(processed_image, platform)
                processed_image = cv2.resize(processed_image, self.target_size)

            # Apply augmentation if requested
            if augment:
                augmentation_level = platform_config.get('augmentation_level', 'light') if platform_config else 'light'
                processed_image = self.augment_image(processed_image, augmentation_level)

            # Convert to tensor
            pil_image = Image.fromarray(processed_image)

            if self.normalize:
                tensor_image = self.normalize_transform(pil_image)
            else:
                tensor_image = self.basic_transform(pil_image)

            processing_time = time.time() - start_time

            return {
                'processed_image': tensor_image,
                'original_shape': original_shape,
                'target_shape': self.target_size,
                'faces_detected': len(faces),
                'face_info': faces,
                'platform': platform,
                'processing_time': processing_time,
                'source_path': source_path,
                'augmented': augment
            }

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def preprocess(self, 
                  image_input: Union[np.ndarray, str, Path], 
                  platform_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main preprocessing method (wrapper for backward compatibility)

        Args:
            image_input: Input image
            platform_config: Platform configuration

        Returns:
            Preprocessing results
        """
        return self.preprocess_single(
            image_input, 
            platform_config=platform_config,
            detect_faces=True,
            augment=False
        )

    def load_dataset_split(self, 
                          dataset_path: Union[str, Path],
                          split: str = 'train',
                          limit: Optional[int] = None,
                          shuffle: bool = True) -> Dict[str, List]:
        """
        Load dataset split following the structure:
        dataset/image/{train,validation,test}/{FAKE,REAL}/

        Args:
            dataset_path: Path to dataset root
            split: Dataset split ('train', 'validation', 'test')
            limit: Maximum number of samples per class
            shuffle: Whether to shuffle the samples

        Returns:
            Dictionary with file paths and labels
        """
        dataset_root = Path(dataset_path)
        split_path = dataset_root / 'image' / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split path not found: {split_path}")

        # Load files from each class
        fake_files = []
        real_files = []

        # Load FAKE samples
        fake_path = split_path / 'FAKE'
        if fake_path.exists():
            fake_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for ext in fake_extensions:
                fake_files.extend(list(fake_path.glob(f'*{ext}')))
                fake_files.extend(list(fake_path.glob(f'*{ext.upper()}')))

        # Load REAL samples
        real_path = split_path / 'REAL'
        if real_path.exists():
            real_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for ext in real_extensions:
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

        logger.info(f"Loaded {len(file_paths)} images from {split} split "
                   f"({len(fake_files)} fake, {len(real_files)} real)")

        return {
            'file_paths': [str(p) for p in file_paths],
            'labels': labels,
            'class_counts': {'FAKE': len(fake_files), 'REAL': len(real_files)},
            'split': split,
            'total_samples': len(file_paths)
        }

    def preprocess_batch(self, 
                        file_paths: List[str],
                        labels: List[int],
                        batch_size: int = 32,
                        augment: bool = False,
                        platform_config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess batch of images for training/evaluation

        Args:
            file_paths: List of image file paths
            labels: List of corresponding labels (0=REAL, 1=FAKE)
            batch_size: Batch size for processing
            augment: Whether to apply data augmentation
            platform_config: Platform-specific configuration

        Returns:
            Dictionary with batched tensors
        """
        processed_images = []
        processed_labels = []

        for i, (file_path, label) in enumerate(zip(file_paths, labels)):
            try:
                # Preprocess single image
                result = self.preprocess_single(
                    file_path,
                    platform_config=platform_config,
                    detect_faces=True,
                    augment=augment
                )

                processed_images.append(result['processed_image'])
                processed_labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to process image {file_path}: {e}")
                continue

        if not processed_images:
            raise ValueError("No images were successfully processed")

        # Stack into batch tensors
        image_batch = torch.stack(processed_images)
        label_batch = torch.tensor(processed_labels, dtype=torch.long)

        return {
            'images': image_batch,
            'labels': label_batch,
            'batch_size': len(processed_images),
            'successful_samples': len(processed_images),
            'failed_samples': len(file_paths) - len(processed_images)
        }

    def create_data_loader(self,
                          dataset_path: Union[str, Path],
                          split: str = 'train',
                          batch_size: int = 32,
                          shuffle: bool = True,
                          augment: bool = None,
                          num_workers: int = 4) -> torch.utils.data.DataLoader:
        """
        Create PyTorch DataLoader for training/evaluation

        Args:
            dataset_path: Path to dataset
            split: Dataset split
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to augment (default: True for train, False for others)
            num_workers: Number of worker processes

        Returns:
            PyTorch DataLoader
        """
        # Set augmentation default based on split
        if augment is None:
            augment = (split == 'train')

        # Load dataset split
        dataset_info = self.load_dataset_split(dataset_path, split, shuffle=shuffle)

        # Create custom dataset class
        class ImageDataset(torch.utils.data.Dataset):
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
                        augment=self.augment
                    )
                    return result['processed_image'], self.labels[idx]
                except Exception as e:
                    logger.warning(f"Error loading image {self.file_paths[idx]}: {e}")
                    # Return a dummy tensor if loading fails
                    dummy_image = torch.zeros(3, *self.preprocessor.target_size)
                    return dummy_image, self.labels[idx]

        # Create dataset and dataloader
        dataset = ImageDataset(
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

        logger.info(f"Created DataLoader for {split} split with {len(dataset)} samples")

        return dataloader

    def get_dataset_statistics(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the image dataset

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
        image_root = dataset_root / 'image'

        if not image_root.exists():
            return {'error': f'Image dataset path not found: {image_root}'}

        splits = ['train', 'validation', 'test']

        for split in splits:
            split_path = image_root / split
            if not split_path.exists():
                stats['splits'][split] = {'error': f'Split not found: {split}'}
                continue

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

        # Calculate overall ratios
        if stats['overall']['total'] > 0:
            stats['overall']['fake_ratio'] = stats['overall']['FAKE'] / stats['overall']['total']
            stats['overall']['real_ratio'] = stats['overall']['REAL'] / stats['overall']['total']

        return stats
