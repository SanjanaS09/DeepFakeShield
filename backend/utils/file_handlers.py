"""
File Handling Utilities for Multi-Modal Deepfake Detection
Comprehensive file operations for images, videos, audio, and model files
Supports multiple formats, validation, conversion, and secure handling
"""

import os
import cv2
import numpy as np
import logging
import mimetypes
import hashlib
import json
import pickle
import tempfile
import shutil
from typing import Dict, List, Tuple, Union, Optional, Any, IO
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class FileHandler:
    """
    Comprehensive file handler for multi-modal deepfake detection system
    Handles images, videos, audio files, models, and configuration files
    """

    def __init__(self, 
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 temp_dir: Optional[str] = None,
                 secure_mode: bool = True):
        """
        Initialize File Handler

        Args:
            max_file_size: Maximum file size in bytes
            temp_dir: Temporary directory for file operations
            secure_mode: Enable security checks
        """
        self.max_file_size = max_file_size
        self.secure_mode = secure_mode

        # Set up temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / 'deepfake_detection'

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize supported formats
        self._init_supported_formats()

        # Initialize file type validators
        self._init_validators()

        logger.info(f"Initialized FileHandler with temp dir: {self.temp_dir}")

    def _init_supported_formats(self):
        """Initialize supported file formats"""

        # Image formats
        self.image_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
            '.webp', '.gif', '.ico', '.ppm', '.pgm', '.pbm'
        }

        self.image_mimetypes = {
            'image/jpeg', 'image/png', 'image/bmp', 'image/tiff',
            'image/webp', 'image/gif', 'image/x-icon'
        }

        # Video formats
        self.video_extensions = {
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 
            '.webm', '.m4v', '.3gp', '.ogv', '.mpg', '.mpeg'
        }

        self.video_mimetypes = {
            'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
            'video/x-matroska', 'video/webm', 'video/x-flv'
        }

        # Audio formats
        self.audio_extensions = {
            '.wav', '.mp3', '.aac', '.flac', '.ogg', '.wma', 
            '.m4a', '.opus', '.aiff', '.au', '.ra'
        }

        self.audio_mimetypes = {
            'audio/wav', 'audio/mpeg', 'audio/aac', 'audio/flac',
            'audio/ogg', 'audio/x-ms-wma', 'audio/mp4'
        }

        # Model formats
        self.model_extensions = {
            '.pth', '.pt', '.pkl', '.pickle', '.joblib', '.h5', 
            '.pb', '.onnx', '.tflite', '.bin'
        }

        # Configuration formats
        self.config_extensions = {
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'
        }

    def _init_validators(self):
        """Initialize file type validators"""

        # Magic bytes for file type detection
        self.magic_bytes = {
            # Images
            b'\xff\xd8\xff': 'jpeg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'BM': 'bmp',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif',
            b'RIFF': 'webp',  # Also AVI, WAV
            b'\x00\x00\x01\x00': 'ico',

            # Videos
            b'\x00\x00\x00 ftypmp4': 'mp4',
            b'\x00\x00\x00\x18ftypmp4': 'mp4',
            b'\x1a\x45\xdf\xa3': 'mkv',
            b'FLV\x01': 'flv',

            # Audio
            b'ID3': 'mp3',
            b'\xff\xfb': 'mp3',
            b'\xff\xf3': 'mp3',
            b'\xff\xf2': 'mp3',
            b'fLaC': 'flac',
            b'OggS': 'ogg',
        }

    def validate_file(self, file_path: Union[str, Path], 
                     expected_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate file existence, size, type, and security

        Args:
            file_path: Path to file
            expected_type: Expected file type ('image', 'video', 'audio', 'model')

        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        validation = {
            'valid': False,
            'exists': False,
            'size_valid': False,
            'type_valid': False,
            'secure': False,
            'file_type': None,
            'mime_type': None,
            'size_bytes': 0,
            'errors': []
        }

        try:
            # Check existence
            if not file_path.exists():
                validation['errors'].append(f"File does not exist: {file_path}")
                return validation

            validation['exists'] = True

            # Check if it's a file (not directory)
            if not file_path.is_file():
                validation['errors'].append(f"Path is not a file: {file_path}")
                return validation

            # Check file size
            file_size = file_path.stat().st_size
            validation['size_bytes'] = file_size

            if file_size > self.max_file_size:
                validation['errors'].append(f"File too large: {file_size} > {self.max_file_size}")
            else:
                validation['size_valid'] = True

            # Check file type
            file_type, mime_type = self._detect_file_type(file_path)
            validation['file_type'] = file_type
            validation['mime_type'] = mime_type

            # Validate against expected type
            if expected_type:
                if not self._is_expected_type(file_type, expected_type):
                    validation['errors'].append(f"File type {file_type} does not match expected {expected_type}")
                else:
                    validation['type_valid'] = True
            else:
                validation['type_valid'] = file_type is not None

            # Security checks
            if self.secure_mode:
                security_check = self._security_check(file_path)
                validation['secure'] = security_check['secure']
                if not security_check['secure']:
                    validation['errors'].extend(security_check['issues'])
            else:
                validation['secure'] = True

            # Overall validity
            validation['valid'] = (
                validation['exists'] and 
                validation['size_valid'] and 
                validation['type_valid'] and 
                validation['secure']
            )

        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"File validation failed for {file_path}: {e}")

        return validation

    def _detect_file_type(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Detect file type using extension and magic bytes"""

        try:
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))

            # Check extension
            extension = file_path.suffix.lower()

            # Determine category based on extension
            file_type = None
            if extension in self.image_extensions:
                file_type = 'image'
            elif extension in self.video_extensions:
                file_type = 'video'
            elif extension in self.audio_extensions:
                file_type = 'audio'
            elif extension in self.model_extensions:
                file_type = 'model'
            elif extension in self.config_extensions:
                file_type = 'config'

            # Verify with magic bytes if possible
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(32)  # Read first 32 bytes

                for magic, detected_type in self.magic_bytes.items():
                    if header.startswith(magic.encode('unicode_escape').decode('ascii').encode('latin-1')):
                        # Verify consistency
                        if file_type and not self._types_consistent(file_type, detected_type):
                            logger.warning(f"File type mismatch for {file_path}: extension suggests {file_type}, magic bytes suggest {detected_type}")
                        break

            except Exception as e:
                logger.warning(f"Could not read magic bytes from {file_path}: {e}")

            return file_type, mime_type

        except Exception as e:
            logger.error(f"File type detection failed for {file_path}: {e}")
            return None, None

    def _types_consistent(self, extension_type: str, magic_type: str) -> bool:
        """Check if extension-based type is consistent with magic bytes type"""

        consistency_map = {
            'image': ['jpeg', 'png', 'bmp', 'gif', 'webp', 'ico'],
            'video': ['mp4', 'mkv', 'flv', 'avi'],
            'audio': ['mp3', 'flac', 'ogg', 'wav']
        }

        return magic_type in consistency_map.get(extension_type, [])

    def _is_expected_type(self, detected_type: str, expected_type: str) -> bool:
        """Check if detected type matches expected type"""
        return detected_type == expected_type

    def _security_check(self, file_path: Path) -> Dict[str, Any]:
        """Perform security checks on file"""

        security = {
            'secure': True,
            'issues': []
        }

        try:
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                security['issues'].append("File is not readable")
                security['secure'] = False

            # Check for suspicious file names
            suspicious_patterns = [
                '..', '~', '$', '|', ';', '&', '`', 
                'exec', 'eval', 'system', 'shell'
            ]

            file_name = file_path.name.lower()
            for pattern in suspicious_patterns:
                if pattern in file_name:
                    security['issues'].append(f"Suspicious pattern in filename: {pattern}")
                    security['secure'] = False

            # Check for very long filenames
            if len(file_name) > 255:
                security['issues'].append("Filename too long")
                security['secure'] = False

            # Check file content for basic security (for text files)
            if file_path.suffix.lower() in {'.txt', '.json', '.yaml', '.yml', '.xml', '.csv'}:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1024)  # Read first 1KB

                    # Check for suspicious content
                    suspicious_content = [
                        '<script', 'javascript:', 'vbscript:', 'onload=', 
                        'onerror=', 'eval(', 'exec(', 'system('
                    ]

                    content_lower = content.lower()
                    for pattern in suspicious_content:
                        if pattern in content_lower:
                            security['issues'].append(f"Suspicious content pattern: {pattern}")
                            security['secure'] = False

                except Exception:
                    pass  # Not a text file or reading failed

        except Exception as e:
            security['issues'].append(f"Security check error: {str(e)}")
            security['secure'] = False

        return security

    def load_image(self, file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image file safely

        Args:
            file_path: Path to image file

        Returns:
            Image array or None if failed
        """
        try:
            file_path = Path(file_path)

            # Validate file
            validation = self.validate_file(file_path, 'image')
            if not validation['valid']:
                logger.error(f"Image validation failed: {validation['errors']}")
                return None

            # Load image
            image = cv2.imread(str(file_path))
            if image is None:
                logger.error(f"Failed to load image: {file_path}")
                return None

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            logger.info(f"Successfully loaded image: {file_path} ({image_rgb.shape})")
            return image_rgb

        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None

    def save_image(self, image: np.ndarray, file_path: Union[str, Path], 
                   quality: int = 95) -> bool:
        """
        Save image to file

        Args:
            image: Image array (RGB)
            file_path: Output file path
            quality: JPEG quality (for JPEG files)

        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)

            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Set save parameters based on format
            extension = file_path.suffix.lower()
            if extension in {'.jpg', '.jpeg'}:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif extension == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            elif extension == '.webp':
                params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            else:
                params = []

            # Save image
            success = cv2.imwrite(str(file_path), image_bgr, params)

            if success:
                logger.info(f"Successfully saved image: {file_path}")
            else:
                logger.error(f"Failed to save image: {file_path}")

            return success

        except Exception as e:
            logger.error(f"Error saving image to {file_path}: {e}")
            return False

    def load_video_frames(self, file_path: Union[str, Path], 
                         max_frames: Optional[int] = None,
                         frame_step: int = 1) -> Optional[List[np.ndarray]]:
        """
        Load frames from video file

        Args:
            file_path: Path to video file
            max_frames: Maximum number of frames to load
            frame_step: Step size for frame sampling

        Returns:
            List of frames or None if failed
        """
        try:
            file_path = Path(file_path)

            # Validate file
            validation = self.validate_file(file_path, 'video')
            if not validation['valid']:
                logger.error(f"Video validation failed: {validation['errors']}")
                return None

            # Open video
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {file_path}")
                return None

            frames = []
            frame_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Apply frame step
                    if frame_count % frame_step == 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)

                        # Check max frames limit
                        if max_frames and len(frames) >= max_frames:
                            break

                    frame_count += 1

            finally:
                cap.release()

            logger.info(f"Successfully loaded {len(frames)} frames from video: {file_path}")
            return frames

        except Exception as e:
            logger.error(f"Error loading video frames from {file_path}: {e}")
            return None

    def load_audio(self, file_path: Union[str, Path]) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio file

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate) or None if failed
        """
        try:
            file_path = Path(file_path)

            # Validate file
            validation = self.validate_file(file_path, 'audio')
            if not validation['valid']:
                logger.error(f"Audio validation failed: {validation['errors']}")
                return None

            # Try different loading methods
            audio_data = None
            sample_rate = None

            # Method 1: librosa
            try:
                import librosa
                audio_data, sample_rate = librosa.load(str(file_path), sr=None, mono=True)
                logger.info(f"Loaded audio with librosa: {file_path}")
            except ImportError:
                logger.warning("librosa not available")
            except Exception as e:
                logger.warning(f"librosa loading failed: {e}")

            # Method 2: soundfile
            if audio_data is None:
                try:
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(str(file_path))
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                    logger.info(f"Loaded audio with soundfile: {file_path}")
                except ImportError:
                    logger.warning("soundfile not available")
                except Exception as e:
                    logger.warning(f"soundfile loading failed: {e}")

            # Method 3: Basic WAV support with scipy
            if audio_data is None:
                try:
                    from scipy.io import wavfile
                    sample_rate, audio_data = wavfile.read(str(file_path))
                    audio_data = audio_data.astype(np.float32)
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    # Normalize to [-1, 1]
                    audio_data = audio_data / np.max(np.abs(audio_data))
                    logger.info(f"Loaded audio with scipy: {file_path}")
                except ImportError:
                    logger.warning("scipy not available")
                except Exception as e:
                    logger.warning(f"scipy loading failed: {e}")

            if audio_data is not None:
                logger.info(f"Successfully loaded audio: {file_path} ({len(audio_data)} samples, {sample_rate}Hz)")
                return audio_data, sample_rate
            else:
                logger.error(f"Failed to load audio file: {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading audio from {file_path}: {e}")
            return None

    def save_audio(self, audio_data: np.ndarray, sample_rate: int,
                   file_path: Union[str, Path]) -> bool:
        """
        Save audio to file

        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            file_path: Output file path

        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)

            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Try different saving methods
            success = False

            # Method 1: soundfile
            try:
                import soundfile as sf
                sf.write(str(file_path), audio_data, sample_rate)
                success = True
                logger.info(f"Saved audio with soundfile: {file_path}")
            except ImportError:
                logger.warning("soundfile not available for saving")
            except Exception as e:
                logger.warning(f"soundfile saving failed: {e}")

            # Method 2: scipy (for WAV files)
            if not success and file_path.suffix.lower() == '.wav':
                try:
                    from scipy.io import wavfile
                    # Convert to int16 for WAV
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wavfile.write(str(file_path), sample_rate, audio_int16)
                    success = True
                    logger.info(f"Saved audio with scipy: {file_path}")
                except ImportError:
                    logger.warning("scipy not available for saving")
                except Exception as e:
                    logger.warning(f"scipy saving failed: {e}")

            if success:
                logger.info(f"Successfully saved audio: {file_path}")
            else:
                logger.error(f"Failed to save audio: {file_path}")

            return success

        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")
            return False

    def load_model(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Load model file

        Args:
            file_path: Path to model file

        Returns:
            Loaded model or None if failed
        """
        try:
            file_path = Path(file_path)

            # Validate file
            validation = self.validate_file(file_path, 'model')
            if not validation['valid']:
                logger.error(f"Model validation failed: {validation['errors']}")
                return None

            extension = file_path.suffix.lower()

            # PyTorch models
            if extension in {'.pth', '.pt'}:
                try:
                    import torch
                    model = torch.load(str(file_path), map_location='cpu')
                    logger.info(f"Loaded PyTorch model: {file_path}")
                    return model
                except ImportError:
                    logger.error("PyTorch not available")
                    return None

            # Pickle files
            elif extension in {'.pkl', '.pickle'}:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded pickle model: {file_path}")
                return model

            # Joblib files
            elif extension == '.joblib':
                try:
                    import joblib
                    model = joblib.load(str(file_path))
                    logger.info(f"Loaded joblib model: {file_path}")
                    return model
                except ImportError:
                    logger.error("joblib not available")
                    return None

            # TensorFlow/Keras models
            elif extension == '.h5':
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(str(file_path))
                    logger.info(f"Loaded Keras model: {file_path}")
                    return model
                except ImportError:
                    logger.error("TensorFlow not available")
                    return None

            # ONNX models
            elif extension == '.onnx':
                try:
                    import onnx
                    model = onnx.load(str(file_path))
                    logger.info(f"Loaded ONNX model: {file_path}")
                    return model
                except ImportError:
                    logger.error("ONNX not available")
                    return None

            else:
                logger.error(f"Unsupported model format: {extension}")
                return None

        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}")
            return None

    def save_model(self, model: Any, file_path: Union[str, Path]) -> bool:
        """
        Save model to file

        Args:
            model: Model object to save
            file_path: Output file path

        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)

            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            extension = file_path.suffix.lower()

            # PyTorch models
            if extension in {'.pth', '.pt'}:
                try:
                    import torch
                    torch.save(model, str(file_path))
                    logger.info(f"Saved PyTorch model: {file_path}")
                    return True
                except ImportError:
                    logger.error("PyTorch not available")
                    return False

            # Pickle files
            elif extension in {'.pkl', '.pickle'}:
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved pickle model: {file_path}")
                return True

            # Joblib files
            elif extension == '.joblib':
                try:
                    import joblib
                    joblib.dump(model, str(file_path))
                    logger.info(f"Saved joblib model: {file_path}")
                    return True
                except ImportError:
                    logger.error("joblib not available")
                    return False

            else:
                logger.error(f"Unsupported model save format: {extension}")
                return False

        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {e}")
            return False

    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """
        Load JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data or None if failed
        """
        try:
            file_path = Path(file_path)

            # Validate file
            validation = self.validate_file(file_path)
            if not validation['valid']:
                logger.error(f"JSON validation failed: {validation['errors']}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Successfully loaded JSON: {file_path}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return None

    def save_json(self, data: Dict, file_path: Union[str, Path], 
                  indent: int = 2) -> bool:
        """
        Save data to JSON file

        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation

        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)

            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)

            logger.info(f"Successfully saved JSON: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {e}")
            return False

    def get_file_hash(self, file_path: Union[str, Path], 
                      algorithm: str = 'sha256') -> Optional[str]:
        """
        Calculate file hash

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            File hash or None if failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return None

            # Initialize hasher
            if algorithm == 'md5':
                hasher = hashlib.md5()
            elif algorithm == 'sha1':
                hasher = hashlib.sha1()
            elif algorithm == 'sha256':
                hasher = hashlib.sha256()
            else:
                logger.error(f"Unsupported hash algorithm: {algorithm}")
                return None

            # Read file in chunks
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)

            file_hash = hasher.hexdigest()
            logger.info(f"Calculated {algorithm} hash for {file_path}: {file_hash}")
            return file_hash

        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None

    def create_temp_file(self, suffix: str = '', prefix: str = 'tmp_') -> Path:
        """
        Create temporary file

        Args:
            suffix: File suffix/extension
            prefix: File prefix

        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, 
            dir=self.temp_dir, delete=False
        )
        temp_file.close()

        temp_path = Path(temp_file.name)
        logger.info(f"Created temporary file: {temp_path}")
        return temp_path

    def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """
        Clean up old temporary files

        Args:
            older_than_hours: Remove files older than this many hours

        Returns:
            Number of files removed
        """
        try:
            import time

            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)

            removed_count = 0

            for temp_file in self.temp_dir.glob('*'):
                if temp_file.is_file():
                    file_mtime = temp_file.stat().st_mtime
                    if file_mtime < cutoff_time:
                        try:
                            temp_file.unlink()
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Could not remove temp file {temp_file}: {e}")

            logger.info(f"Cleaned up {removed_count} temporary files")
            return removed_count

        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
            return 0

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)

        info = {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'exists': False,
            'size_bytes': 0,
            'size_mb': 0.0,
            'file_type': None,
            'mime_type': None,
            'hash_sha256': None,
            'created_time': None,
            'modified_time': None,
            'validation': None
        }

        try:
            if file_path.exists():
                info['exists'] = True

                # File size
                size_bytes = file_path.stat().st_size
                info['size_bytes'] = size_bytes
                info['size_mb'] = size_bytes / (1024 * 1024)

                # File times
                stat = file_path.stat()
                info['created_time'] = stat.st_ctime
                info['modified_time'] = stat.st_mtime

                # File type
                file_type, mime_type = self._detect_file_type(file_path)
                info['file_type'] = file_type
                info['mime_type'] = mime_type

                # File hash (for reasonable file sizes)
                if size_bytes < 50 * 1024 * 1024:  # Less than 50MB
                    info['hash_sha256'] = self.get_file_hash(file_path, 'sha256')

                # Validation
                info['validation'] = self.validate_file(file_path)

        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")

        return info


# --- Utility wrapper functions for detection_routes compatibility ---

# Create a singleton instance for reuse
# ============================================================================
# WRAPPER FUNCTIONS FOR FLASK/API COMPATIBILITY
# ============================================================================

# Create a singleton instance for reuse
_file_handler_instance = FileHandler()

def save_uploaded_file(file_storage, subdir: str = "uploads") -> Path:
    """Save an uploaded file from Flask request securely."""
    try:
        upload_dir = _file_handler_instance.temp_dir / subdir
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        from werkzeug.utils import secure_filename
        filename = secure_filename(file_storage.filename)
        safe_path = upload_dir / filename
        
        file_storage.save(str(safe_path))
        
        validation = _file_handler_instance.validate_file(safe_path)
        if not validation['valid']:
            safe_path.unlink(missing_ok=True)
            raise ValueError(f"Invalid file uploaded: {validation['errors']}")
        
        logger.info(f"Successfully saved uploaded file: {safe_path}")
        return safe_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise


def cleanup_temp_files(hours: int = 24) -> int:
    """Clean up old temporary files."""
    return _file_handler_instance.cleanup_temp_files(older_than_hours=hours)


def get_file_handler() -> FileHandler:
    """Get the singleton FileHandler instance."""
    return _file_handler_instance

def get_dataset_stats(dataset_path):
    """
    Calculate basic stats on the dataset folder.
    Args:
        dataset_path (str or Path): Path to dataset directory.

    Returns:
        dict: Statistics such as number of files, total size in bytes.
    """
    dataset_path = Path(dataset_path)
    stats = {
        'num_files': 0,
        'total_size_bytes': 0
    }
    
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise ValueError(f"Dataset path {dataset_path} does not exist or is not a directory.")

    for root, dirs, files in os.walk(dataset_path):
        stats['num_files'] += len(files)
        for f in files:
            fp = Path(root) / f
            # Make sure to check if the path is a file and not a broken symlink
            if fp.is_file():
                stats['total_size_bytes'] += fp.stat().st_size

    return stats


def scan_directory(directory_path, extensions=None):
    """
    Scan directory for files matching extensions.

    Args:
        directory_path (str or Path): Directory to scan.
        extensions (list of str): List of file extensions to include (e.g., ['.jpg', '.png']). 
                                If None, return all files.

    Returns:
        list of Path: List of matching file paths.
    """
    directory_path = Path(directory_path)
    if extensions:
        extensions = [ext.lower() for ext in extensions]
    
    matching_files = []
    
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"Directory path {directory_path} does not exist or is not a directory.")

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if extensions is None or Path(filename).suffix.lower() in extensions:
                matching_files.append(Path(root) / filename)

    return matching_files

__all__ = [
    'FileHandler',
    'save_uploaded_file',
    'cleanup_temp_files',
    'get_file_handler',
    'validate_uploaded_file',
    'cleanup_specific_file',
    'get_dataset_stats',  # <-- ADD THIS
    'scan_directory' 
]
