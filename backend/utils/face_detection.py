"""
Advanced Face Detection Utilities for Multi-Modal Deepfake Detection
Supports multiple detection methods, face alignment, landmark detection, and quality assessment
Integrates OpenCV, MediaPipe, and dlib for robust face analysis
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Advanced face detector with multiple detection methods and face analysis capabilities
    Supports OpenCV Haar/DNN, MediaPipe, and dlib for comprehensive face detection
    """

    def __init__(self, 
                 detection_methods: List[str] = ['opencv', 'mediapipe'],
                 confidence_threshold: float = 0.5,
                 min_face_size: Tuple[int, int] = (30, 30),
                 device: str = 'cpu'):
        """
        Initialize Face Detector

        Args:
            detection_methods: List of detection methods to use
            confidence_threshold: Minimum confidence for face detection
            min_face_size: Minimum face size (width, height)
            device: Device for computation
        """
        self.detection_methods = detection_methods
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.device = device

        # Initialize detectors
        self._init_opencv_detectors()
        self._init_mediapipe_detector()
        self._init_dlib_detector()
        self._init_dnn_detector()

        logger.info(f"Initialized FaceDetector with methods: {detection_methods}")

    def _init_opencv_detectors(self):
        """Initialize OpenCV Haar cascade detectors"""
        try:
            # Face cascade
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

            # Eye cascade for validation
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

            # Profile face cascade
            profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

            self.opencv_available = True
            logger.info("OpenCV Haar cascades initialized successfully")

        except Exception as e:
            logger.warning(f"OpenCV Haar cascade initialization failed: {e}")
            self.opencv_available = False

    def _init_mediapipe_detector(self):
        """Initialize MediaPipe face detector"""
        try:
            import mediapipe as mp

            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils

            # Face detection model
            self.mp_face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for better accuracy, 0 for speed
                min_detection_confidence=self.confidence_threshold
            )

            # Face mesh for landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=0.5
            )

            self.mediapipe_available = True
            logger.info("MediaPipe face detection initialized successfully")

        except ImportError:
            logger.warning("MediaPipe not available")
            self.mediapipe_available = False
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
            self.mediapipe_available = False

    def _init_dlib_detector(self):
        """Initialize dlib face detector and landmark predictor"""
        try:
            import dlib

            # HOG-based face detector
            self.dlib_detector = dlib.get_frontal_face_detector()

            # Try to load landmark predictor
            predictor_paths = [
                'shape_predictor_68_face_landmarks.dat',
                'models/shape_predictor_68_face_landmarks.dat',
                'utils/shape_predictor_68_face_landmarks.dat'
            ]

            self.dlib_predictor = None
            for path in predictor_paths:
                if Path(path).exists():
                    self.dlib_predictor = dlib.shape_predictor(path)
                    logger.info(f"Loaded dlib predictor from {path}")
                    break

            if self.dlib_predictor is None:
                logger.warning("dlib landmark predictor not found. Some features will be limited.")

            self.dlib_available = True
            logger.info("dlib face detection initialized successfully")

        except ImportError:
            logger.warning("dlib not available")
            self.dlib_available = False
        except Exception as e:
            logger.warning(f"dlib initialization failed: {e}")
            self.dlib_available = False

    def _init_dnn_detector(self):
        """Initialize OpenCV DNN face detector"""
        try:
            # Try to load pre-trained DNN models
            model_paths = [
                ('models/opencv_face_detector_uint8.pb', 'models/opencv_face_detector.pbtxt'),
                ('utils/opencv_face_detector_uint8.pb', 'utils/opencv_face_detector.pbtxt')
            ]

            self.dnn_net = None
            for model_path, config_path in model_paths:
                if Path(model_path).exists() and Path(config_path).exists():
                    self.dnn_net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                    logger.info(f"Loaded DNN face detector from {model_path}")
                    break

            if self.dnn_net is None:
                logger.warning("DNN face detector models not found")

            self.dnn_available = self.dnn_net is not None

        except Exception as e:
            logger.warning(f"DNN face detector initialization failed: {e}")
            self.dnn_available = False

    def detect_faces_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using OpenCV Haar cascades

        Args:
            image: Input image

        Returns:
            List of face detection results
        """
        faces = []

        if not self.opencv_available:
            return faces

        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect frontal faces
            detected_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in detected_faces:
                # Validate with eye detection
                face_region = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=3)

                face_info = {
                    'method': 'opencv',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 1.0,  # Haar cascades don't provide confidence
                    'center': [int(x + w//2), int(y + h//2)],
                    'area': int(w * h),
                    'has_eyes': len(eyes) >= 1,  # At least one eye detected
                    'eye_count': len(eyes)
                }

                faces.append(face_info)

            # Try profile detection if no frontal faces found
            if not faces:
                profile_faces = self.profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=self.min_face_size
                )

                for (x, y, w, h) in profile_faces:
                    face_info = {
                        'method': 'opencv_profile',
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': 0.8,  # Lower confidence for profile
                        'center': [int(x + w//2), int(y + h//2)],
                        'area': int(w * h),
                        'has_eyes': False,
                        'eye_count': 0
                    }
                    faces.append(face_info)

        except Exception as e:
            logger.warning(f"OpenCV face detection failed: {e}")

        return faces

    def detect_faces_mediapipe(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using MediaPipe

        Args:
            image: Input image (RGB)

        Returns:
            List of face detection results
        """
        faces = []

        if not self.mediapipe_available:
            return faces

        try:
            # MediaPipe expects RGB
            if len(image.shape) == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Detect faces
            results = self.mp_face_detector.process(rgb_image)

            if results.detections:
                h, w = rgb_image.shape[:2]

                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # Ensure bbox is within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)

                    face_info = {
                        'method': 'mediapipe',
                        'bbox': [x, y, width, height],
                        'confidence': detection.score[0],
                        'center': [int(x + width//2), int(y + height//2)],
                        'area': int(width * height),
                        'keypoints': []
                    }

                    # Extract key points if available
                    if hasattr(detection.location_data, 'relative_keypoints'):
                        for keypoint in detection.location_data.relative_keypoints:
                            kp_x = int(keypoint.x * w)
                            kp_y = int(keypoint.y * h)
                            face_info['keypoints'].append([kp_x, kp_y])

                    faces.append(face_info)

        except Exception as e:
            logger.warning(f"MediaPipe face detection failed: {e}")

        return faces

    def detect_faces_dlib(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using dlib

        Args:
            image: Input image

        Returns:
            List of face detection results
        """
        faces = []

        if not self.dlib_available:
            return faces

        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect faces
            detected_faces = self.dlib_detector(gray)

            for face_rect in detected_faces:
                x, y = face_rect.left(), face_rect.top()
                w, h = face_rect.width(), face_rect.height()

                face_info = {
                    'method': 'dlib',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 1.0,  # dlib doesn't provide confidence
                    'center': [int(x + w//2), int(y + h//2)],
                    'area': int(w * h),
                    'landmarks': []
                }

                # Extract facial landmarks if predictor is available
                if self.dlib_predictor is not None:
                    try:
                        landmarks = self.dlib_predictor(gray, face_rect)
                        landmark_points = []

                        for i in range(68):  # 68 landmark points
                            point = landmarks.part(i)
                            landmark_points.append([point.x, point.y])

                        face_info['landmarks'] = landmark_points
                        face_info['num_landmarks'] = len(landmark_points)

                        # Compute face quality based on landmarks
                        face_info['landmark_quality'] = self._assess_landmark_quality(landmark_points)

                    except Exception as e:
                        logger.warning(f"Landmark detection failed: {e}")

                faces.append(face_info)

        except Exception as e:
            logger.warning(f"dlib face detection failed: {e}")

        return faces

    def detect_faces_dnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using OpenCV DNN

        Args:
            image: Input image

        Returns:
            List of face detection results
        """
        faces = []

        if not self.dnn_available:
            return faces

        try:
            h, w = image.shape[:2]

            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (300, 300), [104, 117, 123], False, False
            )

            # Set input to the network
            self.dnn_net.setInput(blob)

            # Run forward pass
            detections = self.dnn_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)

                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    bbox_w, bbox_h = x2 - x1, y2 - y1

                    # Filter out very small faces
                    if bbox_w >= self.min_face_size[0] and bbox_h >= self.min_face_size[1]:
                        face_info = {
                            'method': 'dnn',
                            'bbox': [x1, y1, bbox_w, bbox_h],
                            'confidence': float(confidence),
                            'center': [int(x1 + bbox_w//2), int(y1 + bbox_h//2)],
                            'area': int(bbox_w * bbox_h)
                        }

                        faces.append(face_info)

        except Exception as e:
            logger.warning(f"DNN face detection failed: {e}")

        return faces

    def detect_faces(self, image: np.ndarray, methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect faces using specified methods

        Args:
            image: Input image
            methods: Detection methods to use (if None, uses default)

        Returns:
            List of all face detections from all methods
        """
        if methods is None:
            methods = self.detection_methods

        all_faces = []

        for method in methods:
            try:
                if method == 'opencv' and self.opencv_available:
                    faces = self.detect_faces_opencv(image)
                    all_faces.extend(faces)

                elif method == 'mediapipe' and self.mediapipe_available:
                    faces = self.detect_faces_mediapipe(image)
                    all_faces.extend(faces)

                elif method == 'dlib' and self.dlib_available:
                    faces = self.detect_faces_dlib(image)
                    all_faces.extend(faces)

                elif method == 'dnn' and self.dnn_available:
                    faces = self.detect_faces_dnn(image)
                    all_faces.extend(faces)

            except Exception as e:
                logger.warning(f"Face detection method {method} failed: {e}")

        # Remove duplicate detections and merge overlapping ones
        merged_faces = self._merge_duplicate_faces(all_faces)

        # Sort by confidence and area
        merged_faces.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)

        return merged_faces

    def _merge_duplicate_faces(self, faces: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Merge duplicate face detections based on IoU

        Args:
            faces: List of face detections
            iou_threshold: IoU threshold for merging

        Returns:
            List of merged face detections
        """
        if not faces:
            return []

        # Calculate IoU between all pairs
        merged_faces = []
        used_indices = set()

        for i, face1 in enumerate(faces):
            if i in used_indices:
                continue

            # Find all faces that overlap with this one
            overlapping_faces = [face1]
            overlapping_indices = [i]

            for j, face2 in enumerate(faces[i+1:], i+1):
                if j in used_indices:
                    continue

                iou = self._calculate_iou(face1['bbox'], face2['bbox'])
                if iou > iou_threshold:
                    overlapping_faces.append(face2)
                    overlapping_indices.append(j)

            # Merge overlapping faces
            merged_face = self._merge_faces(overlapping_faces)
            merged_faces.append(merged_face)

            # Mark all overlapping faces as used
            used_indices.update(overlapping_indices)

        return merged_faces

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            bbox1: First bounding box [x, y, w, h]
            bbox2: Second bounding box [x, y, w, h]

        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection area
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def _merge_faces(self, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple face detections into one

        Args:
            faces: List of face detections to merge

        Returns:
            Merged face detection
        """
        if len(faces) == 1:
            return faces[0]

        # Use the face with highest confidence as base
        base_face = max(faces, key=lambda x: x['confidence'])

        # Average bounding boxes weighted by confidence
        total_confidence = sum(face['confidence'] for face in faces)

        avg_x = sum(face['bbox'][0] * face['confidence'] for face in faces) / total_confidence
        avg_y = sum(face['bbox'][1] * face['confidence'] for face in faces) / total_confidence
        avg_w = sum(face['bbox'][2] * face['confidence'] for face in faces) / total_confidence
        avg_h = sum(face['bbox'][3] * face['confidence'] for face in faces) / total_confidence

        merged_face = base_face.copy()
        merged_face.update({
            'bbox': [int(avg_x), int(avg_y), int(avg_w), int(avg_h)],
            'center': [int(avg_x + avg_w//2), int(avg_y + avg_h//2)],
            'area': int(avg_w * avg_h),
            'confidence': max(face['confidence'] for face in faces),
            'methods_used': list(set(face['method'] for face in faces)),
            'detection_count': len(faces)
        })

        return merged_face

    def _assess_landmark_quality(self, landmarks: List[List[int]]) -> float:
        """
        Assess quality of facial landmarks

        Args:
            landmarks: List of landmark points

        Returns:
            Quality score between 0 and 1
        """
        if not landmarks or len(landmarks) < 17:  # Minimum landmarks for jaw line
            return 0.0

        try:
            landmarks_array = np.array(landmarks)

            # Check landmark distribution (should be spread across face)
            x_coords = landmarks_array[:, 0]
            y_coords = landmarks_array[:, 1]

            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)

            # Normalize by face size
            face_size = max(x_range, y_range)
            if face_size == 0:
                return 0.0

            # Check symmetry (compare left and right sides)
            if len(landmarks) >= 68:  # Full 68-point model
                # Face outline points (jaw line)
                left_jaw = landmarks_array[0:8]  # Left jaw
                right_jaw = landmarks_array[9:17]  # Right jaw

                # Flip right jaw horizontally for comparison
                face_center_x = np.mean(x_coords)
                right_jaw_flipped = right_jaw.copy()
                right_jaw_flipped[:, 0] = 2 * face_center_x - right_jaw_flipped[:, 0]

                # Calculate symmetry score
                distances = np.linalg.norm(left_jaw - right_jaw_flipped, axis=1)
                avg_distance = np.mean(distances)
                symmetry_score = max(0, 1.0 - avg_distance / face_size)
            else:
                symmetry_score = 0.7  # Default for limited landmarks

            # Check landmark density
            density_score = min(1.0, len(landmarks) / 68.0)

            # Overall quality score
            quality_score = (symmetry_score * 0.6 + density_score * 0.4)

            return float(np.clip(quality_score, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Landmark quality assessment failed: {e}")
            return 0.5

    def align_face(self, image: np.ndarray, face_info: Dict[str, Any], 
                   target_size: Tuple[int, int] = (224, 224),
                   padding: float = 0.2) -> np.ndarray:
        """
        Align and crop face from image

        Args:
            image: Input image
            face_info: Face detection information
            target_size: Output face size
            padding: Padding around face (as fraction of face size)

        Returns:
            Aligned face image
        """
        try:
            x, y, w, h = face_info['bbox']

            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)

            # Calculate crop coordinates
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)

            # Crop face region
            face_crop = image[y1:y2, x1:x2]

            if face_crop.size == 0:
                # Return resized original if crop failed
                return cv2.resize(image, target_size)

            # Resize to target size
            aligned_face = cv2.resize(face_crop, target_size)

            # Advanced alignment using landmarks if available
            if 'landmarks' in face_info and len(face_info['landmarks']) >= 5:
                aligned_face = self._align_face_landmarks(
                    aligned_face, face_info['landmarks'], target_size
                )

            return aligned_face

        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            # Return resized original as fallback
            return cv2.resize(image, target_size)

    def _align_face_landmarks(self, face_image: np.ndarray, 
                             landmarks: List[List[int]], 
                             target_size: Tuple[int, int]) -> np.ndarray:
        """
        Align face using facial landmarks

        Args:
            face_image: Cropped face image
            landmarks: Facial landmark points
            target_size: Target output size

        Returns:
            Landmark-aligned face
        """
        try:
            if len(landmarks) < 5:
                return face_image

            landmarks_array = np.array(landmarks, dtype=np.float32)

            # Define target landmark positions (normalized)
            target_landmarks = np.array([
                [0.31556875000000000, 0.4615741071428571],  # Left eye
                [0.68262291666666670, 0.4615741071428571],  # Right eye
                [0.50026249999999990, 0.6405053571428571],  # Nose tip
                [0.34947187500000004, 0.8246919642857142],  # Left mouth
                [0.65343645833333330, 0.8246919642857142]   # Right mouth
            ], dtype=np.float32)

            # Scale target landmarks to face size
            target_landmarks *= np.array([target_size[0], target_size[1]])

            # Extract relevant landmarks (eyes, nose, mouth corners)
            if len(landmarks) >= 68:  # Full 68-point model
                src_landmarks = np.array([
                    landmarks[36],  # Left eye inner corner
                    landmarks[45],  # Right eye inner corner  
                    landmarks[30],  # Nose tip
                    landmarks[48],  # Left mouth corner
                    landmarks[54]   # Right mouth corner
                ], dtype=np.float32)
            else:
                # Use first 5 landmarks
                src_landmarks = landmarks_array[:5]

            # Compute affine transformation
            transform_matrix = cv2.getAffineTransform(
                src_landmarks[:3], target_landmarks[:3]
            )

            # Apply transformation
            aligned_face = cv2.warpAffine(
                face_image, transform_matrix, target_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

            return aligned_face

        except Exception as e:
            logger.warning(f"Landmark-based alignment failed: {e}")
            return face_image

    def get_face_quality_score(self, image: np.ndarray, face_info: Dict[str, Any]) -> float:
        """
        Compute overall face quality score

        Args:
            image: Input image
            face_info: Face detection information

        Returns:
            Quality score between 0 and 1
        """
        try:
            quality_factors = []

            # Size quality (larger faces are generally better)
            face_area = face_info['area']
            image_area = image.shape[0] * image.shape[1]
            size_ratio = face_area / image_area
            size_quality = min(1.0, size_ratio * 10)  # Normalize
            quality_factors.append(size_quality)

            # Confidence quality
            confidence = face_info.get('confidence', 0.5)
            quality_factors.append(confidence)

            # Landmark quality (if available)
            if 'landmark_quality' in face_info:
                quality_factors.append(face_info['landmark_quality'])

            # Eye detection quality (for OpenCV)
            if 'has_eyes' in face_info:
                eye_quality = 1.0 if face_info['has_eyes'] else 0.5
                quality_factors.append(eye_quality)

            # Aspect ratio quality (faces should be roughly rectangular)
            x, y, w, h = face_info['bbox']
            aspect_ratio = w / h if h > 0 else 1.0
            # Good face aspect ratio is around 0.75-1.25
            aspect_quality = max(0, 1.0 - abs(aspect_ratio - 1.0))
            quality_factors.append(aspect_quality)

            # Overall quality score
            return float(np.mean(quality_factors))

        except Exception as e:
            logger.warning(f"Face quality assessment failed: {e}")
            return 0.5

    def get_best_face(self, image: np.ndarray, methods: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best quality face from image

        Args:
            image: Input image
            methods: Detection methods to use

        Returns:
            Best face detection or None if no faces found
        """
        faces = self.detect_faces(image, methods)

        if not faces:
            return None

        # Score each face
        scored_faces = []
        for face in faces:
            quality_score = self.get_face_quality_score(image, face)
            face['quality_score'] = quality_score
            scored_faces.append(face)

        # Return face with highest quality score
        best_face = max(scored_faces, key=lambda x: x['quality_score'])
        return best_face
