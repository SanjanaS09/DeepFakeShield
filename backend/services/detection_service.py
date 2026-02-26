# """
# Detection Service
# Orchestrates preprocessing, feature extraction, and model inference
# with real-time feedback via callbacks
# """
# import os
# import logging
# import time
# from pathlib import Path
# from typing import Dict, Any, Optional, Callable

# import torch
# import numpy as np
# from PIL import Image

# from models.image_detector import ImageDeepfakeDetector
# from models.video_detector import VideoDetector
# # from models.audio_detector import AudioDetector
# from preprocessing.image_preprocessor import ImagePreprocessor
# from preprocessing.video_preprocessor import VideoPreprocessor
# from preprocessing.audio_preprocessor import AudioPreprocessor
# from features.visual_features import VisualFeatureExtractor
# from features.temporal_features import TemporalFeatureExtractor
# from features.audio_features import AudioFeatureExtractor
# from models.xai_explainer import XAIExplainer
# from models.audio_detector import AudioDetector

# # from models.audio_detection.inference import predict_audio


# logger = logging.getLogger(__name__)


# class DeepfakeDetectionService:
#     """Main detection service that coordinates all components"""
    
#     def __init__(self, device='cpu'):
#         """Initialize detection service with all models"""
#         self.device = device
#         logger.info(f"Initializing DeepfakeDetectionService on {device}")
        
#         # Model paths
#         self.IMAGE_MODEL_PATH = "checkpoints/image/best_model.pth"
#         self.VIDEO_MODEL_PATH = "checkpoints/video/best_model.pth"
#         self.AUDIO_MODEL_PATH = "checkpoints/audio/best_model.pt"
        
#         # Initialize as None (lazy loading)
#         self._image_detector = None
#         self._video_detector = None
#         self._audio_detector = None
#         self._xai_explainer = None
        
#         # Initialize preprocessors
#         self.image_preprocessor = ImagePreprocessor(target_size=(224, 224))
#         self.video_preprocessor = VideoPreprocessor(target_size=(224, 224), frames_per_clip=16)
#         self.audio_preprocessor = AudioPreprocessor(sample_rate=16000, duration=10)
        
#         # Initialize feature extractors
#         self.visual_feature_extractor = VisualFeatureExtractor()
#         self.temporal_feature_extractor = TemporalFeatureExtractor()
#         self.audio_feature_extractor = AudioFeatureExtractor()
        
#         logger.info("DeepfakeDetectionService initialized successfully")
    
#     @property
#     def image_detector(self):
#         """Lazy load image detector WITH TRAINED WEIGHTS"""
#         if self._image_detector is None:
#             logger.info("Loading image detector...")
            
#             if not Path(self.IMAGE_MODEL_PATH).exists():
#                 logger.error(f"Image model checkpoint not found: {self.IMAGE_MODEL_PATH}")
#                 raise FileNotFoundError(f"Image model not found: {self.IMAGE_MODEL_PATH}")
            
#             try:
#                 # Initialize ResNet18 detector with trained weights
#                 self._image_detector = ImageDeepfakeDetector(
#                     model_path='checkpoints/image/best_model.pth',
#                     device=self.device
#                 )
#                 logger.info("✓ Image detector loaded successfully!")
                
#             except Exception as e:
#                 logger.error(f"Error loading image model: {e}", exc_info=True)
#                 raise
        
#         return self._image_detector

    
#     @property
#     def video_detector(self):
#         """Lazy load video detector WITH TRAINED WEIGHTS"""
#         if self._video_detector is None:
#             logger.info("Loading video detector...")
            
#             if not Path(self.VIDEO_MODEL_PATH).exists():
#                 logger.warning(f"Video model checkpoint not found: {self.VIDEO_MODEL_PATH}")
#                 logger.warning("Using pretrained I3D weights instead")
#                 # Fallback to pretrained
#                 self._video_detector = VideoDetector(
#                     backbone='i3d',
#                     num_classes=2,
#                     device=self.device
#                 )
#             else:
#                 try:
#                     self._video_detector = VideoDetector(
#                         backbone='i3d',
#                         num_classes=2,
#                         device=self.device,
#                         pretrained=False
#                     )
                    
#                     checkpoint = torch.load(self.VIDEO_MODEL_PATH, map_location=self.device)
                    
#                     if 'model_state_dict' in checkpoint:
#                         self._video_detector.load_state_dict(checkpoint['model_state_dict'])
#                     elif 'state_dict' in checkpoint:
#                         self._video_detector.load_state_dict(checkpoint['state_dict'], strict=False)
#                     else:
#                         self._video_detector.load_state_dict(checkpoint)
                    
#                     self._video_detector.eval()
#                     logger.info("✅ Video detector loaded successfully!")
                    
#                 except Exception as e:
#                     logger.error(f"Error loading video model: {e}")
#                     # Fallback to pretrained
#                     self._video_detector = VideoDetector(
#                         backbone='i3d',
#                         num_classes=2,
#                         device=self.device
#                     )
        
#         return self._video_detector
    
#     @property
#     def audio_detector(self):
#         """Lazy load audio detector WITH TRAINED WEIGHTS"""
#         if self._audio_detector is None:
#             logger.info("Loading audio detector...")
            
#             if not Path(self.AUDIO_MODEL_PATH).exists():
#                 logger.warning(f"Audio model checkpoint not found: {self.AUDIO_MODEL_PATH}")
#                 logger.warning("Using pretrained ECAPA-TDNN weights instead")
#                 self._audio_detector = AudioDetector(
#                     backbone='ecapa-tdnn',
#                     num_classes=2,
#                     device=self.device
#                 )
#             else:
#                 try:
#                     self._audio_detector = AudioDetector(
#                         backbone='ecapa-tdnn',
#                         num_classes=2,
#                         device=self.device,
#                         pretrained=False
#                     )
                    
#                     checkpoint = torch.load(self.AUDIO_MODEL_PATH, map_location=self.device)
                    
#                     if 'model_state_dict' in checkpoint:
#                         self._audio_detector.load_state_dict(checkpoint['model_state_dict'])
#                     elif 'state_dict' in checkpoint:
#                         self._audio_detector.load_state_dict(checkpoint['state_dict'], strict=False)
#                     else:
#                         self._audio_detector.load_state_dict(checkpoint)
                    
#                     self._audio_detector.eval()
#                     logger.info("✅ Audio detector loaded successfully!")
                    
#                 except Exception as e:
#                     logger.error(f"Error loading audio model: {e}")
#                     self._audio_detector = AudioDetector(
#                         backbone='ecapa-tdnn',
#                         num_classes=2,
#                         device=self.device
#                     )
        
#         return self._audio_detector
    
#     @property
#     def xai_explainer(self):
#         """Lazy load XAI explainer"""
#         if self._xai_explainer is None:
#             self._xai_explainer = XAIExplainer(device=self.device)
#         return self._xai_explainer
    
#     def detect_image(
#         self,
#         image_path: str,
#         session_id: Optional[str] = None,
#         emit_callback: Optional[Callable] = None
#     ) -> Dict[str, Any]:
#         """
#         Detect deepfake in image with real-time preprocessing feedback
        
#         Args:
#             image_path: Path to image file
#             session_id: WebSocket session ID for real-time updates
#             emit_callback: Callback function for emitting processing steps
        
#         Returns:
#             Detection results dictionary
#         """
#         try:
#             start_time = time.time()
            
#             # Step 1: Load image
#             if emit_callback:
#                 emit_callback('Image Loading', 'Reading image file', session_id)
#             image = Image.open(image_path).convert('RGB')
            
#             # Step 2: Preprocessing
#             if emit_callback:
#                 emit_callback('Preprocessing', 'Resizing to 224x224, normalizing pixels', session_id)
            
#             preprocessed = self.image_preprocessor.preprocess_single(
#                 image_path,
#                 detect_faces=True,
#                 augment=False
#             )
#             image_tensor = preprocessed['processed_image']
            
#             if emit_callback:
#                 faces_detected = preprocessed.get('faces_detected', 0)
#                 emit_callback('Face Detection', f'Detected {faces_detected} face(s)', session_id)
            
#             # Step 3: Visual Feature Extraction
#             if emit_callback:
#                 emit_callback('Visual Feature Extraction', 
#                             'Extracting: Color distribution, Texture patterns, Edge features', 
#                             session_id)
            
#             visual_features = self.visual_feature_extractor.extract_features(
#                 image_tensor.unsqueeze(0).to(self.device)
#             )
            
#             # Step 4: Model Inference
#             if emit_callback:
#                 emit_callback('Model Inference', 'Running Xception CNN model', session_id)
            
#             with torch.no_grad():
#                 image_tensor_batch = image_tensor.unsqueeze(0).to(self.device)
#                 detect_result = self.image_detector.detect(image_path)
#                 predicted_class_str = detect_result['prediction']
#                 confidence = detect_result['confidence']
#                 # Convert class string to index for XAI (if needed)
#                 predicted_class_idx = 1 if predicted_class_str == 'FAKE' else 0
            
#             # Step 5: Feature Breakdown Analysis
#             if emit_callback:
#                 emit_callback('Post-Processing', 'Computing artifact scores', session_id)
            
#             feature_breakdown = self.image_detector.get_feature_breakdown(image_path)
            
#             # Step 6: XAI Visualization
#             if emit_callback:
#                 emit_callback('XAI Generation', 'Generating Grad-CAM heatmap', session_id)
            
#             xai_result = self.xai_explainer.explain_image(
#                 self.image_detector,
#                 image_tensor_batch,
#                 target_class=predicted_class_idx
#             )
#             processing_time = time.time() - start_time
            
#             return {
#                 'prediction': predicted_class_str,
#                 'confidence': float(confidence),
#                 'feature_breakdown': feature_breakdown,
#                 'xai_visualization': xai_result,
#                 'processing_time': time.time() - start_time,
#                 'file_size_mb': Path(image_path).stat().st_size / (1024 * 1024)
#             }


#         except Exception as e:
#             logger.error(f"Error in image detection: {str(e)}", exc_info=True)
#             raise
    
#     # def detect_video(
#     #     self,
#     #     video_path: str,
#     #     session_id: Optional[str] = None,
#     #     emit_callback: Optional[Callable] = None
#     # ) -> Dict[str, Any]:
#     #     """Detect deepfake in video with real-time feedback"""
#     #     try:
#     #         start_time = time.time()
            
#     #         # Step 1: Video Info
#     #         if emit_callback:
#     #             emit_callback('Video Loading', 'Reading video metadata', session_id)
            
#     #         video_info = self.video_preprocessor.load_video_info(video_path)
            
#     #         # Step 2: Frame Extraction
#     #         if emit_callback:
#     #             emit_callback('Frame Extraction', 
#     #                         f'Extracting {self.video_preprocessor.frames_per_clip} key frames', 
#     #                         session_id)
            
#     #         preprocessed = self.video_preprocessor.preprocess_single(
#     #             video_path,
#     #             detect_faces=True,
#     #             sampling_method='uniform'
#     #         )
            
#     #         frames_tensor = preprocessed['processed_frames']
            
#     #         # Step 3: Face Detection in Frames
#     #         if emit_callback:
#     #             faces_info = preprocessed.get('face_info', [])
#     #             total_faces = sum(len(f) for f in faces_info) if faces_info else 0
#     #             emit_callback('Face Detection', 
#     #                         f'Detected faces in {total_faces} frames', 
#     #                         session_id)
            
#     #         # Step 4: Temporal Feature Extraction
#     #         if emit_callback:
#     #             emit_callback('Temporal Feature Extraction', 
#     #                         'Extracting: Optical flow, Frame differences, Motion vectors', 
#     #                         session_id)
            
#     #         temporal_features = self.temporal_feature_extractor.extract_features(
#     #             frames_tensor.unsqueeze(0).to(self.device)
#     #         )
            
#     #         # Step 5: Model Inference
#     #         if emit_callback:
#     #             emit_callback('Model Inference', 'Running I3D video model', session_id)
            
#     #         with torch.no_grad():
#     #             frames_batch = frames_tensor.unsqueeze(0).to(self.device)
#     #             prediction_logits = self.video_detector(frames_batch)
#     #             prediction_probs = torch.softmax(prediction_logits, dim=1)
#     #             confidence, predicted_class = torch.max(prediction_probs, 1)
            
#     #         # Step 6: Post-processing
#     #         if emit_callback:
#     #             emit_callback('Post-Processing', 'Analyzing temporal consistency', session_id)
            
#     #         processing_time = time.time() - start_time
            
#     #         return {
#     #             'prediction': 'FAKE' if predicted_class.item() == 1 else 'REAL',
#     #             'confidence': float(confidence.item()),
#     #             'feature_breakdown': temporal_features,
#     #             'temporal_analysis': {
#     #                 'frames_analyzed': preprocessed['frames_sampled'],
#     #                 'sampling_method': preprocessed['sampling_method']
#     #             },
#     #             'processing_time': processing_time,
#     #             'frames_analyzed': preprocessed['frames_sampled'],
#     #             'fps': video_info.get('fps', 0)
#     #         }
            
#     #     except Exception as e:
#     #         logger.error(f"Error in video detection: {str(e)}", exc_info=True)
#     #         raise


    
#     # def detect_audio(
#     #     self,
#     #     audio_path: str,
#     #     session_id: Optional[str] = None,
#     #     emit_callback: Optional[Callable] = None
#     # ) -> Dict[str, Any]:
#     #      """Detect deepfake in audio with real-time feedback"""
#     #     try:
#     #         start_time = time.time()
            
#     #         # Step 1: Load Audio
#     #         if emit_callback:
#     #             emit_callback('Audio Loading', 'Loading audio at 16kHz sample rate', session_id)
            
#     #         # Step 2: Preprocessing
#     #         if emit_callback:
#     #             emit_callback('Audio Preprocessing', 'Resampling and normalizing audio', session_id)
            
#     #         preprocessed = self.audio_preprocessor.preprocess_single(audio_path)
#     #         audio_tensor = preprocessed['processed_audio']
            
#     #         # Step 3: Spectral Analysis
#     #         if emit_callback:
#     #             emit_callback('Spectral Analysis', 'Computing MFCC coefficients', session_id)
            
#     #         # Step 4: Audio Feature Extraction
#     #         if emit_callback:
#     #             emit_callback('Audio Feature Extraction', 
#     #                         'Extracting: MFCC, Spectral centroid, Zero-crossing rate', 
#     #                         session_id)
            
#     #         audio_features = self.audio_feature_extractor.extract_features(
#     #             audio_tensor.unsqueeze(0).to(self.device)
#     #         )
            
#     #         # Step 5: Model Inference
#     #         if emit_callback:
#     #             emit_callback('Model Inference', 'Running ECAPA-TDNN audio model', session_id)
            
#     #         with torch.no_grad():
#     #             audio_batch = audio_tensor.unsqueeze(0).to(self.device)
#     #             prediction_logits = self.audio_detector(audio_batch)
#     #             prediction_probs = torch.softmax(prediction_logits, dim=1)
#     #             confidence, predicted_class = torch.max(prediction_probs, 1)
            
#     #         processing_time = time.time() - start_time
            
#     # return {
#     #             'prediction': 'FAKE' if predicted_class.item() == 1 else 'REAL',
#     #             'confidence': float(confidence.item()),
#     #             'feature_breakdown': audio_features,
#     #             'processing_time': processing_time,
#     #             'duration': preprocessed.get('duration', 0),
#     #             'sample_rate': preprocessed.get('sample_rate', 16000)
#     #         }
            
#     # except Exception as e:
#     #         logger.error(f"Error in audio detection: {str(e)}", exc_info=True)
#     #         raise

#     def detect_video(self, video_path: str, emit_callback=None):
#         logger.info(f"Processing video: {video_path}")

#         if self.video_model is None:
#             logger.error("❌ Video model not available")
#             return {
#                 "prediction": "UNKNOWN",
#                 "confidence": 0.0,
#                 "probabilities": {"REAL": 0.5, "FAKE": 0.5},
#                 "frames_analyzed": 0,
#                 "status": "error",
#                 "error": "Video model not loaded"
#             }

#         try:
#             result = self.video_model.predict(video_path)

#             return {
#                 "prediction": result.get("prediction", "UNKNOWN"),
#                 "confidence": float(result.get("confidence", 0.0)),
#                 "probabilities": result.get(
#                     "probabilities", {"REAL": 0.5, "FAKE": 0.5}
#                 ),
#                 "frames_analyzed": result.get("frames_analyzed", 0),
#                 "status": "success"
#             }

#         except Exception as e:
#             logger.error(f"Video detection failed: {e}", exc_info=True)
#             return {
#                 "prediction": "ERROR",
#                 "confidence": 0.0,
#                 "probabilities": {"REAL": 0.5, "FAKE": 0.5},
#                 "frames_analyzed": 0,
#                 "status": "error",
#                 "error": str(e)
#             }


#     def detect_audio(
#         self,
#         audio_path: str,
#         session_id: Optional[str] = None,
#         emit_callback: Optional[Callable] = None
#     ) -> Dict[str, Any]:
#         """Detect deepfake in audio"""

#         try:
#             start_time = time.time()

#             if emit_callback:
#                 emit_callback('Audio Loading', 'Loading and preprocessing audio', session_id)

#             # Preprocess audio
#             preprocessed = self.audio_preprocessor.preprocess_single(audio_path)
#             audio_tensor = preprocessed['processed_audio'].to(self.device)

#             if emit_callback:
#                 emit_callback('Audio Feature Extraction', 'Extracting audio features', session_id)

#             audio_features = self.audio_feature_extractor.extract_features(
#                 audio_tensor.unsqueeze(0)
#             )

#             if emit_callback:
#                 emit_callback('Model Inference', 'Running audio deepfake detector', session_id)

#             # ✅ SINGLE audio model inference
#             result = self.audio_detector.predict(audio_features)

#             processing_time = time.time() - start_time

#             return {
#                 'prediction': result['prediction'],
#                 'confidence': result['confidence'],
#                 'probabilities': result['probabilities'],
#                 'feature_breakdown': audio_features,
#                 'processing_time': processing_time,
#                 'duration': preprocessed.get('duration', 0),
#                 'sample_rate': preprocessed.get('sample_rate', 16000)
#             }

#         except Exception as e:
#             logger.error(f"Error in audio detection: {str(e)}", exc_info=True)
#             raise
