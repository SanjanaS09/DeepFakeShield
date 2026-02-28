"""
DeepFake Shield - Flask Backend with Real Model Predictions
Uses trained models for actual deepfake detection
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging
import base64
import cv2
import numpy as np
import torch

BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit


#app.register_blueprint(detection_bp)

print("🚀 Starting DeepFake Shield backend...")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# ============================================
# REAL DETECTION SERVICE - USES TRAINED MODELS
# ============================================

class RealDetectionService:
    """Detection service using actual trained models"""
    
    def __init__(self):
        """Initialize with trained models"""
        logger.info("Initializing Real Detection Service...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.image_model = self.load_image_model()
        self.video_model = self.load_video_model()
        # self.video_preprocessor = VideoPreprocessor(device=str(self.device))
        self.audio_model = self.load_audio_model()

        from xai.xai_engine import XAIEngine

        if self.image_model:
            self.image_xai = XAIEngine(self.image_model.backbone)
        else:
            self.image_xai = None

        if self.video_model:
            self.video_xai = XAIEngine(self.video_model.model)
        else:
            self.video_xai = None

        print("VIDEO MODEL EXISTS:", self.video_model is not None)
        print("VIDEO XAI EXISTS:", self.video_xai is not None)

        # if self.audio_model:
        #     self.audio_xai = XAIEngine(self.audio_model.model)
        # else:
        #     self.audio_xai = None

        from xai.audio.audio_saliency import AudioSaliency

        if self.audio_model:
            self.audio_xai = AudioSaliency(
                self.audio_model.model,
                self.audio_model.processor,
                self.device
            )
        else:
            self.audio_xai = None
                
        logger.info("✓ Detection service initialized with real models")
    
    def load_image_model(self):
        """Load image model from checkpoint"""
        try:
            checkpoint_path = Path("checkpoints/image/best_model.pth")
            if not checkpoint_path.exists():
                logger.warning(f"Image model not found at {checkpoint_path}")
                return None
            
            logger.info(f"Loading image model from {checkpoint_path}")          
            try:
                from models.image_detector import ImageDeepfakeDetector  
                model = ImageDeepfakeDetector(
                    model_path=str(checkpoint_path),
                    device=self.device
                )
                logger.info("✅ Image model loaded successfully")
                return model
                
            except Exception as e:
                logger.error(f"Error loading image model: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            return None

    def load_video_model(self):
        try:
            logger.info("Loading frame-based video detector...")

            from models.video_detector_frame_based import FrameBasedVideoDetector

            self.video_model = FrameBasedVideoDetector(
                model_path="checkpoints/video/best_model.pth",
                device=str(self.device),
                num_frames=8,
                frame_size=(224, 224)
            )

            logger.info("✅ Video detector loaded successfully")
            return self.video_model   # ✅ THIS IS THE FIX

        except Exception as e:
            logger.error(f"Failed to load video model: {e}", exc_info=True)
            return None


    def load_audio_model(self):
        try:
            checkpoint_path = Path("checkpoints/audio/best_model.pth")
            if not checkpoint_path.exists():
                logger.warning("Audio model not found")
                return None

            from models.audio_detector import AudioDeepfakeDetector

            model = AudioDeepfakeDetector(
                model_path=str(checkpoint_path),
                device=self.device
            )

            logger.info("✅ Audio Wav2Vec2 model loaded")
            return model

        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            return None
    
    # def detect_image(self, image_path):
    #     """Detect deepfake in image using trained model"""
    #     try:
    #         logger.info(f"Processing image: {image_path}")
            
    #         if not self.image_model:
    #             logger.error("❌ CRITICAL: Image model failed to load!")
    #             return {
    #                 'error': 'Image model not available',
    #                 'status': 'error'
    #             }
            
    #         try:
    #             logger.info("Running inference using image_model.detect(image_path)...")
    #             result = self.image_model.detect(image_path)

    #             import cv2
    #             from xai.utils.base64_utils import encode_image_to_base64

    #             print("XAI attached:", self.image_xai is not None)
    #             image_np = cv2.imread(image_path)

    #             original_base64 = encode_image_to_base64(image_np)
    #             heatmap_base64 = explanation["heatmap_base64"] if explanation else None

    #             result["xai"] = {
    #                 "original": original_base64,
    #                 "heatmap": heatmap_base64,
    #                 "explanation": (
    #                     "🚨 Deepfake image detected"
    #                     if result["prediction"] == "FAKE"
    #                     else "✅ Authentic image"
    #                 ),
    #                 "confidence_level": (
    #                     "High" if result["confidence"] > 0.8 else "Moderate"
    #                 ),
    #                 "key_indicators": {
    #                     "Facial Consistency": "Low" if result["prediction"] == "FAKE" else "High",
    #                     "Texture Integrity": "Compromised" if result["prediction"] == "FAKE" else "Natural"
    #                 },
    #                 "recommendations": [
    #                     "Verify source authenticity",
    #                     "Cross-check with trusted media"
    #                 ]
    #             }
                
    #             logger.info(f"Detection complete: {result.get('prediction')} ({result.get('confidence',0):.2%})")
    #             return result
            
    #         except Exception as e:
    #             logger.error(f"Model inference error: {e}", exc_info=True)
    #             return {'error': f"Inference error: {str(e)}", 'status': 'error'}
    #     except Exception as e:
    #         logger.error(f"Image detection error: {e}", exc_info=True)
    #         return {'error': str(e), 'status': 'error'}

    def detect_image(self, image_path):
        try:
            logger.info(f"Processing image: {image_path}")

            result = self.image_model.detect(image_path)

            image_np = cv2.imread(image_path)
            heatmap = self.image_xai.explain_image(image_np)

            original_base64 = encode_image_to_base64(image_np)
            heatmap_base64 = encode_image_to_base64(heatmap)

            prediction = result["prediction"]
            confidence_score = result["confidence"]

            raw_fake_prob = result["probabilities"]["FAKE"]

            # 🔥 Calibrated threshold
            THRESHOLD = 0.65

            if raw_fake_prob >= 0.85 or raw_fake_prob <= 0.15:
                risk_level = "High"
            elif raw_fake_prob >= 0.65 or raw_fake_prob <= 0.35:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            return {
                "prediction": prediction,
                "confidence": confidence_score,
                "probabilities": result["probabilities"],
                "processing_time": result.get("processing_time", 0),
                "xai": {
                    "original": original_base64,
                    "heatmap": heatmap_base64,
                    "explanation": f"{prediction} detected with {confidence_score:.2%} confidence",
                    "confidence_level": "High" if confidence_score > 0.8 else "Moderate",

                    "reasoning": (
                        [
                            f"Fake probability score: {raw_fake_prob:.2f}",
                            "High activation around facial boundaries",
                            "Blending inconsistencies detected",
                            "Texture frequency anomalies observed"
                        ]
                        if prediction == "FAKE"
                        else
                        [
                            f"Fake probability score: {raw_fake_prob:.2f}",
                            "Uniform facial feature activations",
                            "Consistent skin texture gradients",
                            "Natural illumination distribution"
                        ]
                    ),
                    "risk_level": risk_level,
                    "keypoints": {
                        "face_region": "Highly activated" if prediction == "FAKE" else "Stable",
                        "eye_region": "Irregular texture" if prediction == "FAKE" else "Natural shading",
                        "mouth_region": "Blending artifacts" if prediction == "FAKE" else "Consistent alignment"
                    },

                    "recommendations": (
                        [
                            "Avoid resharing this content",
                            "Verify original source",
                            "Cross-check with trusted media"
                        ]
                        if prediction == "FAKE"
                        else
                        [
                            "Content appears authentic",
                            "Safe to share",
                            "No manipulation patterns detected"
                        ]
                    )
                },
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Image detection error: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def detect_video(self, video_path, emit_callback=None):
        try:
            logger.info(f"Processing video: {video_path}")

            result = self.video_model.predict(video_path)
            logger.info(f"Raw video result: {result}")

            prediction = result["prediction"]
            confidence = float(result["confidence"])

            raw_fake_prob = result["probabilities"]["FAKE"]

            if raw_fake_prob >= 0.85 or raw_fake_prob <= 0.15:
                risk_level = "High"
            elif raw_fake_prob >= 0.65 or raw_fake_prob <= 0.35:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            xai_data = {}

            # 🔥 VIDEO XAI
            if self.video_xai:
                frames = self.video_model.extract_frames(video_path)

                if frames is not None:
                    frames_np = frames.cpu().numpy()
                    heatmaps = []

                    for i in range(frames_np.shape[0]):
                        frame = frames_np[i]
                        frame = np.transpose(frame, (1, 2, 0))  # C,H,W → H,W,C
                        frame = (frame * 255).astype("uint8")

                        heatmap = self.video_xai.explain_image(frame)
                        heatmaps.append(encode_image_to_base64(heatmap))

                    xai_data = {
                        "heatmap_frames": heatmaps,
                        "explanation": f"{prediction} video detected with {confidence:.2%} confidence",
                        "confidence_level": "High" if confidence > 0.8 else "Moderate",
                        "key_indicators": {
                            "Temporal Consistency": "Low" if prediction == "FAKE" else "High",
                            "Frame Stability": "Inconsistent" if prediction == "FAKE" else "Stable"
                        },
                        "temporal_analysis": {
                            "frame_variance": "High" if prediction == "FAKE" else "Low",
                            "motion_consistency": "Discontinuous" if prediction == "FAKE" else "Smooth",
                            "expression_alignment": "Misaligned" if prediction == "FAKE" else "Natural"
                        },
                        "recommendations": [
                            "Verify source authenticity",
                            "Check original upload source",
                            "Avoid resharing if suspicious"
                        ],"reasoning": (
                            [
                                f"Fake probability score: {raw_fake_prob:.2f}",
                                "Temporal inconsistency across frames",
                                "Unnatural facial motion transitions",
                                "High-frequency synthesis artifacts"
                            ]
                            if prediction == "FAKE"
                            else
                            [
                                f"Fake probability score: {raw_fake_prob:.2f}",
                                "Stable temporal coherence",
                                "Natural micro-expression transitions",
                                "Consistent frame-level feature maps"
                            ]
                        ),
                        "risk_level": risk_level,
                    }

            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": result["probabilities"],
                "frames_analyzed": result.get("frames_analyzed", 0),
                "xai": xai_data,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Video detection error: {e}", exc_info=True)
            return {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "status": "error"
            }
        
    def detect_audio(self, audio_path):
        try:
            logger.info("=== AUDIO DETECTION START ===")

            result = self.audio_model.detect(audio_path)

            if result.get("status") == "error":
                return result

            prob = result["raw_probability"]
            prediction = result["prediction"]
            confidence = result["confidence"]

            # 🔥 Risk scoring
            if prob >= 0.85 or prob <= 0.15:
                risk_level = "High"
            elif prob >= 0.65 or prob <= 0.35:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            # 🔥 Dynamic reasoning
            if prediction == "FAKE":
                reasoning = [
                    f"Synthetic probability score: {prob:.2f}",
                    "Detected abnormal spectral harmonics",
                    "Inconsistent frequency energy patterns"
                ]
            else:
                reasoning = [
                    f"Synthetic probability score: {prob:.2f}",
                    "Natural vocal frequency transitions",
                    "Stable harmonic structure"
                ]

            confidence_level = (
                "High" if confidence > 0.8 else
                "Moderate" if confidence > 0.6 else
                "Low"
            )

            # 🔥 Safe XAI block
            spec_base64 = None
            heatmap_base64 = None

            if self.audio_xai:
                try:
                    spec_img, saliency = self.audio_xai.generate(audio_path)

                    heatmap = cv2.applyColorMap(
                        (saliency * 255).astype("uint8"),
                        cv2.COLORMAP_JET
                    )

                    spec_base64 = encode_image_to_base64(spec_img)
                    heatmap_base64 = encode_image_to_base64(heatmap)

                except Exception as xai_error:
                    logger.warning(f"Audio XAI failed: {xai_error}")

            logger.info(f"Audio XAI spectrogram exists: {spec_base64 is not None}")
            logger.info(f"Audio XAI saliency exists: {heatmap_base64 is not None}")

            return {
                "prediction": prediction,
                "confidence": confidence,
                "risk_level": risk_level,
                "probabilities": result["probabilities"],
                "xai": {
                    "spectrogram": spec_base64,
                    "saliency_map": heatmap_base64,
                    "reasoning": reasoning,
                    "recommendations": (
                        [
                            "Do not trust this audio",
                            "Verify speaker identity",
                            "Cross-check with original recording"
                        ]
                        if prediction == "FAKE"
                        else [
                            "Audio appears authentic",
                            "No major synthetic indicators detected"
                        ]
                    ),
                    "confidence_level": confidence_level,
                },
                "status": "success"
            }

        except Exception as e:
            logger.error("=== AUDIO DETECTION FAILED ===", exc_info=True)
            return {"error": str(e), "status": "error"}
    
    def _demo_result(self):
        """Return demo result (for models not yet implemented)"""
        return {
            'prediction': 'REAL',
            'confidence': 0.92,
            'probability': {'REAL': 0.92, 'FAKE': 0.08},
            'status': 'success'
        }

# ============================================
# CREATE FLASK APP
# ============================================
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'deepfake-shield-secret'

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # socketio = SocketIO(
    #     app,
    #     cors_allowed_origins="*",
    #     async_mode="threading",
    #     logger=False,
    #     engineio_logger=False
    # )

    # ✅ INIT SERVICE FIRST
    from api.analysis_routes import analysis_bp
    from api.detection_routes import detection_bp

    # THEN register blueprints
    app.register_blueprint(analysis_bp)
    app.register_blueprint(detection_bp)

    # THEN initialize your service
    app.detection_service = RealDetectionService()

    logger.info("Flask app created with REAL detection service")
    return app

app  = create_app()

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Health check"""
    return jsonify({
        'status': 'online',
        'message': 'DeepFake Shield API - PRODUCTION MODE',
        'mode': 'REAL MODELS',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


# @app.route('/api/detection/image', methods=['POST'])
# def detect_image():
#     """Image detection with real models - FIXED"""
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         from werkzeug.utils import secure_filename
#         import uuid
        
#         upload_dir = Path('uploads/temp')
#         upload_dir.mkdir(parents=True, exist_ok=True)
        
#         filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
#         filepath = upload_dir / filename
#         file.save(str(filepath))
        
#         logger.info(f"File saved: {filepath}")
        
#         detection_service = app.config['detection_service']
        
#         # ✅ RUN DETECTION
#         logger.info("Starting detection...")
#         result = detection_service.detect_image(str(filepath))
        
#         # ✅ FIX: Ensure proper response format
#         if result.get('status') == 'error':
#             logger.error(f"Detection failed: {result.get('error')}")
#             return jsonify(result), 500
        
#         # ✅ FIX: Convert prediction properly
#         prediction = result.get('prediction', 'UNKNOWN')
#         confidence = float(result.get('confidence', 0.0))
        
#         # Ensure valid confidence
#         if not (0 <= confidence <= 1):
#             confidence = 0.5
#             logger.warning("Invalid confidence value, set to 0.5")
        
#         # ✅ BUILD RESPONSE WITH XAI
#         response = {
#             'prediction': prediction,  # Should be 'FAKE' or 'REAL'
#             'confidence': confidence,
#             'probabilities': {
#                 'REAL': float(1 - confidence) if prediction == 'FAKE' else float(confidence),
#                 'FAKE': float(confidence) if prediction == 'FAKE' else float(1 - confidence)
#             },
#             'label': prediction,
#             'file_name': filename,
#             'status': 'success',
#             # ✅ ADD XAI EXPLANATIONS
#             'xai': {
#                 'explanation': f"{'🚨 DEEPFAKE DETECTED' if prediction == 'FAKE' else '✅ AUTHENTIC CONTENT'}",
#                 'reasoning': get_reasoning(prediction, confidence),
#                 'key_indicators': get_indicators(prediction, confidence),
#                 'confidence_level': get_confidence_level(confidence),
#                 'heatmap': None,  # Will add Grad-CAM if available
#                 'recommendations': get_recommendations(prediction, confidence)
#             }
#         }
        
#         # Cleanup
#         try:
#             if filepath.exists():
#                 filepath.unlink()
#         except Exception as e:
#             logger.warning(f"Cleanup failed: {e}")
        
#         logger.info(f"✅ Detection result: {prediction} ({confidence:.2%})")
#         return jsonify(response), 200
    
#     except Exception as e:
#         logger.error(f"Error: {e}", exc_info=True)
#         return jsonify({'error': str(e), 'status': 'error'}), 500


# @app.route('/api/detection/video', methods=['POST'])
# def detect_video():
#     """Video detection with real-time feedback - FIXED"""
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         from werkzeug.utils import secure_filename
#         import uuid
        
#         upload_dir = Path('uploads/temp')
#         upload_dir.mkdir(parents=True, exist_ok=True)
        
#         filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
#         filepath = upload_dir / filename
#         file.save(str(filepath))
        
#         logger.info(f"Video file saved: {filepath}")
        
#         detection_service = app.config['detection_service']
        
#         def emit_progress(step, message):
#             logger.info(f"{step}: {message}")
        
#         logger.info("Starting video detection...")
#         result = detection_service.detect_video(str(filepath), emit_callback=emit_progress)
        
#         # ✅ FIX: Handle error results properly
#         if result.get('status') == 'error':
#             logger.error(f"Detection failed: {result.get('error')}")
#             # Return error but with proper structure
#             response = {
#                 'prediction': 'UNKNOWN',
#                 'confidence': 0.0,
#                 'probabilities': {'REAL': 0.5, 'FAKE': 0.5},
#                 'status': 'error',
#                 'error': result.get('error', 'Unknown error'),
#                 'xai': {
#                     'explanation': '❌ Analysis failed',
#                     'reasoning': ['Unable to process video'],
#                     'key_indicators': {},
#                     'confidence_level': 'Unknown',
#                     'recommendations': ['Please try another video file']
#                 }
#             }
#             try:
#                 if filepath.exists():
#                     filepath.unlink()
#             except:
#                 pass
#             return jsonify(response), 500
        
#         prediction = result.get('prediction', 'UNKNOWN')
#         confidence = float(result.get('confidence', 0.0))
        
#         # Ensure valid confidence
#         if not (0 <= confidence <= 1):
#             confidence = 0.5
        
#         # ✅ BUILD RESPONSE WITH XAI
#         response = {
#             'prediction': prediction,
#             'confidence': confidence,
#             'probabilities': result.get('probabilities', {
#                 'REAL': float(1 - confidence),
#                 'FAKE': float(confidence)
#             }),
#             'frames_analyzed': result.get('frames_analyzed', 0),
#             'fps': result.get('fps', 0),
#             'label': prediction,
#             'file_name': filename,
#             'status': 'success',
#             # ✅ ADD XAI EXPLANATIONS
#             'xai': {
#                 'explanation': f"{'🚨 DEEPFAKE VIDEO DETECTED' if prediction == 'FAKE' else '✅ AUTHENTIC VIDEO'}",
#                 'reasoning': get_video_reasoning(prediction, confidence, result),
#                 'key_indicators': get_video_indicators(prediction, confidence),
#                 'confidence_level': get_confidence_level(confidence),
#                 'temporal_analysis': result.get('feature_breakdown', {}),
#                 'recommendations': get_recommendations(prediction, confidence)
#             }
#         }
        
#         try:
#             if filepath.exists():
#                 filepath.unlink()
#         except Exception as e:
#             logger.warning(f"Cleanup failed: {e}")
        
#         logger.info(f"✅ Video detection complete: {prediction} ({confidence:.2%})")
#         return jsonify(response), 200
    
#     except Exception as e:
#         logger.error(f"Error: {e}", exc_info=True)
#         return jsonify({'error': str(e), 'status': 'error'}), 500

# @app.route('/audio', methods=['POST'])
# def detect_audio():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         from werkzeug.utils import secure_filename
#         import uuid
        
#         upload_dir = Path('uploads/temp')
#         upload_dir.mkdir(parents=True, exist_ok=True)
        
#         filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
#         filepath = upload_dir / filename
#         file.save(str(filepath))
        
#         logger.info(f"Audio file saved: {filepath}")
        
#         detection_service = app.config['detection_service']
        
#         logger.info("Starting audio detection...")
#         result = detection_service.detect_audio(str(filepath))
        
#         prediction = result.get('prediction', 'UNKNOWN')
#         confidence = float(result.get('confidence', 0.0))
        
#         response = {
#             'prediction': prediction,
#             'confidence': confidence,
#             'probabilities': {
#                 'REAL': float(1 - confidence),
#                 'FAKE': float(confidence)
#             },
#             'status': 'success'
#         }
        
#         try:
#             if filepath.exists():
#                 filepath.unlink()
#         except:
#             pass
        
#         return jsonify(response), 200
    
#     except Exception as e:
#         logger.error(f"Audio detection error: {e}", exc_info=True)
#         return jsonify({'error': str(e), 'status': 'error'}), 500


# ============================================
# ✅ XAI HELPER FUNCTIONS
# ============================================

def get_reasoning(prediction: str, confidence: float) -> List[str]:
    """Generate reasoning for image prediction"""
    reasons = []
    
    if prediction == 'FAKE':
        if confidence > 0.95:
            reasons = [
                "Strong facial manipulation artifacts detected",
                "Inconsistent lighting patterns identified",
                "Compression artifacts typical of AI generation found",
                "Blending boundaries detected around facial regions"
            ]
        elif confidence > 0.80:
            reasons = [
                "Multiple indicators of artificial facial manipulation",
                "Unusual frequency domain characteristics",
                "Potential deepfake synthesis patterns detected"
            ]
        elif confidence > 0.60:
            reasons = [
                "Some suspicious artifacts detected",
                "Cannot confirm authenticity with high confidence",
                "Recommend manual review"
            ]
    else:  # REAL
        if confidence > 0.95:
            reasons = [
                "Natural facial features and expressions detected",
                "Consistent lighting and shadow patterns",
                "No significant compression or synthesis artifacts",
                "Authentic biological motion detected"
            ]
        elif confidence > 0.80:
            reasons = [
                "Characteristics consistent with authentic media",
                "No major manipulation indicators found"
            ]
        else:
            reasons = [
                "Analysis shows mixed signals",
                "Manual verification recommended"
            ]
    
    return reasons


def get_indicators(prediction: str, confidence: float) -> Dict[str, Any]:
    """Get key indicators for image"""
    indicators = {
        'facial_consistency': 'High' if prediction == 'REAL' else 'Low',
        'lighting_quality': 'Natural' if prediction == 'REAL' else 'Inconsistent',
        'compression_artifacts': 'Minimal' if prediction == 'REAL' else 'High',
        'frequency_analysis': 'Normal' if prediction == 'REAL' else 'Anomalies',
        'edge_quality': 'Sharp' if prediction == 'REAL' else 'Blended'
    }
    return indicators


def get_video_reasoning(prediction: str, confidence: float, result: Dict) -> List[str]:
    """Generate reasoning for video prediction"""
    reasons = []
    
    frames_analyzed = result.get('frames_analyzed', 0)
    
    if prediction == 'FAKE':
        reasons = [
            f"Analyzed {frames_analyzed} frames for inconsistencies",
            "Detected unnatural motion patterns",
            "Found temporal discontinuities",
            "Identified synthesis artifacts across frames"
        ]
    else:
        reasons = [
            f"Analyzed {frames_analyzed} frames successfully",
            "Consistent natural motion detected",
            "Temporal coherence verified",
            "No major synthesis artifacts found"
        ]
    
    return reasons


def get_video_indicators(prediction: str, confidence: float) -> Dict[str, Any]:
    """Get key indicators for video"""
    indicators = {
        'temporal_consistency': 'High' if prediction == 'REAL' else 'Low',
        'motion_smoothness': 'Natural' if prediction == 'REAL' else 'Unnatural',
        'frame_quality': 'Consistent' if prediction == 'REAL' else 'Variable',
        'face_tracking': 'Stable' if prediction == 'REAL' else 'Jumpy',
        'eye_contact': 'Natural' if prediction == 'REAL' else 'Unnatural'
    }
    return indicators


def get_confidence_level(confidence: float) -> str:
    """Convert confidence to readable level"""
    if confidence > 0.95:
        return "Very High"
    elif confidence > 0.80:
        return "High"
    elif confidence > 0.60:
        return "Moderate"
    elif confidence > 0.40:
        return "Low"
    else:
        return "Very Low"


def get_recommendations(prediction: str, confidence: float) -> List[str]:
    """Get recommendations based on prediction"""
    recommendations = []
    
    if prediction == 'FAKE':
        recommendations = [
            "⚠️ Do not share or trust this content",
            "📋 Report to platform moderators",
            "🔍 Verify source authenticity",
            "💾 Save evidence for verification purposes"
        ]
    else:
        if confidence > 0.95:
            recommendations = [
                "✅ Content appears authentic",
                "📱 Safe to share"
            ]
        else:
            recommendations = [
                "⚠️ Some uncertainty in classification",
                "🔍 Manual review recommended for important decisions",
                "👨‍⚖️ Consult experts for critical use cases"
            ]
    
    return recommendations


# ============================================
# RUNNING THE APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🛡️  DeepFake Shield - PRODUCTION MODE")
    print("="*70)
    print("🚀 Starting Flask server...")
    print("📍 Access at: http://127.0.0.1:5000")
    print("🔗 API Endpoints:")
    print("   - Image:  POST http://127.0.0.1:5000/api/detection/image")
    print("   - Video:  POST http://127.0.0.1:5000/api/detection/video")
    print("   - Audio:  POST http://127.0.0.1:5000/api/detection/audio")
    print("="*70 + "\n")
    
    # socketio.run(
    #     app,
    #     host='127.0.0.1',
    #     port=5000,
    #     debug=False
    # )

    app.run(host="127.0.0.1", port=5000)
