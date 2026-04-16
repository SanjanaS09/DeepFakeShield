"""
Detection Routes with WebSocket Integration
Sends real-time preprocessing and feature extraction updates to frontend
"""
import os
import logging
import uuid
import time
from pathlib import Path
from datetime import datetime

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

# from services import detection_service
# from services.detection_service import DeepfakeDetectionService
from utils.validators import validate_file_type, validate_file_size
from utils.file_handlers import save_uploaded_file, cleanup_temp_file

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__,url_prefix='/api/detection')


def emit_processing_step(step_name, details, session_id=None):
    try:
        from flask_socketio import emit
        if session_id:
            emit(
                'processing_step',
                {
                    'name': step_name,
                    'details': details,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'completed'
                },
                room=session_id,
                namespace='/'
            )
    except Exception:
        pass  # Ignore when not in socket context

@detection_bp.route('/image', methods=['POST'])
def detect_image():
    """
    Image deepfake detection endpoint with real-time preprocessing feedback
    """
    start_time = time.time()
    temp_file = None
    
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files.get("image") or request.files.get("file")
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get session ID for WebSocket
        session_id = request.form.get('session_id', None)
        
        # Validate file type
        if not validate_file_type(file.filename, 'image'):
            return jsonify({"error": "Invalid image format"}), 400
        
        emit_processing_step('File Upload', f'Uploaded {file.filename}', session_id)
        
        # Save file
        temp_file = save_uploaded_file(file, 'image')
        logger.info(f"Saved image to {temp_file}")
        
        emit_processing_step('File Saved', f'Saved to temporary location', session_id)
        
        # Get detection service
        detection_service = current_app.detection_service
        
        # Preprocess with real-time updates
        emit_processing_step('Preprocessing Started', 'Initializing image preprocessor', session_id)
        
        # Detect and process
        emit_processing_step('Preprocessing Started', 'Running image model', session_id)

        result = detection_service.detect_image(str(temp_file))
        
        emit_processing_step('Detection Complete', 'Analysis finished', session_id)
        
        # Prepare response
        processing_time = time.time() - start_time
        response = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "processing_time": processing_time,
            "feature_breakdown": result.get('feature_breakdown', {}),
            "xai": result.get('xai', {}),
            "file_info": {
                "filename": file.filename,
                "size_mb": result.get('file_size_mb', 0)
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in image detection: {str(e)}", exc_info=True)
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500
    
    finally:
        # Cleanup
        if temp_file:
            cleanup_temp_file(temp_file)

@detection_bp.route('/video', methods=['POST'])
def detect_video():
    detection_service = current_app.detection_service
    """
    Video deepfake detection endpoint with real-time preprocessing feedback
    """
    start_time = time.time()
    temp_file = None
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        session_id = request.form.get('session_id', None)
        
        if not validate_file_type(file.filename, 'video'):
            return jsonify({"error": "Invalid video format"}), 400
        
        emit_processing_step('File Upload', f'Uploaded {file.filename}', session_id)
        
        temp_file = save_uploaded_file(file, 'video')
        logger.info(f"Saved video to {temp_file}")
        
        emit_processing_step('File Saved', f'Saved to temporary location', session_id)
        
        detection_service = current_app.detection_service
        
        emit_processing_step(
            "Video Processing Started",
            "Running video deepfake detection",
            session_id
        )

        result = detection_service.detect_video(
            str(temp_file),
            emit_callback=lambda step, msg: emit_processing_step(step, msg, session_id)
        )

        emit_processing_step(
            "Video Detection Complete",
            "Video analysis finished",
            session_id
        )

        
        processing_time = time.time() - start_time
        response = {
            "prediction": result.get("prediction", "UNKNOWN"),
            "confidence": float(result.get("confidence", 0.0)),
            "probabilities": result.get(
                "probabilities",
                {
                    "REAL": 1 - float(result.get("confidence", 0.5)),
                    "FAKE": float(result.get("confidence", 0.5)),
                }
            ),
            "processing_time": processing_time,
            "feature_breakdown": result.get("feature_breakdown", {}),
            "temporal_analysis": result.get("temporal_analysis", {}),
            "xai": result.get("xai", {}),
            "file_info": {
                "filename": file.filename,
                "frames_analyzed": result.get("frames_analyzed", 0),
                "fps": result.get("fps", 0),
            },
        }

        if result is None:
            logger.error("Video model returned None")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in video detection: {str(e)}", exc_info=True)
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)


@detection_bp.route('/audio', methods=['POST'])
def detect_audio():
    """
    Audio deepfake detection endpoint with real-time preprocessing feedback
    """
    start_time = time.time()
    temp_file = None
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        session_id = request.form.get('session_id', None)
        
        if not validate_file_type(file.filename, 'audio'):
            return jsonify({"error": "Invalid audio format"}), 400
        
        emit_processing_step('File Upload', f'Uploaded {file.filename}', session_id)
        
        temp_file = save_uploaded_file(file, 'audio')
        logger.info(f"Saved audio to {temp_file}")
        
        emit_processing_step('File Saved', f'Saved to temporary location', session_id)
        
        detection_service = current_app.detection_service
        
        emit_processing_step(
            "Audio Processing Started",
            "Running audio deepfake detection",
            session_id
        )

        result = detection_service.detect_audio(str(temp_file))

        emit_processing_step(
            "Audio Detection Complete",
            "Audio analysis finished",
            session_id
        )

        processing_time = time.time() - start_time
        if result.get("status") == "error":
            return jsonify(result), 500

        response = {
            "prediction": result.get("prediction", "UNKNOWN"),
            "confidence": float(result.get("confidence", 0.0)),
            "probabilities": result.get("probabilities", {
                "REAL": 0.5,
                "FAKE": 0.5
            }),
            "processing_time": processing_time,
            "feature_breakdown": result.get("feature_breakdown", {}),
            "xai": result.get("xai", {}),
            "file_info": {
                "filename": file.filename,
                "duration_seconds": result.get("duration", 0),
                "sample_rate": result.get("sample_rate", 0)
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in audio detection: {str(e)}", exc_info=True)
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)


@detection_bp.route('/multimodal', methods=['POST'])
def detect_multimodal():
    """
    Multi-modal deepfake detection (fusion)
    """
    return jsonify({"message": "Multimodal detection endpoint - Coming soon"}), 501