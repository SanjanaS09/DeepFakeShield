# # CORRECT DETECTION ROUTES - FIXED FOR YOUR ACTUAL SERVICE
# # File: backend/api/detection_routes.py

# """
# Detection Routes for Multi-Modal Deepfake Detection API
# Handles image, video, audio, and multi-modal deepfake detection requests
# """

# import asyncio
# import uuid
# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List, Optional

# from flask import Blueprint, request, jsonify, current_app
# import numpy as np

# from services.detection_service import DeepfakeDetectionService, DetectionMode, DetectionRequest
# from utils.file_handlers import save_uploaded_file, cleanup_temp_files

# logger = logging.getLogger(__name__)

# detection_bp = Blueprint('detection', __name__, url_prefix='/api/detection')


# def get_event_loop():
#     """Get or create event loop for async operations"""
#     try:
#         loop = asyncio.get_event_loop()
#         if loop.is_closed():
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#         return loop
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop


# @detection_bp.route('/image', methods=['POST'])
# def detect_image_deepfake():
#     """
#     Detect deepfake in image
#     """
#     try:
#         logger.info("Image detection request received")
        
#         # Check if file exists
#         if 'file' not in request.files:
#             logger.warning("No file part in request")
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
        
#         # Check if file is selected
#         if file.filename == '':
#             logger.warning("No file selected")
#             return jsonify({'error': 'No file selected'}), 400
        
#         # Simple file type check - allow common image formats
#         allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}
#         file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
#         if file_ext not in allowed_extensions:
#             logger.warning(f"Invalid file type: {file_ext}")
#             return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
        
#         # Save file
#         file_path = save_uploaded_file(file, 'image')
#         logger.info(f"File saved to: {file_path}")
        
#         # Get detection service
#         detection_service = current_app.detection_service
        
#         # Create detection request
#         detection_request = DetectionRequest(
#             file_path=str(file_path),
#             mode=DetectionMode.IMAGE,
#             enable_xai=True,
#             extract_features=True,
#             quality_assessment=True,
#             request_id=str(uuid.uuid4())
#         )
        
#         # Run async detection
#         loop = get_event_loop()
#         detection_result = loop.run_until_complete(
#             detection_service.detect_deepfake(detection_request)
#         )
        
#         logger.info(f"Detection result: is_deepfake={detection_result.is_deepfake}, confidence={detection_result.confidence_score}")
        
#         # Prepare response
#         response = {
#             'detection_id': detection_result.request_id,
#             'timestamp': datetime.now().isoformat(),
#             'is_fake': detection_result.is_deepfake,
#             'confidence': float(detection_result.confidence_score),
#             'processing_time': float(detection_result.processing_time),
#             'model_results': detection_result.model_results,
#             'quality_metrics': detection_result.quality_metrics or {},
#             'explanation': detection_result.explanation or {},
#         }
        
#         # Cleanup
#         cleanup_temp_files([str(file_path)])
        
#         logger.info(f"Detection completed. Response: {response}")
#         return jsonify(response), 200
        
#     except Exception as e:
#         logger.error(f"Unexpected error in image detection: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


# @detection_bp.route('/video', methods=['POST'])
# def detect_video_deepfake():
#     """
#     Detect deepfake in video
#     """
#     try:
#         logger.info("Video detection request received")
        
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         # Simple file type check
#         allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
#         file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
#         if file_ext not in allowed_extensions:
#             return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
        
#         # Save file
#         file_path = save_uploaded_file(file, 'video')
#         logger.info(f"File saved to: {file_path}")
        
#         # Get detection service
#         detection_service = current_app.detection_service
        
#         # Create detection request
#         detection_request = DetectionRequest(
#             file_path=str(file_path),
#             mode=DetectionMode.VIDEO,
#             enable_xai=True,
#             extract_features=True,
#             quality_assessment=True,
#             request_id=str(uuid.uuid4())
#         )
        
#         # Run async detection
#         loop = get_event_loop()
#         detection_result = loop.run_until_complete(
#             detection_service.detect_deepfake(detection_request)
#         )
        
#         logger.info(f"Detection result: is_deepfake={detection_result.is_deepfake}, confidence={detection_result.confidence_score}")
        
#         response = {
#             'detection_id': detection_result.request_id,
#             'timestamp': datetime.now().isoformat(),
#             'is_fake': detection_result.is_deepfake,
#             'confidence': float(detection_result.confidence_score),
#             'processing_time': float(detection_result.processing_time),
#             'model_results': detection_result.model_results,
#             'quality_metrics': detection_result.quality_metrics or {},
#             'explanation': detection_result.explanation or {},
#         }
        
#         cleanup_temp_files([str(file_path)])
#         logger.info(f"Detection completed. Response: {response}")
#         return jsonify(response), 200
        
#     except Exception as e:
#         logger.error(f"Unexpected error in video detection: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


# @detection_bp.route('/audio', methods=['POST'])
# def detect_audio_deepfake():
#     """
#     Detect deepfake in audio
#     """
#     try:
#         logger.info("Audio detection request received")
        
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         # Simple file type check
#         allowed_extensions = {'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'wma'}
#         file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
#         if file_ext not in allowed_extensions:
#             return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
        
#         # Save file
#         file_path = save_uploaded_file(file, 'audio')
#         logger.info(f"File saved to: {file_path}")
        
#         # Get detection service
#         detection_service = current_app.detection_service
        
#         # Create detection request
#         detection_request = DetectionRequest(
#             file_path=str(file_path),
#             mode=DetectionMode.AUDIO,
#             enable_xai=True,
#             extract_features=True,
#             quality_assessment=True,
#             request_id=str(uuid.uuid4())
#         )
        
#         # Run async detection
#         loop = get_event_loop()
#         detection_result = loop.run_until_complete(
#             detection_service.detect_deepfake(detection_request)
#         )
        
#         logger.info(f"Detection result: is_deepfake={detection_result.is_deepfake}, confidence={detection_result.confidence_score}")
        
#         response = {
#             'detection_id': detection_result.request_id,
#             'timestamp': datetime.now().isoformat(),
#             'is_fake': detection_result.is_deepfake,
#             'confidence': float(detection_result.confidence_score),
#             'processing_time': float(detection_result.processing_time),
#             'model_results': detection_result.model_results,
#             'quality_metrics': detection_result.quality_metrics or {},
#             'explanation': detection_result.explanation or {},
#         }
        
#         cleanup_temp_files([str(file_path)])
#         logger.info(f"Detection completed. Response: {response}")
#         return jsonify(response), 200
        
#     except Exception as e:
#         logger.error(f"Unexpected error in audio detection: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


# @detection_bp.route('/multimodal', methods=['POST'])
# def detect_multimodal_deepfake():
#     """
#     Detect deepfake using multiple modalities
#     """
#     try:
#         logger.info("Multimodal detection request received")
        
#         # Check for files
#         has_image = 'image' in request.files and request.files['image'].filename != ''
#         has_video = 'video' in request.files and request.files['video'].filename != ''
#         has_audio = 'audio' in request.files and request.files['audio'].filename != ''
        
#         if not (has_image or has_video or has_audio):
#             return jsonify({'error': 'At least one file required (image, video, or audio)'}), 400
        
#         detection_service = current_app.detection_service
#         results = {}
        
#         # Process image
#         if has_image:
#             try:
#                 file = request.files['image']
#                 file_path = save_uploaded_file(file, 'image')
                
#                 detection_request = DetectionRequest(
#                     file_path=str(file_path),
#                     mode=DetectionMode.IMAGE,
#                     request_id=str(uuid.uuid4())
#                 )
                
#                 loop = get_event_loop()
#                 result = loop.run_until_complete(
#                     detection_service.detect_deepfake(detection_request)
#                 )
                
#                 results['image'] = {
#                     'is_fake': result.is_deepfake,
#                     'confidence': float(result.confidence_score),
#                 }
#                 cleanup_temp_files([str(file_path)])
                
#             except Exception as e:
#                 logger.error(f"Image detection error: {str(e)}")
#                 results['image'] = {'error': str(e)}
        
#         # Process video
#         if has_video:
#             try:
#                 file = request.files['video']
#                 file_path = save_uploaded_file(file, 'video')
                
#                 detection_request = DetectionRequest(
#                     file_path=str(file_path),
#                     mode=DetectionMode.VIDEO,
#                     request_id=str(uuid.uuid4())
#                 )
                
#                 loop = get_event_loop()
#                 result = loop.run_until_complete(
#                     detection_service.detect_deepfake(detection_request)
#                 )
                
#                 results['video'] = {
#                     'is_fake': result.is_deepfake,
#                     'confidence': float(result.confidence_score),
#                 }
#                 cleanup_temp_files([str(file_path)])
                
#             except Exception as e:
#                 logger.error(f"Video detection error: {str(e)}")
#                 results['video'] = {'error': str(e)}
        
#         # Process audio
#         if has_audio:
#             try:
#                 file = request.files['audio']
#                 file_path = save_uploaded_file(file, 'audio')
                
#                 detection_request = DetectionRequest(
#                     file_path=str(file_path),
#                     mode=DetectionMode.AUDIO,
#                     request_id=str(uuid.uuid4())
#                 )
                
#                 loop = get_event_loop()
#                 result = loop.run_until_complete(
#                     detection_service.detect_deepfake(detection_request)
#                 )
                
#                 results['audio'] = {
#                     'is_fake': result.is_deepfake,
#                     'confidence': float(result.confidence_score),
#                 }
#                 cleanup_temp_files([str(file_path)])
                
#             except Exception as e:
#                 logger.error(f"Audio detection error: {str(e)}")
#                 results['audio'] = {'error': str(e)}
        
#         response = {
#             'detection_id': str(uuid.uuid4()),
#             'timestamp': datetime.now().isoformat(),
#             'results': results,
#             'overall_is_fake': any(r.get('is_fake', False) for r in results.values() if isinstance(r, dict) and 'is_fake' in r),
#         }
        
#         logger.info(f"Multimodal detection completed. Response: {response}")
#         return jsonify(response), 200
        
#     except Exception as e:
#         logger.error(f"Unexpected error in multimodal detection: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


# @detection_bp.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     try:
#         detection_service = current_app.detection_service
#         health = detection_service.health_check()
#         return jsonify(health), 200
#     except Exception as e:
#         logger.error(f"Health check failed: {str(e)}")
#         return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# @detection_bp.route('/stats', methods=['GET'])
# def get_stats():
#     """Get detection statistics"""
#     try:
#         detection_service = current_app.detection_service
#         stats = detection_service.get_stats()
#         return jsonify(stats), 200
#     except Exception as e:
#         logger.error(f"Stats retrieval failed: {str(e)}")
#         return jsonify({'error': str(e)}), 500

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

from services.detection_service import DeepfakeDetectionService
from utils.validators import validate_file_type, validate_file_size
from utils.file_handlers import save_uploaded_file, cleanup_temp_file

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__)


def emit_processing_step(step_name, details, session_id=None):
    """Emit processing step to frontend via WebSocket"""
    try:
        from flask_socketio import emit
        emit('processing_step', {
            'name': step_name,
            'details': details,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'completed'
        }, room=session_id)
    except Exception as e:
        logger.warning(f"Failed to emit processing step: {e}")


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
        
        file = request.files['file']
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
        result = detection_service.detect_image(
            str(temp_file),
            session_id=session_id,
            emit_callback=emit_processing_step
        )
        
        emit_processing_step('Detection Complete', 'Analysis finished', session_id)
        
        # Prepare response
        processing_time = time.time() - start_time
        response = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "processing_time": processing_time,
            "feature_breakdown": result.get('feature_breakdown', {}),
            "xai_visualization": result.get('xai_visualization', {}),
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
        
        emit_processing_step('Preprocessing Started', 'Initializing video preprocessor', session_id)
        
        result = detection_service.detect_video(
            str(temp_file),
            session_id=session_id,
            emit_callback=emit_processing_step
        )
        
        emit_processing_step('Detection Complete', 'Analysis finished', session_id)
        
        processing_time = time.time() - start_time
        response = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "processing_time": processing_time,
            "feature_breakdown": result.get('feature_breakdown', {}),
            "temporal_analysis": result.get('temporal_analysis', {}),
            "xai_visualization": result.get('xai_visualization', {}),
            "file_info": {
                "filename": file.filename,
                "frames_analyzed": result.get('frames_analyzed', 0),
                "fps": result.get('fps', 0)
            }
        }
        
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
        
        emit_processing_step('Preprocessing Started', 'Initializing audio preprocessor', session_id)
        
        result = detection_service.detect_audio(
            str(temp_file),
            session_id=session_id,
            emit_callback=emit_processing_step
        )
        
        emit_processing_step('Detection Complete', 'Analysis finished', session_id)
        
        processing_time = time.time() - start_time
        response = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "processing_time": processing_time,
            "feature_breakdown": result.get('feature_breakdown', {}),
            "xai_visualization": result.get('xai_visualization', {}),
            "file_info": {
                "filename": file.filename,
                "duration_seconds": result.get('duration', 0),
                "sample_rate": result.get('sample_rate', 0)
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