"""
Detection Routes for Multi-Modal Deepfake Detection API
Handles image, video, audio, and multi-modal deepfake detection requests
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
import numpy as np

from utils.validators import validate_file_type, validate_file_size
from utils.file_handlers import save_uploaded_file, cleanup_temp_files
from services.detection_service import DeepfakeDetectionService as DetectionService
from services.explanation_service import ExplanationService
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.video_preprocessor import VideoPreprocessor
from preprocessing.audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)

# Create Blueprint
detection_bp = Blueprint('detection', __name__)

# Initialize services
detection_service = None
explanation_service = None
image_preprocessor = None
video_preprocessor = None
audio_preprocessor = None

def init_detection_services(app):
    """Initialize detection services with app context"""
    global detection_service, explanation_service
    global image_preprocessor, video_preprocessor, audio_preprocessor

    with app.app_context():
        detection_service = DetectionService()
        explanation_service = ExplanationService()
        image_preprocessor = ImagePreprocessor()
        video_preprocessor = VideoPreprocessor()
        audio_preprocessor = AudioPreprocessor()

@detection_bp.route('/image', methods=['POST'])
def detect_image_deepfake():
    """
    Detect deepfakes in uploaded images

    Expected form data:
    - file: Image file (jpg, jpeg, png, bmp, tiff)
    - return_explanation: Optional boolean for XAI explanations
    - platform: Optional platform type (whatsapp, youtube, zoom, generic)
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type and size
        file_type = validate_file_type(file, allowed_types='image')
        if not file_type:
            return jsonify({'error': 'Invalid image file type'}), 400

        if not validate_file_size(file, current_app.config['MAX_CONTENT_LENGTH']):
            return jsonify({'error': 'File size exceeds limit'}), 413

        # Get optional parameters
        return_explanation = request.form.get('return_explanation', 'false').lower() == 'true'
        platform = request.form.get('platform', 'generic')

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = save_uploaded_file(file, unique_filename)

        try:
            # Preprocess image
            preprocessed_data = image_preprocessor.preprocess(
                file_path, 
                platform_config=current_app.config['PLATFORM_CONFIGS'].get(platform, {})
            )

            # Run detection
            detection_result = detection_service.detect_image_deepfake(
                preprocessed_data['processed_image'],
                return_features=True
            )

            # Get feature breakdown
            feature_breakdown = detection_service.get_image_feature_breakdown(
                preprocessed_data['processed_image']
            )

            # Prepare response
            response_data = {
                'prediction': detection_result['label'],
                'confidence': float(detection_result['confidence']),
                'probabilities': {
                    'REAL': float(detection_result['probabilities'][0]),
                    'FAKE': float(detection_result['probabilities'][1])
                },
                'feature_breakdown': feature_breakdown,
                'file_info': {
                    'filename': filename,
                    'file_type': file_type,
                    'platform': platform,
                    'processed_resolution': preprocessed_data.get('resolution', 'unknown')
                },
                'processing_time': preprocessed_data.get('processing_time', 0),
                'timestamp': datetime.utcnow().isoformat()
            }

            # Add explanations if requested
            if return_explanation:
                try:
                    explanations = explanation_service.generate_image_explanations(
                        preprocessed_data['processed_image'],
                        detection_result,
                        methods=['gradcam', 'integrated_gradients']
                    )

                    response_data['explanations'] = {
                        'text_explanation': explanations.get('text_explanation', ''),
                        'visual_explanations': explanations.get('visual_explanations', {}),
                        'attention_maps': explanations.get('attention_maps', {})
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate explanations: {e}")
                    response_data['explanations'] = {'error': 'Explanation generation failed'}

            return jsonify(response_data)

        finally:
            # Clean up temporary files
            cleanup_temp_files(file_path)

    except Exception as e:
        logger.error(f"Error in image detection: {str(e)}")
        return jsonify({'error': 'Internal server error during image detection'}), 500

@detection_bp.route('/video', methods=['POST'])
def detect_video_deepfake():
    """
    Detect deepfakes in uploaded videos

    Expected form data:
    - file: Video file (mp4, avi, mov, mkv, webm, flv)
    - return_explanation: Optional boolean for XAI explanations
    - platform: Optional platform type
    - frames_per_clip: Optional number of frames to analyze (default: 16)
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type and size
        file_type = validate_file_type(file, allowed_types='video')
        if not file_type:
            return jsonify({'error': 'Invalid video file type'}), 400

        if not validate_file_size(file, current_app.config['MAX_CONTENT_LENGTH']):
            return jsonify({'error': 'File size exceeds limit'}), 413

        # Get optional parameters
        return_explanation = request.form.get('return_explanation', 'false').lower() == 'true'
        platform = request.form.get('platform', 'generic')
        frames_per_clip = int(request.form.get('frames_per_clip', 16))

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = save_uploaded_file(file, unique_filename)

        try:
            # Preprocess video
            preprocessed_data = video_preprocessor.preprocess(
                file_path,
                frames_per_clip=frames_per_clip,
                platform_config=current_app.config['PLATFORM_CONFIGS'].get(platform, {})
            )

            # Run detection
            detection_result = detection_service.detect_video_deepfake(
                preprocessed_data['processed_frames'],
                return_features=True
            )

            # Get temporal analysis
            temporal_analysis = detection_service.get_temporal_analysis(
                preprocessed_data['processed_frames']
            )

            # Prepare response
            response_data = {
                'prediction': detection_result['label'],
                'confidence': float(detection_result['confidence']),
                'probabilities': {
                    'REAL': float(detection_result['probabilities'][0]),
                    'FAKE': float(detection_result['probabilities'][1])
                },
                'temporal_analysis': temporal_analysis,
                'file_info': {
                    'filename': filename,
                    'file_type': file_type,
                    'platform': platform,
                    'duration': preprocessed_data.get('duration', 0),
                    'fps': preprocessed_data.get('fps', 0),
                    'frames_analyzed': len(preprocessed_data.get('frame_timestamps', []))
                },
                'processing_time': preprocessed_data.get('processing_time', 0),
                'timestamp': datetime.utcnow().isoformat()
            }

            # Add explanations if requested
            if return_explanation:
                try:
                    explanations = explanation_service.generate_video_explanations(
                        preprocessed_data['processed_frames'],
                        detection_result,
                        frame_timestamps=preprocessed_data.get('frame_timestamps', [])
                    )

                    response_data['explanations'] = {
                        'text_explanation': explanations.get('text_explanation', ''),
                        'temporal_explanations': explanations.get('temporal_explanations', {}),
                        'key_frames': explanations.get('key_frames', [])
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate video explanations: {e}")
                    response_data['explanations'] = {'error': 'Explanation generation failed'}

            return jsonify(response_data)

        finally:
            # Clean up temporary files
            cleanup_temp_files(file_path)

    except Exception as e:
        logger.error(f"Error in video detection: {str(e)}")
        return jsonify({'error': 'Internal server error during video detection'}), 500

@detection_bp.route('/audio', methods=['POST'])
def detect_audio_deepfake():
    """
    Detect deepfakes in uploaded audio files

    Expected form data:
    - file: Audio file (wav, mp3, aac, flac, m4a, ogg)
    - return_explanation: Optional boolean for XAI explanations
    - platform: Optional platform type
    - duration: Optional duration to analyze in seconds (default: 10)
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type and size
        file_type = validate_file_type(file, allowed_types='audio')
        if not file_type:
            return jsonify({'error': 'Invalid audio file type'}), 400

        if not validate_file_size(file, current_app.config['MAX_CONTENT_LENGTH']):
            return jsonify({'error': 'File size exceeds limit'}), 413

        # Get optional parameters
        return_explanation = request.form.get('return_explanation', 'false').lower() == 'true'
        platform = request.form.get('platform', 'generic')
        duration = float(request.form.get('duration', 10.0))

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = save_uploaded_file(file, unique_filename)

        try:
            # Preprocess audio
            preprocessed_data = audio_preprocessor.preprocess(
                file_path,
                duration=duration,
                platform_config=current_app.config['PLATFORM_CONFIGS'].get(platform, {})
            )

            # Run detection
            detection_result = detection_service.detect_audio_deepfake(
                preprocessed_data['processed_audio'],
                return_features=True
            )

            # Get voice analysis
            voice_analysis = detection_service.get_voice_analysis(
                preprocessed_data['processed_audio']
            )

            # Prepare response
            response_data = {
                'prediction': detection_result['label'],
                'confidence': float(detection_result['confidence']),
                'probabilities': {
                    'REAL': float(detection_result['probabilities'][0]),
                    'FAKE': float(detection_result['probabilities'][1])
                },
                'voice_analysis': voice_analysis,
                'file_info': {
                    'filename': filename,
                    'file_type': file_type,
                    'platform': platform,
                    'duration': preprocessed_data.get('duration', 0),
                    'sample_rate': preprocessed_data.get('sample_rate', 0)
                },
                'processing_time': preprocessed_data.get('processing_time', 0),
                'timestamp': datetime.utcnow().isoformat()
            }

            # Add explanations if requested
            if return_explanation:
                try:
                    explanations = explanation_service.generate_audio_explanations(
                        preprocessed_data['processed_audio'],
                        detection_result,
                        voice_analysis
                    )

                    response_data['explanations'] = {
                        'text_explanation': explanations.get('text_explanation', ''),
                        'spectral_analysis': explanations.get('spectral_analysis', {}),
                        'voice_quality_assessment': explanations.get('voice_quality', {})
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate audio explanations: {e}")
                    response_data['explanations'] = {'error': 'Explanation generation failed'}

            return jsonify(response_data)

        finally:
            # Clean up temporary files
            cleanup_temp_files(file_path)

    except Exception as e:
        logger.error(f"Error in audio detection: {str(e)}")
        return jsonify({'error': 'Internal server error during audio detection'}), 500

@detection_bp.route('/multimodal', methods=['POST'])
def detect_multimodal_deepfake():
    """
    Detect deepfakes using multi-modal fusion

    Expected form data:
    - image: Optional image file
    - video: Optional video file  
    - audio: Optional audio file
    - return_explanation: Optional boolean for XAI explanations
    - platform: Optional platform type
    - fusion_method: Optional fusion method (transformer, attention, concat, weighted)
    """
    try:
        # Check if at least one modality is provided
        modalities_provided = []
        if 'image' in request.files and request.files['image'].filename:
            modalities_provided.append('image')
        if 'video' in request.files and request.files['video'].filename:
            modalities_provided.append('video')
        if 'audio' in request.files and request.files['audio'].filename:
            modalities_provided.append('audio')

        if not modalities_provided:
            return jsonify({'error': 'At least one modality file must be provided'}), 400

        # Get optional parameters
        return_explanation = request.form.get('return_explanation', 'false').lower() == 'true'
        platform = request.form.get('platform', 'generic')
        fusion_method = request.form.get('fusion_method', 'transformer')

        # Process each modality
        processed_data = {}
        file_paths = []

        try:
            # Process image if provided
            if 'image' in modalities_provided:
                image_file = request.files['image']
                if validate_file_type(image_file, 'image'):
                    filename = f"{uuid.uuid4()}_{secure_filename(image_file.filename)}"
                    file_path = save_uploaded_file(image_file, filename)
                    file_paths.append(file_path)

                    processed_data['image'] = image_preprocessor.preprocess(
                        file_path,
                        platform_config=current_app.config['PLATFORM_CONFIGS'].get(platform, {})
                    )

            # Process video if provided
            if 'video' in modalities_provided:
                video_file = request.files['video']
                if validate_file_type(video_file, 'video'):
                    filename = f"{uuid.uuid4()}_{secure_filename(video_file.filename)}"
                    file_path = save_uploaded_file(video_file, filename)
                    file_paths.append(file_path)

                    processed_data['video'] = video_preprocessor.preprocess(
                        file_path,
                        platform_config=current_app.config['PLATFORM_CONFIGS'].get(platform, {})
                    )

            # Process audio if provided
            if 'audio' in modalities_provided:
                audio_file = request.files['audio']
                if validate_file_type(audio_file, 'audio'):
                    filename = f"{uuid.uuid4()}_{secure_filename(audio_file.filename)}"
                    file_path = save_uploaded_file(audio_file, filename)
                    file_paths.append(file_path)

                    processed_data['audio'] = audio_preprocessor.preprocess(
                        file_path,
                        platform_config=current_app.config['PLATFORM_CONFIGS'].get(platform, {})
                    )

            # Run multi-modal detection
            detection_result = detection_service.detect_multimodal_deepfake(
                processed_data,
                fusion_method=fusion_method,
                return_features=True
            )

            # Get modality contributions
            modality_contributions = detection_service.get_modality_contributions(
                processed_data, detection_result
            )

            # Prepare response
            response_data = {
                'prediction': detection_result['label'],
                'confidence': float(detection_result['confidence']),
                'probabilities': {
                    'REAL': float(detection_result['probabilities'][0]),
                    'FAKE': float(detection_result['probabilities'][1])
                },
                'modality_contributions': modality_contributions,
                'fusion_info': {
                    'method': fusion_method,
                    'modalities_used': modalities_provided,
                    'attention_weights': detection_result.get('attention_weights', {})
                },
                'processing_time': sum([data.get('processing_time', 0) for data in processed_data.values()]),
                'timestamp': datetime.utcnow().isoformat()
            }

            # Add explanations if requested
            if return_explanation:
                try:
                    explanations = explanation_service.generate_multimodal_explanations(
                        processed_data,
                        detection_result,
                        modality_contributions
                    )

                    response_data['explanations'] = {
                        'text_explanation': explanations.get('text_explanation', ''),
                        'cross_modal_analysis': explanations.get('cross_modal_analysis', {}),
                        'individual_modality_explanations': explanations.get('individual_explanations', {})
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate multimodal explanations: {e}")
                    response_data['explanations'] = {'error': 'Explanation generation failed'}

            return jsonify(response_data)

        finally:
            # Clean up all temporary files
            for file_path in file_paths:
                cleanup_temp_files(file_path)

    except Exception as e:
        logger.error(f"Error in multimodal detection: {str(e)}")
        return jsonify({'error': 'Internal server error during multimodal detection'}), 500

@detection_bp.route('/batch', methods=['POST'])
def detect_batch_files():
    """
    Process multiple files in batch for detection

    Expected form data:
    - files[]: Multiple files of any supported type
    - detection_mode: 'individual' or 'fusion' 
    - return_explanation: Optional boolean
    - platform: Optional platform type
    """
    try:
        # Get uploaded files
        uploaded_files = request.files.getlist('files[]')
        if not uploaded_files:
            return jsonify({'error': 'No files provided for batch processing'}), 400

        # Get parameters
        detection_mode = request.form.get('detection_mode', 'individual')
        return_explanation = request.form.get('return_explanation', 'false').lower() == 'true'
        platform = request.form.get('platform', 'generic')

        # Process files
        results = []
        file_paths = []

        try:
            for i, file in enumerate(uploaded_files):
                if file.filename == '':
                    continue

                # Determine file type
                file_type = validate_file_type(file)
                if not file_type:
                    results.append({
                        'file_index': i,
                        'filename': file.filename,
                        'error': 'Unsupported file type'
                    })
                    continue

                # Save file
                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                file_path = save_uploaded_file(file, filename)
                file_paths.append(file_path)

                # Process based on file type
                try:
                    if file_type == 'image':
                        preprocessed = image_preprocessor.preprocess(file_path)
                        detection_result = detection_service.detect_image_deepfake(
                            preprocessed['processed_image']
                        )
                    elif file_type == 'video':
                        preprocessed = video_preprocessor.preprocess(file_path)
                        detection_result = detection_service.detect_video_deepfake(
                            preprocessed['processed_frames']
                        )
                    elif file_type == 'audio':
                        preprocessed = audio_preprocessor.preprocess(file_path)
                        detection_result = detection_service.detect_audio_deepfake(
                            preprocessed['processed_audio']
                        )
                    else:
                        raise ValueError(f"Unsupported file type: {file_type}")

                    # Add to results
                    result = {
                        'file_index': i,
                        'filename': file.filename,
                        'file_type': file_type,
                        'prediction': detection_result['label'],
                        'confidence': float(detection_result['confidence']),
                        'probabilities': {
                            'REAL': float(detection_result['probabilities'][0]),
                            'FAKE': float(detection_result['probabilities'][1])
                        }
                    }

                    results.append(result)

                except Exception as e:
                    results.append({
                        'file_index': i,
                        'filename': file.filename,
                        'error': f'Processing failed: {str(e)}'
                    })

            # Aggregate results if fusion mode
            if detection_mode == 'fusion' and len(results) > 1:
                # Simple ensemble voting
                fake_votes = sum(1 for r in results if r.get('prediction') == 'FAKE')
                total_confidence = sum(r.get('confidence', 0) for r in results if 'confidence' in r)
                avg_confidence = total_confidence / len([r for r in results if 'confidence' in r])

                ensemble_result = {
                    'ensemble_prediction': 'FAKE' if fake_votes > len(results) / 2 else 'REAL',
                    'ensemble_confidence': avg_confidence,
                    'individual_votes': {'FAKE': fake_votes, 'REAL': len(results) - fake_votes},
                    'consensus_level': max(fake_votes, len(results) - fake_votes) / len(results)
                }
            else:
                ensemble_result = None

            response_data = {
                'batch_results': results,
                'summary': {
                    'total_files': len(uploaded_files),
                    'processed_successfully': len([r for r in results if 'prediction' in r]),
                    'failed_files': len([r for r in results if 'error' in r]),
                    'fake_detections': len([r for r in results if r.get('prediction') == 'FAKE']),
                    'real_detections': len([r for r in results if r.get('prediction') == 'REAL'])
                },
                'timestamp': datetime.utcnow().isoformat()
            }

            if ensemble_result:
                response_data['ensemble_result'] = ensemble_result

            return jsonify(response_data)

        finally:
            # Clean up all temporary files
            for file_path in file_paths:
                cleanup_temp_files(file_path)

    except Exception as e:
        logger.error(f"Error in batch detection: {str(e)}")
        return jsonify({'error': 'Internal server error during batch processing'}), 500

@detection_bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File size exceeds maximum allowed size'}), 413

@detection_bp.errorhandler(BadRequest)
def handle_bad_request(error):
    """Handle bad request errors"""
    return jsonify({'error': 'Bad request: ' + str(error.description)}), 400
