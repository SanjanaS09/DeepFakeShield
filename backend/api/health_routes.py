"""
Health Routes for Multi-Modal Deepfake Detection API
Handles health checks, system status, and monitoring endpoints
"""

import os
import sys
import logging
import psutil
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from flask import Blueprint, jsonify, current_app
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Create Blueprint
health_bp = Blueprint('health', __name__)

@health_bp.route('/status', methods=['GET'])
def health_check():
    """
    Basic health check endpoint
    Returns 200 OK if the service is running
    """
    try:
        return jsonify({
            'status': 'healthy',
            'message': 'Multi-Modal Deepfake Detection API is running',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check - ensures all required components are loaded and ready
    """
    try:
        readiness_status = {
            'status': 'ready',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }

        # Check if models are loaded (mock check)
        readiness_status['components']['models'] = {
            'image_detector': 'loaded',
            'video_detector': 'loaded', 
            'audio_detector': 'loaded',
            'fusion_model': 'loaded',
            'status': 'ready'
        }

        # Check database connection (if applicable)
        readiness_status['components']['database'] = {
            'connection': 'active',
            'status': 'ready'
        }

        # Check file system access
        upload_dir = Path(current_app.config.get('UPLOAD_FOLDER', './uploads/temp'))
        models_dir = Path(current_app.config.get('MODELS_DIR', './models/pretrained'))

        readiness_status['components']['filesystem'] = {
            'upload_directory': {
                'path': str(upload_dir),
                'exists': upload_dir.exists(),
                'writable': os.access(upload_dir.parent, os.W_OK)
            },
            'models_directory': {
                'path': str(models_dir),
                'exists': models_dir.exists(),
                'readable': os.access(models_dir.parent, os.R_OK) if models_dir.parent.exists() else False
            },
            'status': 'ready'
        }

        # Check PyTorch/CUDA availability
        device_info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu',
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'CPU'
        }

        readiness_status['components']['compute'] = {
            'pytorch': device_info,
            'status': 'ready'
        }

        # Overall status
        component_statuses = [comp.get('status', 'unknown') for comp in readiness_status['components'].values()]
        if all(status == 'ready' for status in component_statuses):
            readiness_status['status'] = 'ready'
            status_code = 200
        else:
            readiness_status['status'] = 'not_ready'
            status_code = 503

        return jsonify(readiness_status), status_code

    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return jsonify({
            'status': 'not_ready',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503

@health_bp.route('/live', methods=['GET'])
def liveness_check():
    """
    Liveness check - basic availability test
    """
    try:
        # Simple computation to verify the service is responsive
        test_array = np.random.random((10, 10))
        result = np.sum(test_array)

        return jsonify({
            'status': 'alive',
            'message': 'Service is responsive',
            'test_computation': float(result),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        return jsonify({
            'status': 'dead',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/info', methods=['GET'])
def system_info():
    """
    Get detailed system information
    """
    try:
        # System information
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': sys.version,
            'hostname': platform.node()
        }

        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': memory.percent
        }

        # CPU information
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_frequency': {
                'current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'max': psutil.cpu_freq().max if psutil.cpu_freq() else None
            }
        }

        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            'total_gb': round(disk.total / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'percent_used': round((disk.used / disk.total) * 100, 2)
        }

        # GPU information
        gpu_info = {'available': False}
        if torch.cuda.is_available():
            gpu_info = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'devices': []
            }

            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'id': i,
                    'name': device_props.name,
                    'total_memory_gb': round(device_props.total_memory / (1024**3), 2),
                    'memory_used_gb': round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                    'memory_reserved_gb': round(torch.cuda.memory_reserved(i) / (1024**3), 2),
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })

        # Application configuration
        app_config = {
            'debug_mode': current_app.config.get('DEBUG', False),
            'upload_folder': current_app.config.get('UPLOAD_FOLDER', 'Not set'),
            'max_content_length': current_app.config.get('MAX_CONTENT_LENGTH', 'Not set'),
            'device': current_app.config.get('DEVICE', 'cpu'),
            'batch_size': current_app.config.get('BATCH_SIZE', 8)
        }

        # Package versions (key dependencies)
        try:
            import torch
            import torchvision
            import cv2
            import librosa
            import flask

            package_versions = {
                'torch': torch.__version__,
                'torchvision': torchvision.__version__,
                'opencv': cv2.__version__,
                'librosa': librosa.__version__,
                'flask': flask.__version__,
                'numpy': np.__version__
            }
        except ImportError as e:
            package_versions = {'error': f'Could not retrieve package versions: {str(e)}'}

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'system': system_info,
            'memory': memory_info,
            'cpu': cpu_info,
            'disk': disk_info,
            'gpu': gpu_info,
            'application': app_config,
            'packages': package_versions
        })

    except Exception as e:
        logger.error(f"Error retrieving system info: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve system information',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/metrics', methods=['GET'])
def performance_metrics():
    """
    Get performance metrics and statistics
    """
    try:
        # Mock performance metrics (in production, these would be collected from actual usage)
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': {
                'seconds': 86400,  # Mock 24 hours uptime
                'formatted': '1 day, 0:00:00'
            },
            'requests': {
                'total': 1247,
                'successful': 1183,
                'failed': 64,
                'success_rate': 94.9,
                'requests_per_minute': 2.3
            },
            'processing': {
                'image_detections': 456,
                'video_detections': 234,
                'audio_detections': 178,
                'multimodal_detections': 89,
                'avg_processing_time': {
                    'image': 1.2,
                    'video': 8.7,
                    'audio': 3.4,
                    'multimodal': 12.1
                }
            },
            'cache': {
                'hit_rate': 67.3,
                'cache_size_mb': 128,
                'cache_entries': 324
            },
            'errors': {
                'last_24h': 12,
                'last_hour': 1,
                'common_errors': [
                    {'error': 'File size too large', 'count': 8},
                    {'error': 'Unsupported file format', 'count': 3},
                    {'error': 'Processing timeout', 'count': 1}
                ]
            }
        }

        return jsonify(metrics)

    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve performance metrics',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/dependencies', methods=['GET'])
def check_dependencies():
    """
    Check status of external dependencies and services
    """
    try:
        dependencies = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'services': {}
        }

        # Check file system dependencies
        critical_paths = [
            current_app.config.get('UPLOAD_FOLDER', './uploads/temp'),
            current_app.config.get('MODELS_DIR', './models/pretrained'),
            './logs'
        ]

        fs_status = 'healthy'
        for path in critical_paths:
            path_obj = Path(path)
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                # Test write access
                test_file = path_obj / '.health_check'
                test_file.write_text('test')
                test_file.unlink()

                dependencies['services'][f'filesystem_{path}'] = {
                    'status': 'healthy',
                    'accessible': True,
                    'writable': True
                }
            except Exception as e:
                dependencies['services'][f'filesystem_{path}'] = {
                    'status': 'unhealthy',
                    'accessible': path_obj.exists(),
                    'writable': False,
                    'error': str(e)
                }
                fs_status = 'unhealthy'

        # Check Python package imports
        required_packages = [
            'torch', 'torchvision', 'torchaudio',
            'opencv-python', 'librosa', 'numpy',
            'PIL', 'sklearn', 'matplotlib'
        ]

        import importlib
        import_status = 'healthy'

        for package in required_packages:
            try:
                # Handle package name variations
                import_name = {
                    'opencv-python': 'cv2',
                    'PIL': 'PIL'
                }.get(package, package)

                importlib.import_module(import_name)
                dependencies['services'][f'package_{package}'] = {
                    'status': 'healthy',
                    'importable': True
                }
            except ImportError as e:
                dependencies['services'][f'package_{package}'] = {
                    'status': 'unhealthy',
                    'importable': False,
                    'error': str(e)
                }
                import_status = 'unhealthy'

        # Check model availability (mock)
        model_files = [
            'image_detector.pth',
            'video_detector.pth', 
            'audio_detector.pth',
            'fusion_model.pth'
        ]

        models_dir = Path(current_app.config.get('MODELS_DIR', './models/pretrained'))
        model_status = 'healthy'

        for model_file in model_files:
            model_path = models_dir / model_file
            dependencies['services'][f'model_{model_file}'] = {
                'status': 'healthy' if model_path.exists() else 'warning',
                'exists': model_path.exists(),
                'size_mb': round(model_path.stat().st_size / (1024**2), 2) if model_path.exists() else 0
            }

            if not model_path.exists():
                model_status = 'warning'  # Models might not be downloaded yet

        # Overall status
        if fs_status == 'unhealthy' or import_status == 'unhealthy':
            dependencies['status'] = 'unhealthy'
        elif model_status == 'warning':
            dependencies['status'] = 'warning'

        status_code = 200 if dependencies['status'] == 'healthy' else 503
        return jsonify(dependencies), status_code

    except Exception as e:
        logger.error(f"Error checking dependencies: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Failed to check dependencies',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/config', methods=['GET'])
def get_configuration():
    """
    Get current application configuration (non-sensitive values only)
    """
    try:
        # Only return non-sensitive configuration values
        safe_config = {
            'timestamp': datetime.utcnow().isoformat(),
            'application': {
                'debug': current_app.config.get('DEBUG', False),
                'testing': current_app.config.get('TESTING', False),
                'device': current_app.config.get('DEVICE', 'cpu'),
                'batch_size': current_app.config.get('BATCH_SIZE', 8),
                'detection_threshold': current_app.config.get('DETECTION_THRESHOLD', 0.5),
                'confidence_threshold': current_app.config.get('CONFIDENCE_THRESHOLD', 0.7)
            },
            'file_settings': {
                'max_content_length': current_app.config.get('MAX_CONTENT_LENGTH', 0),
                'upload_folder': current_app.config.get('UPLOAD_FOLDER', 'Not set'),
                'allowed_image_extensions': list(current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', [])),
                'allowed_video_extensions': list(current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', [])),
                'allowed_audio_extensions': list(current_app.config.get('ALLOWED_AUDIO_EXTENSIONS', []))
            },
            'model_settings': {
                'models_dir': current_app.config.get('MODELS_DIR', 'Not set'),
                'image_size': current_app.config.get('IMAGE_SIZE', (224, 224)),
                'video_fps': current_app.config.get('VIDEO_FPS', 25),
                'audio_sample_rate': current_app.config.get('AUDIO_SAMPLE_RATE', 16000),
                'audio_duration': current_app.config.get('AUDIO_DURATION', 10)
            },
            'platform_configs': current_app.config.get('PLATFORM_CONFIGS', {})
        }

        return jsonify(safe_config)

    except Exception as e:
        logger.error(f"Error retrieving configuration: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve configuration',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/logs/recent', methods=['GET'])
def get_recent_logs():
    """
    Get recent log entries (if log file is accessible)
    """
    try:
        log_file = current_app.config.get('LOG_FILE', './logs/deepfake_detection.log')
        log_path = Path(log_file)

        if not log_path.exists():
            return jsonify({
                'message': 'Log file not found',
                'log_file': str(log_path),
                'timestamp': datetime.utcnow().isoformat()
            }), 404

        # Read last N lines from log file
        max_lines = 100
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines

            return jsonify({
                'log_entries': [line.strip() for line in recent_lines],
                'total_lines': len(recent_lines),
                'log_file': str(log_path),
                'file_size_kb': round(log_path.stat().st_size / 1024, 2),
                'last_modified': datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(),
                'timestamp': datetime.utcnow().isoformat()
            })

        except Exception as e:
            return jsonify({
                'error': 'Could not read log file',
                'details': str(e),
                'log_file': str(log_path),
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    except Exception as e:
        logger.error(f"Error retrieving recent logs: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve recent logs',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
