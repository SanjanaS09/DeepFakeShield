"""
API Package for Multi-Modal Deepfake Detection System
Contains all REST API endpoints and middleware
"""

from .detection_routes import detection_bp
from .analysis_routes import analysis_bp
from .health_routes import health_bp
from .middleware import setup_middleware

__all__ = [
    'detection_bp',
    'analysis_bp', 
    'health_bp',
    'setup_middleware'
]
