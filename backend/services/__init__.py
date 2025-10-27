"""
Services Package for Multi-Modal Deepfake Detection System
Contains service layer classes for detection, analysis, and explanation functionality
Implements business logic and coordinates between models, features, and utilities
"""

from .detection_service import DeepfakeDetectionService
from .analysis_service import MediaAnalysisService
from .explanation_service import ExplanationService

__all__ = [
    'DeepfakeDetectionService',
    'MediaAnalysisService', 
    'ExplanationService'
]
