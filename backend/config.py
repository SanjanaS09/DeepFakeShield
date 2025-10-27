"""
Configuration settings for the Multi-Modal Deepfake Detection System
Supports development, testing, and production environments
"""

import os
from datetime import timedelta
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

class Config:
    """Base configuration class"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 'yes']

    # Server settings
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))

    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL') or f'sqlite:///{BASE_DIR / "deepfake_detection.db"}'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    # File upload settings
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    UPLOAD_FOLDER = BASE_DIR / 'uploads' / 'temp'
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'aac', 'flac', 'm4a', 'ogg'}

    # Model settings
    MODELS_DIR = BASE_DIR / 'models' / 'pretrained'
    DEVICE = os.environ.get('DEVICE', 'cpu')  # 'cpu', 'cuda', 'mps'
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))

    # Detection thresholds
    DETECTION_THRESHOLD = float(os.environ.get('DETECTION_THRESHOLD', 0.5))
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.7))

    # Feature extraction settings
    IMAGE_SIZE = (224, 224)
    VIDEO_FPS = 25
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 10  # seconds

    # XAI settings
    GRADCAM_LAYER_NAMES = ['layer4', 'conv5', 'features']
    ATTENTION_HEADS = 8

    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'deepfake_detection.log'
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    # Security settings
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)

    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'memory://'
    RATELIMIT_DEFAULT = "100 per hour"

    # Cache settings
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = 300

    # Cross-platform adaptation
    PLATFORM_CONFIGS = {
        'whatsapp': {
            'compression_factor': 0.8,
            'max_resolution': (480, 360),
            'audio_bitrate': 64000
        },
        'youtube': {
            'compression_factor': 0.9,
            'max_resolution': (1920, 1080),
            'audio_bitrate': 128000
        },
        'zoom': {
            'compression_factor': 0.7,
            'max_resolution': (640, 480),
            'audio_bitrate': 64000
        },
        'generic': {
            'compression_factor': 1.0,
            'max_resolution': (1920, 1080),
            'audio_bitrate': 128000
        }
    }

    # Model architecture configs
    MODEL_CONFIGS = {
        'image': {
            'backbone': 'xception',  # xception, resnet50, efficientnet-b0, vit-b16
            'pretrained': True,
            'num_classes': 2,
            'dropout': 0.5
        },
        'video': {
            'backbone': 'i3d',  # i3d, slowfast, x3d
            'pretrained': True,
            'num_classes': 2,
            'frames_per_clip': 16,
            'frame_sampling_rate': 2
        },
        'audio': {
            'backbone': 'ecapa_tdnn',  # ecapa_tdnn, wav2vec2, hubert
            'pretrained': True,
            'num_classes': 2,
            'sample_rate': AUDIO_SAMPLE_RATE
        },
        'fusion': {
            'fusion_type': 'transformer',  # concat, attention, transformer
            'num_heads': 8,
            'num_layers': 4,
            'hidden_dim': 512
        }
    }

    @staticmethod
    def init_app(app):
        """Initialize application-specific configuration"""
        # Ensure required directories exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.LOG_FILE.parent, exist_ok=True)


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    LOG_LEVEL = 'DEBUG'


class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    LOG_LEVEL = 'WARNING'

    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DATABASE_URL = os.environ.get('DATABASE_URL')

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler

        if not app.debug:
            # Setup file logging
            file_handler = RotatingFileHandler(
                cls.LOG_FILE,
                maxBytes=cls.LOG_MAX_BYTES,
                backupCount=cls.LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('Deepfake Detection System startup')


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
