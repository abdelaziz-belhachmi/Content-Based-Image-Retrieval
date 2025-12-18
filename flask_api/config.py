"""
Flask REST API Configuration
"""

import os
from pathlib import Path


class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # API settings
    API_PREFIX = '/api'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # Paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MODELS_FOLDER = BASE_DIR.parent / 'yolo_training' / 'models'
    
    # YOLO model
    YOLO_MODEL_PATH = MODELS_FOLDER / 'yolov8n_custom_best.pt'
    YOLO_CONFIDENCE_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    
    # Feature extraction settings
    DOMINANT_COLORS_K = 5
    GABOR_SCALES = 4
    GABOR_ORIENTATIONS = 6
    LBP_RADIUS = 3
    LBP_POINTS = 24
    
    # Similarity search settings
    DEFAULT_TOP_K = 10
    DEFAULT_DISTANCE_METRIC = 'cosine'
    
    # Class mapping
    CLASS_NAMES = {
        0: 'pineapple',
        1: 'apple',
        2: 'bell_pepper',
        3: 'bicycle',
        4: 'broccoli',
        5: 'bus',
        6: 'car',
        7: 'cat',
        8: 'dog',
        9: 'elephant',
        10: 'horse',
        11: 'lemon',
        12: 'motorcycle',
        13: 'strawberry',
        14: 'tomato'
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig)
