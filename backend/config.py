import os
from datetime import timedelta

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://admin:password@localhost:5432/fakenews'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # Blockchain
    BLOCKCHAIN_URL = os.getenv('BLOCKCHAIN_URL', 'http://127.0.0.1:8545')
    ARTICLE_REGISTRY_ADDRESS = os.getenv('ARTICLE_REGISTRY_ADDRESS', '')
    PUBLISHER_REGISTRY_ADDRESS = os.getenv('PUBLISHER_REGISTRY_ADDRESS', '')
    ANNOTATION_REGISTRY_ADDRESS = os.getenv('ANNOTATION_REGISTRY_ADDRESS', '')
    PRIVATE_KEY = os.getenv('PRIVATE_KEY', '')
    
    # IPFS
    IPFS_HOST = os.getenv('IPFS_HOST', '127.0.0.1')
    IPFS_PORT = int(os.getenv('IPFS_PORT', '5001'))
    
    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', 'data/models/best_model.pth')
    BERT_MODEL = os.getenv('BERT_MODEL', 'ai4bharat/indic-bert')
    
    # Languages supported
    SUPPORTED_LANGUAGES = ['hindi', 'gujarati', 'marathi', 'telugu']
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}