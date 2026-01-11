import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database configuration
DATABASE_PATH = os.path.join(BASE_DIR, 'real_estate.db')

# Model configuration
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'real_estate_model.joblib')

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed')

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Feature columns
FEATURE_COLUMNS = [
    'area_sqft',
    'bedrooms',
    'bathrooms',
    'floors',
    'age_years',
    'parking_spaces',
    'distance_to_city_center_km',
    'has_garden',
    'has_pool',
    'has_garage'
]

# Flask configuration
SECRET_KEY = 'your-secret-key-change-in-production'
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000
