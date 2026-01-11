import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import config

class DataPreprocessor:
    """Handles data preprocessing for real estate prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = config.FEATURE_COLUMNS
    
    def prepare_data(self, properties: list) -> pd.DataFrame:
        """Convert property list to DataFrame"""
        df = pd.DataFrame(properties)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and handle missing values"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Only remove outliers if we have sufficient data (more than 20 samples)
        if len(df) > 20:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in ['id', 'has_garden', 'has_pool', 'has_garage']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def extract_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target variable"""
        X = df[self.feature_columns]
        y = df['actual_price']
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features using StandardScaler"""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def prepare_for_training(self, properties: list) -> Tuple[np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline for training"""
        df = self.prepare_data(properties)
        df = self.clean_data(df)
        X, y = self.extract_features_target(df)
        X_scaled = self.scale_features(X, fit=True)
        return X_scaled, y.values
    
    def prepare_for_prediction(self, property_data: dict) -> np.ndarray:
        """Prepare single property for prediction"""
        # Create DataFrame from single property
        df = pd.DataFrame([property_data])
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_columns]
        X_scaled = self.scale_features(X, fit=False)
        return X_scaled
