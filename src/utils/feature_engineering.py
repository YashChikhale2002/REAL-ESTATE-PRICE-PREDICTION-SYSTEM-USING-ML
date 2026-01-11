import pandas as pd
import numpy as np

class FeatureEngineer:
    """Creates additional features for improved predictions"""
    
    def __init__(self):
        pass
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features"""
        df = df.copy()
        
        # Price per square foot
        if 'actual_price' in df.columns and 'area_sqft' in df.columns:
            df['price_per_sqft'] = df['actual_price'] / df['area_sqft']
        
        # Total rooms
        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        
        # Luxury score
        if all(col in df.columns for col in ['has_garden', 'has_pool', 'has_garage']):
            df['luxury_score'] = df['has_garden'] + df['has_pool'] + df['has_garage']
        
        # Age category
        if 'age_years' in df.columns:
            df['age_category'] = pd.cut(df['age_years'], 
                                       bins=[0, 5, 10, 20, 100],
                                       labels=['new', 'recent', 'old', 'very_old'])
        
        # Area category
        if 'area_sqft' in df.columns:
            df['area_category'] = pd.cut(df['area_sqft'],
                                        bins=[0, 1000, 2000, 3000, 10000],
                                        labels=['small', 'medium', 'large', 'very_large'])
        
        # Distance score (inverse of distance)
        if 'distance_to_city_center_km' in df.columns:
            df['proximity_score'] = 1 / (df['distance_to_city_center_km'] + 1)
        
        return df
    
    def get_feature_importance_names(self) -> list:
        """Return names of engineered features"""
        return [
            'price_per_sqft', 'total_rooms', 'luxury_score',
            'proximity_score'
        ]
