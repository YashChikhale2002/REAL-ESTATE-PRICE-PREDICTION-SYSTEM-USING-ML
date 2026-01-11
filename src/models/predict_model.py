import joblib
import numpy as np
from typing import Dict, Union
import os
import config
from src.database.db_manager import DatabaseManager

class PricePredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_path: str = None):
        self.db = DatabaseManager()
        self.model = None
        self.preprocessor = None
        self.model_path = model_path or config.MODEL_PATH
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
        # Load preprocessor
        preprocessor_path = self.model_path.replace('.joblib', '_preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        
        print("Model loaded successfully!")
    
    def predict_price(self, property_data: Dict) -> float:
        """Predict price for a single property"""
        if self.model is None:
            raise ValueError("Model not loaded. Train a model first.")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded.")
        
        # Prepare data for prediction
        X = self.preprocessor.prepare_for_prediction(property_data)
        
        # Make prediction
        predicted_price = self.model.predict(X)[0]
        
        return float(predicted_price)
    
    def predict_and_save(self, property_data: Dict, 
                        property_id: int = None) -> Dict:
        """Predict price and save to database"""
        predicted_price = self.predict_price(property_data)
        
        # Save prediction to database
        model_name = type(self.model).__name__
        features_str = str(property_data)
        
        prediction_id = self.db.save_prediction(
            property_id, 
            predicted_price, 
            model_name,
            features_str
        )
        
        return {
            'prediction_id': prediction_id,
            'predicted_price': predicted_price,
            'property_id': property_id
        }
    
    def predict_batch(self, properties_list: list) -> list:
        """Predict prices for multiple properties"""
        predictions = []
        
        for prop in properties_list:
            try:
                price = self.predict_price(prop)
                predictions.append({
                    'property': prop,
                    'predicted_price': price
                })
            except Exception as e:
                predictions.append({
                    'property': prop,
                    'error': str(e)
                })
        
        return predictions
    
    def get_price_range(self, property_data: Dict, 
                       confidence: float = 0.15) -> Dict:
        """Get price range with confidence interval"""
        predicted_price = self.predict_price(property_data)
        
        lower_bound = predicted_price * (1 - confidence)
        upper_bound = predicted_price * (1 + confidence)
        
        return {
            'predicted_price': predicted_price,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_percentage': confidence * 100
        }

if __name__ == "__main__":
    # Example usage
    predictor = PricePredictor()
    
    sample_property = {
        'area_sqft': 2000,
        'bedrooms': 3,
        'bathrooms': 2,
        'floors': 2,
        'age_years': 5,
        'parking_spaces': 2,
        'distance_to_city_center_km': 8,
        'has_garden': 1,
        'has_pool': 0,
        'has_garage': 1
    }
    
    result = predictor.get_price_range(sample_property)
    print(f"\nPredicted Price: ${result['predicted_price']:,.2f}")
    print(f"Price Range: ${result['lower_bound']:,.2f} - ${result['upper_bound']:,.2f}")
