import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train_model import ModelTrainer
from src.models.predict_model import PricePredictor
from src.database.db_manager import DatabaseManager
import config

class TestRealEstateML(unittest.TestCase):
    
    def setUp(self):
        """Setup test database"""
        self.db = DatabaseManager(':memory:')
        self.db.create_tables()
        self.db.insert_sample_data()
    
    def test_database_creation(self):
        """Test database table creation"""
        properties = self.db.get_all_properties()
        self.assertGreater(len(properties), 0)
    
    def test_property_insertion(self):
        """Test adding property to database"""
        property_data = {
            'area_sqft': 2000,
            'bedrooms': 3,
            'bathrooms': 2,
            'floors': 1,
            'age_years': 5,
            'parking_spaces': 2,
            'distance_to_city_center_km': 10,
            'has_garden': 1,
            'has_pool': 0,
            'has_garage': 1,
            'actual_price': 400000
        }
        
        property_id = self.db.add_property(property_data)
        self.assertIsNotNone(property_id)
        
        retrieved = self.db.get_property_by_id(property_id)
        self.assertEqual(retrieved['area_sqft'], 2000)
    
    def test_prediction(self):
        """Test price prediction"""
        # This test requires a trained model
        # Skip if model doesn't exist
        if not os.path.exists(config.MODEL_PATH):
            self.skipTest("Model not trained yet")
        
        predictor = PricePredictor()
        
        property_data = {
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
        
        predicted_price = predictor.predict_price(property_data)
        self.assertGreater(predicted_price, 0)

if __name__ == '__main__':
    unittest.main()
