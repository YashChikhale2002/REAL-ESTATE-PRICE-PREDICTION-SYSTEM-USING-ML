import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import config

class DatabaseManager:
    """Manages SQLite database operations for real estate data"""
    
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_tables(self):
        """Create necessary database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Properties table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                area_sqft REAL NOT NULL,
                bedrooms INTEGER NOT NULL,
                bathrooms INTEGER NOT NULL,
                floors INTEGER NOT NULL,
                age_years INTEGER NOT NULL,
                parking_spaces INTEGER NOT NULL,
                distance_to_city_center_km REAL NOT NULL,
                has_garden INTEGER DEFAULT 0,
                has_pool INTEGER DEFAULT 0,
                has_garage INTEGER DEFAULT 0,
                actual_price REAL,
                location TEXT,
                property_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER,
                predicted_price REAL NOT NULL,
                model_used TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                features TEXT,
                FOREIGN KEY (property_id) REFERENCES properties (id)
            )
        ''')
        
        # Model training history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy_score REAL,
                rmse REAL,
                mae REAL,
                r2_score REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database tables created successfully!")
    
    def add_property(self, property_data: Dict) -> int:
        """Add a new property to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO properties (
                area_sqft, bedrooms, bathrooms, floors, age_years,
                parking_spaces, distance_to_city_center_km, has_garden,
                has_pool, has_garage, actual_price, location, property_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            property_data['area_sqft'],
            property_data['bedrooms'],
            property_data['bathrooms'],
            property_data['floors'],
            property_data['age_years'],
            property_data['parking_spaces'],
            property_data['distance_to_city_center_km'],
            property_data.get('has_garden', 0),
            property_data.get('has_pool', 0),
            property_data.get('has_garage', 0),
            property_data.get('actual_price'),
            property_data.get('location'),
            property_data.get('property_type')
        ))
        
        property_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return property_id
    
    def save_prediction(self, property_id: Optional[int], predicted_price: float, 
                       model_used: str, features: str = None) -> int:
        """Save a prediction to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                property_id, predicted_price, model_used, features
            ) VALUES (?, ?, ?, ?)
        ''', (property_id, predicted_price, model_used, features))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return prediction_id
    
    def get_all_properties(self) -> List[Dict]:
        """Retrieve all properties from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM properties ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        properties = [dict(row) for row in rows]
        conn.close()
        return properties
    
    def get_property_by_id(self, property_id: int) -> Optional[Dict]:
        """Retrieve a specific property by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM properties WHERE id = ?', (property_id,))
        row = cursor.fetchone()
        
        conn.close()
        return dict(row) if row else None
    
    def get_prediction_history(self, limit: int = 50) -> List[Dict]:
        """Retrieve prediction history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.*, pr.location, pr.property_type
            FROM predictions p
            LEFT JOIN properties pr ON p.property_id = pr.id
            ORDER BY p.prediction_date DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        predictions = [dict(row) for row in rows]
        conn.close()
        return predictions
    
    def save_training_history(self, model_name: str, metrics: Dict, parameters: str):
        """Save model training history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_training_history (
                model_name, accuracy_score, rmse, mae, r2_score, parameters
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            metrics.get('accuracy'),
            metrics.get('rmse'),
            metrics.get('mae'),
            metrics.get('r2_score'),
            parameters
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_history(self) -> List[Dict]:
        """Retrieve model training history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_training_history 
            ORDER BY training_date DESC
        ''')
        
        rows = cursor.fetchall()
        history = [dict(row) for row in rows]
        conn.close()
        return history
    
    def insert_sample_data(self):
        """Insert sample property data for testing"""
        sample_properties = [
            {
                'area_sqft': 1500, 'bedrooms': 3, 'bathrooms': 2, 'floors': 1,
                'age_years': 5, 'parking_spaces': 2, 'distance_to_city_center_km': 10,
                'has_garden': 1, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 350000, 'location': 'Downtown', 'property_type': 'Apartment'
            },
            {
                'area_sqft': 2500, 'bedrooms': 4, 'bathrooms': 3, 'floors': 2,
                'age_years': 2, 'parking_spaces': 3, 'distance_to_city_center_km': 5,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 650000, 'location': 'Suburbs', 'property_type': 'Villa'
            },
            {
                'area_sqft': 1200, 'bedrooms': 2, 'bathrooms': 1, 'floors': 1,
                'age_years': 15, 'parking_spaces': 1, 'distance_to_city_center_km': 20,
                'has_garden': 0, 'has_pool': 0, 'has_garage': 0,
                'actual_price': 200000, 'location': 'Outskirts', 'property_type': 'Apartment'
            },
            {
                'area_sqft': 3000, 'bedrooms': 5, 'bathrooms': 4, 'floors': 2,
                'age_years': 1, 'parking_spaces': 4, 'distance_to_city_center_km': 3,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 850000, 'location': 'City Center', 'property_type': 'Villa'
            },
            {
                'area_sqft': 1800, 'bedrooms': 3, 'bathrooms': 2, 'floors': 1,
                'age_years': 8, 'parking_spaces': 2, 'distance_to_city_center_km': 12,
                'has_garden': 1, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 420000, 'location': 'Residential Area', 'property_type': 'House'
            },
            {
                'area_sqft': 2200, 'bedrooms': 4, 'bathrooms': 3, 'floors': 2,
                'age_years': 3, 'parking_spaces': 3, 'distance_to_city_center_km': 7,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 580000, 'location': 'Suburbs', 'property_type': 'House'
            },
            {
                'area_sqft': 1600, 'bedrooms': 3, 'bathrooms': 2, 'floors': 1,
                'age_years': 6, 'parking_spaces': 2, 'distance_to_city_center_km': 11,
                'has_garden': 0, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 380000, 'location': 'Downtown', 'property_type': 'Apartment'
            },
            {
                'area_sqft': 2800, 'bedrooms': 4, 'bathrooms': 3, 'floors': 2,
                'age_years': 4, 'parking_spaces': 3, 'distance_to_city_center_km': 6,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 720000, 'location': 'City Center', 'property_type': 'Villa'
            },
            {
                'area_sqft': 1400, 'bedrooms': 2, 'bathrooms': 2, 'floors': 1,
                'age_years': 10, 'parking_spaces': 1, 'distance_to_city_center_km': 15,
                'has_garden': 0, 'has_pool': 0, 'has_garage': 0,
                'actual_price': 280000, 'location': 'Outskirts', 'property_type': 'Condo'
            },
            {
                'area_sqft': 3200, 'bedrooms': 5, 'bathrooms': 4, 'floors': 3,
                'age_years': 1, 'parking_spaces': 4, 'distance_to_city_center_km': 4,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 950000, 'location': 'City Center', 'property_type': 'Villa'
            },
            {
                'area_sqft': 1900, 'bedrooms': 3, 'bathrooms': 2, 'floors': 1,
                'age_years': 7, 'parking_spaces': 2, 'distance_to_city_center_km': 9,
                'has_garden': 1, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 440000, 'location': 'Residential Area', 'property_type': 'House'
            },
            {
                'area_sqft': 2100, 'bedrooms': 3, 'bathrooms': 3, 'floors': 2,
                'age_years': 5, 'parking_spaces': 2, 'distance_to_city_center_km': 8,
                'has_garden': 1, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 510000, 'location': 'Suburbs', 'property_type': 'House'
            },
            {
                'area_sqft': 1300, 'bedrooms': 2, 'bathrooms': 1, 'floors': 1,
                'age_years': 12, 'parking_spaces': 1, 'distance_to_city_center_km': 18,
                'has_garden': 0, 'has_pool': 0, 'has_garage': 0,
                'actual_price': 250000, 'location': 'Outskirts', 'property_type': 'Apartment'
            },
            {
                'area_sqft': 2600, 'bedrooms': 4, 'bathrooms': 3, 'floors': 2,
                'age_years': 3, 'parking_spaces': 3, 'distance_to_city_center_km': 6,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 680000, 'location': 'Suburbs', 'property_type': 'Villa'
            },
            {
                'area_sqft': 1700, 'bedrooms': 3, 'bathrooms': 2, 'floors': 1,
                'age_years': 9, 'parking_spaces': 2, 'distance_to_city_center_km': 13,
                'has_garden': 1, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 390000, 'location': 'Downtown', 'property_type': 'Apartment'
            },
            {
                'area_sqft': 2400, 'bedrooms': 4, 'bathrooms': 3, 'floors': 2,
                'age_years': 4, 'parking_spaces': 3, 'distance_to_city_center_km': 7,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 620000, 'location': 'Residential Area', 'property_type': 'House'
            },
            {
                'area_sqft': 1550, 'bedrooms': 3, 'bathrooms': 2, 'floors': 1,
                'age_years': 6, 'parking_spaces': 2, 'distance_to_city_center_km': 11,
                'has_garden': 0, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 365000, 'location': 'Suburbs', 'property_type': 'Condo'
            },
            {
                'area_sqft': 2900, 'bedrooms': 5, 'bathrooms': 4, 'floors': 2,
                'age_years': 2, 'parking_spaces': 4, 'distance_to_city_center_km': 5,
                'has_garden': 1, 'has_pool': 1, 'has_garage': 1,
                'actual_price': 780000, 'location': 'City Center', 'property_type': 'Villa'
            },
            {
                'area_sqft': 1450, 'bedrooms': 2, 'bathrooms': 2, 'floors': 1,
                'age_years': 11, 'parking_spaces': 1, 'distance_to_city_center_km': 16,
                'has_garden': 0, 'has_pool': 0, 'has_garage': 0,
                'actual_price': 295000, 'location': 'Outskirts', 'property_type': 'Apartment'
            },
            {
                'area_sqft': 2000, 'bedrooms': 3, 'bathrooms': 2, 'floors': 2,
                'age_years': 5, 'parking_spaces': 2, 'distance_to_city_center_km': 10,
                'has_garden': 1, 'has_pool': 0, 'has_garage': 1,
                'actual_price': 475000, 'location': 'Residential Area', 'property_type': 'House'
            }
        ]
        
        for prop in sample_properties:
            self.add_property(prop)
        
        print(f"Inserted {len(sample_properties)} sample properties!")
