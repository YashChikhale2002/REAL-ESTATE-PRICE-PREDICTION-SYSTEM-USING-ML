"""Initialize the database with tables and optional sample data"""
from src.database.db_manager import DatabaseManager
import config
import os

def initialize_database(add_sample_data: bool = True):
    """Initialize database with tables and optionally add sample data"""
    
    # Ensure model directory exists
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Create database tables
    db = DatabaseManager()
    db.create_tables()
    
    # Add sample data if requested
    if add_sample_data:
        db.insert_sample_data()
    
    print("Database initialization complete!")

if __name__ == "__main__":
    initialize_database(add_sample_data=True)
