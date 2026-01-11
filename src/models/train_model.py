import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple
import config
from src.database.db_manager import DatabaseManager
from src.utils.data_preprocessing import DataPreprocessor

class ModelTrainer:
    """Train and evaluate machine learning models for price prediction"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.model_name = None
        self.metrics = {}
    
    def load_data_from_db(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from database"""
        properties = self.db.get_all_properties()
        
        # Filter properties with actual prices
        properties = [p for p in properties if p['actual_price'] is not None]
        
        if len(properties) < 10:
            raise ValueError(f"Insufficient training data. Need at least 10 properties with actual prices, but only {len(properties)} found. Please add more property data to the database.")
        
        X, y = self.preprocessor.prepare_for_training(properties)
        return X, y

    def train_linear_regression(self, X_train, y_train) -> LinearRegression:
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train, y_train) -> RandomForestRegressor:
        """Train Random Forest model"""
        params = config.MODEL_PARAMS['random_forest']
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        return model
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        params = config.MODEL_PARAMS['xgboost']
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test) -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'accuracy': r2_score(y_test, y_pred) * 100
        }
        
        return metrics
    
    def train_and_evaluate(self, model_type: str = 'random_forest', 
                        test_size: float = 0.2) -> Dict:
        """Complete training and evaluation pipeline"""
        print(f"Training {model_type} model...")
        
        # Load data
        X, y = self.load_data_from_db()
        
        # Determine appropriate test size and CV folds based on data size
        n_samples = len(X)
        print(f"Total samples: {n_samples}")
        
        if n_samples < 10:
            raise ValueError(f"Insufficient training data. Need at least 10 properties, but only {n_samples} available. Please add more property data.")
        
        # Adjust test size for small datasets
        if n_samples < 30:
            test_size = 0.15  # Use smaller test set for small datasets
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train model based on type
        if model_type == 'linear_regression':
            self.model = self.train_linear_regression(X_train, y_train)
        elif model_type == 'random_forest':
            self.model = self.train_random_forest(X_train, y_train)
        elif model_type == 'xgboost':
            self.model = self.train_xgboost(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_name = model_type
        
        # Evaluate model
        self.metrics = self.evaluate_model(self.model, X_test, y_test)
        
        # Perform cross-validation with appropriate number of folds
        # Use min of 5 or n_samples-1 for cv folds
        cv_folds = min(5, n_samples - 1)
        if cv_folds >= 2:
            try:
                cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, 
                                        scoring='r2')
                self.metrics['cv_mean'] = cv_scores.mean()
                self.metrics['cv_std'] = cv_scores.std()
                print(f"CV Score: {self.metrics['cv_mean']:.4f} (+/- {self.metrics['cv_std']:.4f})")
            except Exception as e:
                print(f"Warning: Cross-validation skipped - {str(e)}")
                self.metrics['cv_mean'] = 0
                self.metrics['cv_std'] = 0
        else:
            print("Warning: Insufficient data for cross-validation")
            self.metrics['cv_mean'] = 0
            self.metrics['cv_std'] = 0
        
        print(f"Model Training Complete!")
        print(f"R² Score: {self.metrics['r2_score']:.4f}")
        print(f"RMSE: ${self.metrics['rmse']:,.2f}")
        print(f"MAE: ${self.metrics['mae']:,.2f}")
        
        return self.metrics

    def save_model(self, filepath: str = None):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if filepath is None:
            filepath = config.MODEL_PATH
        
        # Save model
        joblib.dump(self.model, filepath)
        
        # Save preprocessor
        preprocessor_path = filepath.replace('.joblib', '_preprocessor.joblib')
        joblib.dump(self.preprocessor, preprocessor_path)
        
        # Save to database
        self.db.save_training_history(
            self.model_name,
            self.metrics,
            str(config.MODEL_PARAMS.get(self.model_name, {}))
        )
        
        print(f"Model saved to {filepath}")
    
    def compare_models(self) -> pd.DataFrame:
        """Train and compare multiple models"""
        models = ['linear_regression', 'random_forest', 'xgboost']
        results = []
        
        for model_type in models:
            print(f"\n{'='*50}")
            print(f"Training {model_type}...")
            print(f"{'='*50}")
            
            metrics = self.train_and_evaluate(model_type)
            results.append({
                'Model': model_type,
                'R² Score': metrics['r2_score'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'CV Mean': metrics['cv_mean']
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(comparison_df.to_string(index=False))
        
        return comparison_df

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Train and compare all models
    comparison = trainer.compare_models()
    
    # Train best model and save
    trainer.train_and_evaluate('random_forest')
    trainer.save_model()
