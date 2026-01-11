from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from src.models.train_model import ModelTrainer
from src.models.predict_model import PricePredictor
from src.database.db_manager import DatabaseManager
import traceback

def create_routes(app):
    """Create and register Flask routes"""
    
    db = DatabaseManager()
    
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        """Prediction page"""
        if request.method == 'POST':
            try:
                # Get form data
                property_data = {
                    'area_sqft': float(request.form['area_sqft']),
                    'bedrooms': int(request.form['bedrooms']),
                    'bathrooms': int(request.form['bathrooms']),
                    'floors': int(request.form['floors']),
                    'age_years': int(request.form['age_years']),
                    'parking_spaces': int(request.form['parking_spaces']),
                    'distance_to_city_center_km': float(request.form['distance_to_city_center_km']),
                    'has_garden': int(request.form.get('has_garden', 0)),
                    'has_pool': int(request.form.get('has_pool', 0)),
                    'has_garage': int(request.form.get('has_garage', 0))
                }
                
                # Make prediction
                predictor = PricePredictor()
                result = predictor.get_price_range(property_data)
                
                # Save to database
                predictor.predict_and_save(property_data)
                
                return render_template('predict.html', 
                                     prediction=result,
                                     property_data=property_data)
            
            except Exception as e:
                flash(f'Error making prediction: {str(e)}', 'error')
                return render_template('predict.html', error=str(e))
        
        return render_template('predict.html')
    
    @app.route('/train', methods=['GET', 'POST'])
    def train():
        """Train model page"""
        if request.method == 'POST':
            try:
                model_type = request.form.get('model_type', 'random_forest')
                
                trainer = ModelTrainer()
                metrics = trainer.train_and_evaluate(model_type)
                trainer.save_model()
                
                flash(f'Model trained successfully! R² Score: {metrics["r2_score"]:.4f}', 'success')
                return render_template('index.html', metrics=metrics)
            
            except Exception as e:
                flash(f'Error training model: {str(e)}', 'error')
                traceback.print_exc()
        
        return redirect(url_for('index'))
    
    @app.route('/history')
    def history():
        """Prediction history page"""
        predictions = db.get_prediction_history(limit=50)
        training_history = db.get_training_history()
        
        return render_template('history.html', 
                             predictions=predictions,
                             training_history=training_history)
    
    @app.route('/add_property', methods=['POST'])
    def add_property():
        """Add new property to database"""
        try:
            property_data = {
                'area_sqft': float(request.form['area_sqft']),
                'bedrooms': int(request.form['bedrooms']),
                'bathrooms': int(request.form['bathrooms']),
                'floors': int(request.form['floors']),
                'age_years': int(request.form['age_years']),
                'parking_spaces': int(request.form['parking_spaces']),
                'distance_to_city_center_km': float(request.form['distance_to_city_center_km']),
                'has_garden': int(request.form.get('has_garden', 0)),
                'has_pool': int(request.form.get('has_pool', 0)),
                'has_garage': int(request.form.get('has_garage', 0)),
                'actual_price': float(request.form['actual_price']),
                'location': request.form.get('location', ''),
                'property_type': request.form.get('property_type', '')
            }
            
            property_id = db.add_property(property_data)
            flash(f'Property added successfully! ID: {property_id}', 'success')
        
        except Exception as e:
            flash(f'Error adding property: {str(e)}', 'error')
        
        return redirect(url_for('index'))
    
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """API endpoint for predictions"""
        try:
            data = request.get_json()
            
            predictor = PricePredictor()
            result = predictor.get_price_range(data)
            
            return jsonify({
                'success': True,
                'result': result
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
    
    @app.route('/api/properties', methods=['GET'])
    def api_get_properties():
        """API endpoint to get all properties"""
        try:
            properties = db.get_all_properties()
            return jsonify({
                'success': True,
                'properties': properties
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
