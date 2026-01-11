from flask import Flask
import config
from src.api.routes import create_routes

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = config.SECRET_KEY
    
    # Register routes
    create_routes(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
