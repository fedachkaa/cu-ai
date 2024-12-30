from flask import Flask
import os
from dotenv import load_dotenv

def create_app():
    app = Flask(__name__)
    
    load_dotenv()
    environment = os.getenv('ENVIRONMENT', 'production').lower()

    if environment == 'development':
        from config.config_development import DevelopmentConfig
        app.config.from_object(DevelopmentConfig)
    elif environment == 'production':
        from config.config_production import ProductionConfig
        app.config.from_object(ProductionConfig)
    else:
        raise ValueError(f"Unknown environment: {environment}")

    with app.app_context():
        from .routes import blueprint
        app.register_blueprint(blueprint, url_prefix='/api')
    
    return app
