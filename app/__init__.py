from flask import Flask

def create_app():
    app = Flask(__name__)
    
    with app.app_context():
        from .routes import blueprint
        app.register_blueprint(blueprint, url_prefix='/api')
    
    return app
