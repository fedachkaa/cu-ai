from flask import Blueprint, request, jsonify, current_app
from functools import wraps
from .train import train_model
from .prediction import make_prediction

blueprint = Blueprint('api', __name__)

def check_authorisation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f'Bearer {current_app.config["AUTH_SECRET_KEY"]}':
            return jsonify({'error': 'Unauthorized'}), 401
        return func(*args, **kwargs)
    return wrapper

@blueprint.route('/predict', methods=['POST'])
@check_authorisation
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided for prediction'}), 400

    success, message, result = make_prediction(data)

    if success:
        return jsonify(result), 200
    return jsonify({'error': message}), 500

@blueprint.route('/train', methods=['POST'])
@check_authorisation
def train():
    data = request.json
    
    if not data:
         return jsonify({'error': 'No data provided for training'}), 400
    
    success, message = train_model(data)
    
    if success:
        return jsonify({'message': message})
    return jsonify({'error': message}), 500
