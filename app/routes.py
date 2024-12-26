from flask import Blueprint, request, jsonify
from .train import train_model
from .prediction import make_prediction

blueprint = Blueprint('api', __name__)

@blueprint.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided for prediction'}), 400

    success, message, result = make_prediction(data)

    if success:
        return jsonify({'message': message, 'data': result})
    return jsonify({'error': message}), 500

@blueprint.route('/train', methods=['POST'])
def train():
    data = request.json
    
    if not data:
         return jsonify({'error': 'No data provided for training'}), 400
    
    success, message = train_model(data)
    
    if success:
        return jsonify({'message': message})
    return jsonify({'error': message}), 500
