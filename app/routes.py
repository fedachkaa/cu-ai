from flask import Blueprint, request, jsonify
from .train import train_model

blueprint = Blueprint('api', __name__)

@blueprint.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return jsonify({'message': 'predict route'})

@blueprint.route('/train', methods=['POST'])
def train():
    data = request.json
    
    if not data:
         return jsonify({'error': 'No data provided for training'}), 400
    
    success, message = train_model(data)
    
    if success:
        return jsonify({'message': message})
    return jsonify({'error': message}), 500
