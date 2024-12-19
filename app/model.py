import os
import joblib

MODEL_PATH = 'models/model.pkl'

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

def check_model_exists():
    return os.path.exists(MODEL_PATH)
