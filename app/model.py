import os
import joblib
from sklearn.linear_model import SGDClassifier

MODEL_PATH = 'models/model.pkl'

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else SGDClassifier()
    return model

def check_model_exists():
    return os.path.exists(MODEL_PATH)
