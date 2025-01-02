import os
import joblib
from sklearn.linear_model import SGDClassifier

MODEL_PATH = 'models/most_suitable_host_model.pkl'

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    return model