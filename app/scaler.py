import os
import pickle
from sklearn.preprocessing import StandardScaler

SCALER_PATH = 'scalers/host_scaler.pkl'

def save_scaler_to_file(data):
    directory = os.path.dirname(SCALER_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory) 
        
    if os.path.exists(SCALER_PATH) and os.path.getsize(SCALER_PATH) > 0:
        return

    with open(SCALER_PATH, 'wb') as file:
        pickle.dump(data, file)


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        return StandardScaler()

    try:
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        return StandardScaler()