import os
from dotenv import load_dotenv

load_dotenv()

class ProductionConfig:
    AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
    HOST_SCALER_PATH = 'scalers/host_scaler.pkl'