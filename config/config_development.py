import os
from dotenv import load_dotenv

load_dotenv()

class DevelopmentConfig:
    AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
    ENVIRONMENT = 'development'