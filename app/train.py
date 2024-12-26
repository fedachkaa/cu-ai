import pandas as pd
from app.model import save_model, load_model
from app.utils import calculate_suitability, save_data_to_file
import numpy as np

def train_model(data):
    try:
        df = pd.DataFrame(data)

        df['language_match'] = df.apply(lambda x: int(x['language'] in x['host_languages']), axis=1)
        df['gender_match'] = df.apply(lambda x: int(x['host_gender'].lower() == x['gender'].lower()), axis=1)
        df['category_encoded'] = pd.factorize(df['category'])[0]
        df['budget'] = df['budget'].fillna(0)
        df['host_age'] = df['host_age'].fillna(df['host_age'].median())
        df['is_suspended_host'] = df['is_suspended_host'].fillna(0)
        df['is_host_available'] = df.apply(lambda row: 1 if row['date_start'] in row['availability_dates'] else 0, axis=1)
        df['suitability_score'] = df.apply(calculate_suitability, axis=1)
        df['suitability_class'] = (df['suitability_score'] > 8).astype(int)

        save_data_to_file(df)

        features = ['budget', 'is_suspended_host', 'gender_match', 'host_age', 'language_match', 'is_host_available']
        X = df[features]
        y = df['suitability_class']

        model = load_model()
        model.partial_fit(X, y, classes=np.unique(y))
        save_model(model)

        return True, "Model training completed successfully."

    except Exception as e:
        return False, str(e)