import pandas as pd
from app.model import save_model, load_model
from app.utils import calculate_suitability, save_data_to_file, calculate_match_expertises_percentage
import numpy as np

def train_model(data):
    try:
        df = pd.DataFrame(data)

        df['category_encoded'] = pd.factorize(df['category'])[0]
        df['budget'] = df['budget'].fillna(0)
        df['host_age'] = df['host_age'].fillna(df['host_age'].median())
        df['is_suspended_host'] = df['is_suspended_host'].fillna(0)

        df['language_match'] = df.apply(lambda x: int(x['language'] in x['host_languages']), axis=1)
        df['gender_match'] = df.apply(lambda x: int(x['host_gender'].lower() == x['gender'].lower()), axis=1)
        df['experience_match'] = df.apply(lambda x: int(x['parent_id'] in x['host_suitable_experiences']), axis=1)
        df['age_match'] = df.apply(lambda x: 1 if isinstance(x['age'], dict) and x['age'].get('min') <= x['host_age'] <= x['age'].get('max') else 0, axis=1)
        df['availability_match'] = df.apply(lambda row: 1 if row['date_start'] in row['availability_dates'] else 0, axis=1)
        df['expertises_matched'] = df.apply(
            lambda row: calculate_match_expertises_percentage(row['host_expertises'], row['activities']), axis=1
        )
        df['suitability_score'] = df.apply(calculate_suitability, axis=1)
        df['suitability_class'] = (df['suitability_score'] > 2300).astype(int)

        unique_classes = np.unique(df['suitability_class'])
        if len(unique_classes) <= 1:
            return False, f"Cannot train model, only one class found: {unique_classes[0]}."

        save_data_to_file(df, 'output.xlsx')

        features = ['is_suspended_host', 'language_match', 'gender_match', 'experience_match', 'age_match', 'availability_match', 'expertises_matched']
        X = df[features]
        y = df['suitability_class']

        model = load_model()
        model.partial_fit(X, y, classes=np.unique(y))
        save_model(model)

        return True, "Model training completed successfully."

    except Exception as e:
        return False, str(e)