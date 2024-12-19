import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from app.model import save_model

def train_model(data):
    try:
        df = pd.DataFrame(data)

        df['language_match'] = df.apply(
            lambda x: int(x['language'] in x['host_languages']), axis=1
        )
        df['host_gender'] = df['host_gender'].map({'male': 0, 'female': 1})
        df['category_encoded'] = pd.factorize(df['category'])[0]

        features = ['budget', 'is_suspended_host', 'host_gender', 'host_age', 'language_match', 'category_encoded']
        X = df[features]
        y = df['assigned']

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        save_model(model)

        return True, "Model training completed successfully."

    except Exception as e:
        return False, str(e)
