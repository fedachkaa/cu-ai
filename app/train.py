import pandas as pd
import numpy as np
import random
from flask import current_app
from app.model import save_model, load_model
from app.utils.common_utils import save_data_to_file, save_scaler_to_file, load_scaler
from app.utils.host_data_utils import preprocess_host_data, get_host_features

def train_model(data):
    try:
        df = pd.DataFrame(data)
        df = preprocess_host_data(df)

        simulated_rows = simulate_unassigned_data(df)
        simulated_df = pd.DataFrame(simulated_rows)
        df = pd.concat([df, simulated_df], ignore_index=True)

        if (current_app.config["ENVIRONMENT"] == 'development'):
            save_data_to_file(df, 'output.xlsx')

        X = df[get_host_features()]
        y = df['assigned']

        scaler = load_scaler(current_app.config['HOST_SCALER_PATH'])

        X_scaled = scaler.fit_transform(X)

        model = load_model()
        model.partial_fit(X_scaled, y, classes=np.unique(y))
        save_model(model)
        save_scaler_to_file(scaler, current_app.config['HOST_SCALER_PATH'])

        return True, "Model training completed successfully."

    except Exception as e:
        return False, str(e)
    

def simulate_unassigned_data(df):
    simulated_rows = []
    for index, row in df.iterrows():
        simulated_row = row.copy()
        simulated_row['assigned'] = 0

        if index % 2 == 0:
            simulated_row['is_suspended_host'] = 0
        else:
            simulated_row['availability_match'] = 0

        if random.random() > 0.5:
            simulated_row['language_match'] = 1 - row['language_match']
            simulated_row['gender_match'] = 1 - row['gender_match']
            simulated_row['experience_match'] = 1 - row['experience_match']
            simulated_row['age_match'] = 1 - row['age_match']
            simulated_row['expertises_matched'] = 1 - row['expertises_matched']

            simulated_rows.append(simulated_row)

    return simulated_rows