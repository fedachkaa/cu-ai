import pandas as pd
import numpy as np
import random
from flask import current_app
from app.model import save_model, load_model
from app.scaler import save_scaler_to_file, load_scaler
from app.utils.common_utils import save_data_to_file
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

        scaler = load_scaler()

        X_scaled = scaler.fit_transform(X)

        model = load_model()
        model.partial_fit(X_scaled, y, classes=np.unique(y))
        
        save_model(model)
        save_scaler_to_file(scaler)

        return True, "Model training completed successfully."

    except Exception as e:
        return False, str(e)
    

def simulate_unassigned_data(df):
    simulated_rows = []

    status_cycle = [
        {'is_suspended_host': 1, 'is_suitable_host': 0, 'other_bookings_matched': 0},
        {'is_suspended_host': 1, 'is_suitable_host': 1, 'other_bookings_matched': 0},
        {'is_suspended_host': 1, 'is_suitable_host': 0, 'other_bookings_matched': 1},
        {'is_suspended_host': 0, 'is_suitable_host': 0, 'other_bookings_matched': 0},
    ]

    for index, row in df.iterrows():
        simulated_row = row.copy()

        simulated_row['assigned'] = 0
        simulated_row['availability_match'] = 0
        
        status = status_cycle[index % len(status_cycle)]
        simulated_row.update(status)

        simulated_row['language_match'] = random.randint(0,1)
        simulated_row['gender_match'] = random.randint(0,1)
        simulated_row['experience_match'] = random.randint(0,1)
        simulated_row['age_match'] = random.randint(0,1)
        simulated_row['expertises_matched'] = random.randint(0,1)
        simulated_row['status_matched'] = random.randint(0, 1)
        simulated_row['ic_matched'] = random.randint(0, 1)

        simulated_rows.append(simulated_row)

    return simulated_rows