import pandas as pd
from flask import current_app
from app.model import load_model
from app.scaler import load_scaler
from app.utils.common_utils import save_data_to_file, save_feature_importance
from app.utils.host_data_utils import preprocess_host_data, get_host_features

def make_prediction(data):
    try:
        df, X = prepare_booking_data(data['booking'], data['hosts'])

        model = load_model()
        
        if (current_app.config["ENVIRONMENT"] == 'development'):
            save_feature_importance(get_host_features(), model.coef_[0])
                
        scaler = load_scaler()
        X_scaled = scaler.fit_transform(X)

        df['predicted_suitability'] = model.predict_proba(X_scaled)[:, 1]
        sorted_hosts = df.sort_values(by='predicted_suitability', ascending=False)

        return True, 'Success', sorted_hosts[['host_id', 'predicted_suitability']].to_dict(orient='records')

    except Exception as e:
        return False, str(e), []


def prepare_booking_data(booking_data, hosts):
    for host in hosts:
        host.update(booking_data)

    df = pd.DataFrame(hosts)

    df = preprocess_host_data(df)

    if (current_app.config["ENVIRONMENT"] == 'development'):
        save_data_to_file(df, 'predict.xlsx')

    X = df[get_host_features()]
    
    return df, X