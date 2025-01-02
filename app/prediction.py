import pandas as pd
import json
from flask import current_app
from app.model import load_model
from app.utils.common_utils import save_data_to_file, save_feature_importance
from app.utils.host_data_utils import preprocess_host_data, get_host_features, set_feature_priorities

def make_prediction(data):
    try:
        df, X = prepare_booking_data(data['booking'], data['hosts'])

        model = load_model()
        
        if (current_app.config["ENVIRONMENT"] == 'development'):
            save_feature_importance(get_host_features(), model.coef_[0])
                
        df['predicted_suitability'] = model.predict_proba(X)[:, 1]
        sorted_hosts = df.sort_values(by='predicted_suitability', ascending=False)

        return True, 'Success', json.dumps(sorted_hosts[['host_id', 'predicted_suitability']].to_dict(orient='records'))

    except Exception as e:
        return False, str(e), []


def prepare_booking_data(booking_data, hosts):
    for host in hosts:
        host.update(booking_data)

    df = pd.DataFrame(hosts)

    df = preprocess_host_data(df)
    df = set_feature_priorities(df)

    if (current_app.config["ENVIRONMENT"] == 'development'):
        save_data_to_file(df, 'predict.xlsx')

    X = df[get_host_features()]
    
    return df, X