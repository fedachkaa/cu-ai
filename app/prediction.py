import pandas as pd
from app.model import load_model
from app.utils import save_data_to_file, calculate_match_expertises_percentage
import json
import matplotlib.pyplot as plt

def make_prediction(data):
    try:
        df, X = prepare_booking_data(data['booking'], data['hosts'])

        model = load_model()

        coefficients = model.coef_[0]
        features = ['is_suspended_host', 'language_match', 'gender_match', 'experience_match', 'age_match', 'availability_match', 'expertises_matched']
        
        plt.figure(figsize=(10, 6))
        plt.bar(features, coefficients)
        plt.xticks(rotation=20)
        plt.title("Feature Importance based on Coefficients")
        plt.savefig('feature_importance.png')
        plt.close()
                
        predicted_probabilities = model.predict_proba(X)[:, 1]
        df['predicted_suitability'] = predicted_probabilities
        sorted_hosts = df.sort_values(by='predicted_suitability', ascending=False)

        return True, 'Success', json.dumps(sorted_hosts[['host_id', 'predicted_suitability']].to_dict(orient='records'))

    except Exception as e:
        return False, str(e), []



def prepare_booking_data(booking_data, hosts):
    for host in hosts:
        host.update(booking_data)

    df = pd.DataFrame(hosts)

    df['category_encoded'] = pd.factorize(df['category'])[0]
    df['budget'] = df['budget'].fillna(0)
    df['host_age'] = df['host_age'].fillna(df['host_age'].median())
    df['is_suspended_host'] = df['is_suspended_host'].fillna(0)

    df['language_match'] = df.apply(lambda x: int(x['language'] in x['host_languages']), axis=1)
    df['gender_match'] = df.apply(lambda x: int(x['host_gender'].lower() == x['gender'].lower()), axis=1)
    df['experience_match'] = df.apply(lambda x: int(x['parent_id'] in x['host_suitable_experiences']), axis=1)
    df['age_match'] = df.apply(lambda x: int(x['age'] != 'unknown' and x['age']['min'] <= x['host_age'] <= x['age']['max']), axis=1)
    df['availability_match'] = df.apply(lambda row: 1 if row['date_start'] in row['availability_dates'] else 0, axis=1)
    df['expertises_matched'] = df.apply(
        lambda row: calculate_match_expertises_percentage(row['host_expertises'], row['activities']), axis=1
    )

    save_data_to_file(df, 'predict.xlsx')
    features = ['is_suspended_host', 'language_match', 'gender_match', 'experience_match', 'age_match', 'availability_match', 'expertises_matched']
    X = df[features]
    
    return df, X