import pandas as pd

def calculate_match_expertises_percentage(host_expertises, booking_expertises):
    if not isinstance(booking_expertises, dict) or not booking_expertises:
        return 1

    if not isinstance(host_expertises, dict) or not host_expertises:
        if not booking_expertises:
            return 1
        return 0

    total_matches = 0
    total_categories = 0

    for category, booking_items in booking_expertises.items():
        booking_expertises_set = set(booking_items)
        total_categories += len(booking_expertises_set)

        if category in host_expertises and booking_items:
            host_items = host_expertises.get(category, [])
            host_expertises_set = set(host_items)

            matches = booking_expertises_set & host_expertises_set
            total_matches += len(matches)

    if total_categories > 0:
        return (total_matches / total_categories)
    return 1

def preprocess_host_data(df):
    df['category_encoded'] = pd.factorize(df['category'])[0]
    df['budget'] = df['budget'].fillna(0)
    df['host_age'] = df['host_age'].fillna(df['host_age'].median())
    df['is_suspended_host'] = df['is_suspended_host'].fillna(0)
    df['language_match'] = df.apply(lambda x: int(x['language'] in x['host_languages']), axis=1)
    df['gender_match'] = df.apply(lambda x: int(x['gender'].lower() == x['host_gender'].lower()) if x['gender'].lower() in ['male', 'female'] else 0, axis=1)
    df['experience_match'] = df.apply(lambda x: int(x['parent_id'] in x['host_suitable_experiences']), axis=1)
    df['age_match'] = df.apply(lambda x: 1 if isinstance(x['age'], dict) and x['age'].get('min') <= x['host_age'] <= x['age'].get('max') else 0, axis=1)
    df['availability_match'] = df.apply(lambda row: 1 if row['date_start'] in row['availability_dates'] else 0, axis=1)
    df['expertises_matched'] = df.apply(lambda row: calculate_match_expertises_percentage(row['host_expertises'], row['activities']), axis=1)

    return df

def get_host_features():
    return [
        'is_suspended_host', 
        'language_match', 
        'gender_match', 
        'experience_match', 
        'age_match', 
        'availability_match', 
        'expertises_matched'
    ]
