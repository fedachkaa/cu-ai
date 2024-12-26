import pandas as pd

def calculate_suitability(row):
    score = 0

    if row['is_host_available']: 
        score += 5
    else:
        score -=1000

    if row['is_suspended_host']:
        score -= 1000

    if row['language'] in row['host_languages']:
        score += 3

    status_to_score = {
        'pro': 5,
        'super': 4,
        'savyy': 3,
        'rookie': 2,
        'newbie': 1
    }

    score += status_to_score.get(row['host_status'], 0)

    if row['parent_id'] in row['host_suitable_experiences']:
        score += 2

    if row['age'] != 'unknown' and isinstance(row['age'], dict):
        if row['age']['min'] <= row['host_age'] <= row['age']['max']:
            score += 3

    return score

def save_data_to_file(df):
    file_path = 'output.xlsx'

    try:
        try:
            df_existing = pd.read_excel(file_path)
        except FileNotFoundError:
            df.to_excel(file_path, index=False)
            return

        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_excel(file_path, index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
