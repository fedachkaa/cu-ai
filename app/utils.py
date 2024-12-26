import pandas as pd

def calculate_suitability(row):
    score = 0

    score += row['expertises_matched']

    if row['availability_match']: 
        score += 1000
    else:
        score -=1000

    if row['is_suspended_host']:
        score -= 1000
    else:
        score += 1000

    if row['language_match']:
        score += 300

    status_to_score = {
        'pro': 500,
        'super': 400,
        'savyy': 300,
        'rookie': 200,
        'newbie': 100
    }

    score += status_to_score.get(row['host_status'], 0)

    if row['experience_match']:
        score += 200

    if row['age_match']:
        score += 200

    return score

def save_data_to_file(df, file_path):
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


def calculate_match_expertises_percentage(host_expertises, booking_expertises):
    if not isinstance(booking_expertises, dict) or not booking_expertises:
        return 100

    if not isinstance(host_expertises, dict) or not host_expertises:
        if not booking_expertises:
            return 100
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
        return (total_matches / total_categories) * 100
    return 100
