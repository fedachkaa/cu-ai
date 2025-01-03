import pandas as pd
from datetime import datetime, timedelta

def preprocess_host_data(df, is_train=False):
    df['category_encoded'] = pd.factorize(df['category'])[0]
    df['budget'] = df['budget'].fillna(0)
    df['host_age'] = df['host_age'].fillna(df['host_age'].median())
    df['is_suspended_host'] = df['is_suspended_host'].fillna(0)
    df['language_match'] = df.apply(lambda x: int(x['language'] in x['host_languages']), axis=1)
    df['gender_match'] = df.apply(gender_match, axis=1)
    df['experience_match'] = df.apply(suitable_experiences_match, axis=1)
    df['age_match'] = df.apply(age_match, axis=1)
    df['availability_match'] = df.apply(availability_match, axis=1)
    df['expertises_matched'] = df.apply(calculate_match_expertises_percentage, axis=1)
    df['other_bookings_matched'] = df.apply(check_no_overlap_bookings, axis=1)
    df['status_matched'] = df.apply(status_match, axis=1)
    df['ic_matched'] = df.apply(lambda row: ic_match(row, is_train), axis=1)

    return df


def get_host_features():
    return [
        'is_suspended_host', 
        'language_match', 
        'gender_match', 
        'experience_match', 
        'age_match', 
        'availability_match', 
        'expertises_matched',
        'is_suitable_host',
        'other_bookings_matched',
        'status_matched',
        'ic_matched',
    ]

def gender_match(row):
    gender = row['gender'].lower()
    host_gender = row['host_gender'].lower()

    if gender in ['male', 'female']:
        return int(gender == host_gender)
    
    return 0


def suitable_experiences_match(row):
    if row['parent_id'] in row['host_suitable_experiences']:
        return 1
    if row['parent_id'] not in row['host_unsuitable_experiences']:
        return 1
    
    return 0


def age_match(row):
    if not isinstance(row['age'], dict):
        return 0
    if row['age'].get('min') <= row['host_age'] <= row['age'].get('max'):
        return 1
    
    return 0


def availability_match(row):
    if not len(row['availability_date']):
        return 0

    if row['availability_date'].get('date') != row['date_start']:
        return 0
    
    if not row['time_start']:
        return 1
    
    if row['availability_date'].get('time_intervals') is None:
        return 0
    
    booking_time_start = datetime.strptime(row['time_start'], "%H:%M")
    booking_time_end = booking_time_start + timedelta(minutes=row['duration_min'])

    is_start_in_interval = False
    is_end_in_interval = False

    for time_interval in row['availability_date'].get('time_intervals'):
        interval_start = datetime.strptime(time_interval['from'], "%H:%M")
        interval_end = datetime.strptime(time_interval['to'], "%H:%M")

        if not is_start_in_interval and booking_time_start >= interval_start and booking_time_start <= interval_end:
            is_start_in_interval = True

        if not is_end_in_interval and booking_time_end >= interval_start and booking_time_end <= interval_end:
            is_end_in_interval = True

        if not is_end_in_interval and is_start_in_interval and time_interval['to'] == '22:00':
            is_end_in_interval = True

    return int(is_start_in_interval and is_end_in_interval)


def calculate_match_expertises_percentage(row):
    host_expertises = row['host_expertises']
    booking_expertises = row['activities']
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


def check_no_overlap_bookings(row):
    if not row['time_start']:
        row['time_start'] = '00:00'

    new_booking_start = datetime.strptime(f"{row['date_start']} {row['time_start']}", "%Y-%m-%d %H:%M")
    new_booking_end = new_booking_start + timedelta(minutes=row['duration_min'])
    
    for booking in row['bookings_on_day']:
        if not booking['time_start']:
            booking['time_start'] = '00:00'

        existing_booking_start = datetime.strptime(f"{booking['date_start']} {booking['time_start']}", "%Y-%m-%d %H:%M")
        existing_booking_end = existing_booking_start + timedelta(minutes=booking['duration_min'])
        
        if (existing_booking_start > new_booking_start and existing_booking_start < new_booking_end):
            return 0
        
        if (existing_booking_end > new_booking_start and existing_booking_end < new_booking_end): 
            return 0
        
        if (existing_booking_start < new_booking_start and existing_booking_end > new_booking_end):
            return 0
    
    return 1


def status_match(row):
    if row['host_status'] != 'newbie':
        return 1
    if row['host_status'] == 'newbie' and row['budget'] == 0:
        return 1
    
    return 0


def ic_match(row, is_train):
    if is_train:
        return 1
    
    date_start = datetime.strptime(row['date_start'], "%Y-%m-%d")

    if date_start > datetime.now() + timedelta(days=2):
        return 1
    if date_start <= datetime.now() + timedelta(days=2) and row['host_instant_confirmation'] == 1:
        return 1
     
    return 0