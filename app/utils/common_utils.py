import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

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


def save_feature_importance(features, coefficients):
    plt.figure(figsize=(10, 6))
    plt.bar(features, coefficients)
    plt.xticks(rotation=20)
    plt.title("Feature Importance based on Coefficients")
    plt.savefig('feature_importance.png')
    plt.close()


def save_scaler_to_file(data, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory) 
        
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return

    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_scaler(filename):
    if not os.path.exists(filename):
        return StandardScaler()

    try:
        with open(filename, 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        return StandardScaler()

