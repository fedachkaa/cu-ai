import os
import pandas as pd
import matplotlib.pyplot as plt

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
    os.makedirs('logs', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(features, coefficients)
    plt.xticks(rotation=20)
    plt.title("Feature Importance based on Coefficients")
    plt.savefig('logs/feature_importance.png')
    plt.close()

