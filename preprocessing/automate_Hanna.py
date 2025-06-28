# preprocessing/automate_Hanna.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(input_path='heart.csv'):
    # Load dataset
    df = pd.read_csv(input_path)

    # Kolom numerik dan kategorikal
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Cek dan tampilkan jumlah outlier (opsional)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR)))
    print("Outliers per kolom:\n", outliers_iqr.sum())

    # Split fitur dan target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling numerik
    scaler = StandardScaler()
    X_train_scaled_part = scaler.fit_transform(X_train[num_cols])
    X_test_scaled_part = scaler.transform(X_test[num_cols])

    # Gabungkan dengan kolom kategori
    X_train_scaled_df = pd.DataFrame(X_train_scaled_part, columns=num_cols).reset_index(drop=True)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_part, columns=num_cols).reset_index(drop=True)
    X_train_cat = X_train[cat_cols].reset_index(drop=True)
    X_test_cat = X_test[cat_cols].reset_index(drop=True)

    X_train_final = pd.concat([X_train_scaled_df, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_scaled_df, X_test_cat], axis=1)

    # Simpan hasil ke folder preprocessing/outputs/
    output_dir = 'preprocessing/outputs'
    os.makedirs(output_dir, exist_ok=True)

    X_train_final.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test_final.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')

    print("✅ Dataset selesai diproses dan disimpan ke preprocessing/outputs/")

if __name__ == "__main__":
    preprocess_data()
