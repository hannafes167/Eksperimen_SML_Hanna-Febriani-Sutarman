# preprocessing/automate_Hanna.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(input_path='heart.csv'):
    df = pd.read_csv(input_path)

    # Tentukan kolom
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Cek dan tangani outlier (optional, hanya cetak info)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR)))
    print("Outliers per kolom:\n", outliers_iqr.sum())

    # Split X dan y
    X = df.drop('target', axis=1)
    y = df['target']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerik saja
    X_train_to_scale = X_train[num_cols]
    X_test_to_scale = X_test[num_cols]
    X_train_not_scaled = X_train[cat_cols]
    X_test_not_scaled = X_test[cat_cols]

    scaler = StandardScaler()
    X_train_scaled_part = scaler.fit_transform(X_train_to_scale)
    X_test_scaled_part = scaler.transform(X_test_to_scale)

    X_train_scaled_df = pd.DataFrame(X_train_scaled_part, columns=num_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_part, columns=num_cols)

    X_train_scaled = pd.concat([X_train_scaled_df, X_train_not_scaled], axis=1)
    X_test_scaled = pd.concat([X_test_scaled_df, X_test_not_scaled], axis=1)

    # Urutkan kolom
    X_train_scaled = X_train_scaled[num_cols + cat_cols]
    X_test_scaled = X_test_scaled[num_cols + cat_cols]

    # Pastikan folder output ada
    os.makedirs("output", exist_ok=True)

    # Simpan ke file
    X_train_scaled.to_csv('output/X_train.csv', index=False)
    X_test_scaled.to_csv('output/X_test.csv', index=False)
    y_train.to_csv('output/y_train.csv', index=False)
    y_test.to_csv('output/y_test.csv', index=False)
    joblib.dump(scaler, 'output/scaler.pkl')

    print("✅ Dataset selesai diproses dan disimpan.")

if __name__ == "__main__":
    preprocess_data()
