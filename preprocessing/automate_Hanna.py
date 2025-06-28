# preprocessing/automate_Hanna.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(input_path='heart.csv'):
    df = pd.read_csv(input_path)

    # Kolom numerik dan kategorikal
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Cek dan tangani outlier (optional)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR)))
    print("Outliers per kolom:\n", outliers_iqr.sum())

    # Pisahkan fitur dan target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling numerik
    X_train_scaled_part = StandardScaler().fit_transform(X_train[num_cols])
    X_test_scaled_part = StandardScaler().fit(X_train[num_cols]).transform(X_test[num_cols])  # optional: pakai scaler yang sama

    # Gabung kembali dengan fitur kategori
    X_train_scaled = pd.DataFrame(X_train_scaled_part, columns=num_cols).reset_index(drop=True)
    X_test_scaled = pd.DataFrame(X_test_scaled_part, columns=num_cols).reset_index(drop=True)
    X_train_cat = X_train[cat_cols].reset_index(drop=True)
    X_test_cat = X_test[cat_cols].reset_index(drop=True)

    X_train_final = pd.concat([X_train_scaled, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_scaled, X_test_cat], axis=1)

    # Simpan hasil
    os.makedirs('output', exist_ok=True)
    X_train_final.to_csv('output/X_train.csv', index=False)
    X_test_final.to_csv('output/X_test.csv', index=False)
    y_train.to_csv('output/y_train.csv', index=False)
    y_test.to_csv('output/y_test.csv', index=False)
    joblib.dump(StandardScaler().fit(X[num_cols]), 'output/scaler.pkl')

    print("✅ Dataset selesai diproses dan disimpan.")

if __name__ == "__main__":
    preprocess_data()
