import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle



COFFEE_COLS = [
    'Americano', 'Americano with Milk', 'Cappuccino',
    'Cocoa', 'Cortado', 'Espresso', 'Hot Chocolate', 'Latte'
]

COLS_TO_DROP_VIF = ['is_weekend', 'week_of_year', 'dow_5', 'dow_6']

SPLIT_TRAIN = '2024-12-01'
SPLIT_VAL   = '2025-02-01'


## load data
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV dan konversi kolom datetime."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"[load_data] Data loaded: {df.shape}")
    return df


## aggregation product per day
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi transaksi ke level harian per produk (wide format)."""
    daily = df.groupby([df['datetime'].dt.date, 'coffee_name']).size().unstack(fill_value=0)
    daily.index = pd.to_datetime(daily.index)

    for col in COFFEE_COLS:
        if col not in daily.columns:
            daily[col] = 0

    daily = daily[COFFEE_COLS]
    print(f"[aggregate_daily] Shape setelah agregasi: {daily.shape}")
    return daily


## handle missing dates
def handle_missing_dates(daily: pd.DataFrame) -> pd.DataFrame:
    """Isi hari kosong dengan 0 (hari tutup/libur)."""
    daily = daily.asfreq('D', fill_value=0)
    print(f"[handle_missing_dates] Total hari setelah reindex: {len(daily)}")
    return daily


## feature engineering
def feature_engineering(daily: pd.DataFrame) -> pd.DataFrame:
    """Buat fitur temporal, lag, dan rolling per produk."""

    ## temporal features
    daily['day_of_week']  = daily.index.dayofweek
    daily['is_weekend']   = daily['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    daily['month']        = daily.index.month
    daily['week_of_year'] = daily.index.isocalendar().week.astype(int)

    ## lag & rolling per produk
    for col in COFFEE_COLS:
        daily[f'{col}_lag1']       = daily[col].shift(1)
        daily[f'{col}_lag2']       = daily[col].shift(2)
        daily[f'{col}_lag7']       = daily[col].shift(7)
        daily[f'{col}_lag14']      = daily[col].shift(14)
        daily[f'{col}_roll_mean7'] = daily[col].shift(1).rolling(7).mean()
        daily[f'{col}_roll_std7']  = daily[col].shift(1).rolling(7).std()

    ## drop nan because of lag
    before = len(daily)
    daily = daily.dropna()
    print(f"[feature_engineering] Dropped {before - len(daily)} baris NaN. Shape: {daily.shape}")
    return daily


## encoding
def encode_features(daily: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding kolom day_of_week."""
    daily = pd.get_dummies(daily, columns=['day_of_week'], prefix='dow', dtype=int)
    print(f"[encode_features] Shape setelah encoding: {daily.shape}")
    return daily


## drop features that we not use
def drop_multicollinear(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop fitur dengan VIF tinggi berdasarkan hasil analisis EDA."""
    cols_to_drop = COLS_TO_DROP_VIF + \
                   [col for col in daily.columns if 'roll_mean7' in col]

    cols_to_drop = [c for c in cols_to_drop if c in daily.columns]
    daily = daily.drop(columns=cols_to_drop)
    print(f"[drop_multicollinear] Shape setelah drop: {daily.shape}")
    return daily


## train val test
def split_data(daily: pd.DataFrame):
    """Time-based split: train / val / test."""
    train = daily[daily.index < SPLIT_TRAIN]
    val   = daily[(daily.index >= SPLIT_TRAIN) & (daily.index < SPLIT_VAL)]
    test  = daily[daily.index >= SPLIT_VAL]

    print(f"[split_data] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


## scaling
def scale_features(train, val, test):
    """StandardScaler fit pada train, transform ke val & test."""
    feature_cols = [col for col in train.columns if col not in COFFEE_COLS]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val   = scaler.transform(val[feature_cols])
    X_test  = scaler.transform(test[feature_cols])

    y_train = train[COFFEE_COLS]
    y_val   = val[COFFEE_COLS]
    y_test  = test[COFFEE_COLS]

    print(f"[scale_features] X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols


## save output
def save_outputs(X_train, X_val, X_test, y_train, y_val, y_test,
                 scaler, feature_cols, output_dir: str = None):
    """Simpan semua output preprocessing ke folder."""
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'coffee_sales_preprocessing')

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train, columns=feature_cols).to_csv(f'{output_dir}/X_train.csv', index=False)
    pd.DataFrame(X_val,   columns=feature_cols).to_csv(f'{output_dir}/X_val.csv',   index=False)
    pd.DataFrame(X_test,  columns=feature_cols).to_csv(f'{output_dir}/X_test.csv',  index=False)

    y_train.to_csv(f'{output_dir}/y_train.csv')
    y_val.to_csv(f'{output_dir}/y_val.csv')
    y_test.to_csv(f'{output_dir}/y_test.csv')

    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"[save_outputs] Semua file tersimpan di folder '{output_dir}/'")


## main pipeline
def run_preprocessing(filepath: str, output_dir: str = None):
    print("=" * 50)
    print("AUTOMATE PREPROCESSING - COFFEE SALES")
    print("=" * 50)

    df      = load_data(filepath)
    daily   = aggregate_daily(df)
    daily   = handle_missing_dates(daily)
    daily   = feature_engineering(daily)
    daily   = encode_features(daily)
    daily   = drop_multicollinear(daily)

    train, val, test = split_data(daily)

    X_train, X_val, X_test, \
    y_train, y_val, y_test, \
    scaler, feature_cols = scale_features(train, val, test)

    save_outputs(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 scaler, feature_cols, output_dir)

    print("=" * 50)
    print("PREPROCESSING SELESAI")
    print("=" * 50)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_filepath = os.path.join(script_dir, '..', 'coffee_raw.csv')
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_filepath

    output_dir = os.path.join(script_dir, 'coffee_sales_preprocessing')
    run_preprocessing(filepath, output_dir)
