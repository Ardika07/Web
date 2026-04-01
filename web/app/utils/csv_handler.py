import pandas as pd
import streamlit as st

def load_tide_data(filepath, time_col, wl_col):
    import pandas as pd
    import numpy as np

    # Baca file (pake sep=None biar otomatis deteksi koma/titik koma)
    df = pd.read_csv(filepath, sep=None, engine='python')
    
    # Standarisasi kolom
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[wl_col] = pd.to_numeric(df[wl_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)

    # --- HITUNG MISSING DATA SEBELUM DITAMBAL ---
    total_rows = len(df)
    missing_count = df[wl_col].isna().sum()
    missing_percent = (missing_count / total_rows) * 100

    # Simpan nilai asli buat statistik nanti
    df['is_missing'] = df[wl_col].isna()

    # --- PROSES PENAMBALAN (INTERPOLASI) ---
    # Biar FFT & UTide nggak ngambek karena ada NaN
    if missing_count > 0:
        df[wl_col] = df[wl_col].interpolate(method='linear', limit_direction='both')
        
    return df, missing_percent
        
    return df
