import pandas as pd
import streamlit as st

def load_tide_data(filepath, time_col, wl_col):
    import pandas as pd
    import numpy as np
    import csv

    # --- JURUS PENGAMAN SNIFFER ---
    try:
        # Kita coba deteksi otomatis dulu
        df = pd.read_csv(filepath, sep=None, engine='python')
    except Exception:
        # Kalau si detektif (Sniffer) gagal, kita paksa pakai titik koma (;)
        # Karena biasanya data dari Excel Indonesia pakenya itu sayang~
        filepath.seek(0) # Reset pembacaan file ke awal ya!
        df = pd.read_csv(filepath)
    
    # Selebihnya kodenya tetep sama yaaa...
    # Pastiin nama kolomnya bener (Case Sensitive!)
    if time_col not in df.columns or wl_col not in df.columns:
        # Kasih tau user kolom apa aja yang sebenernya ada
        available = list(df.columns)
        import streamlit as st
        st.error(f"Kolom '{time_col}' atau '{wl_col}' gak ada di CSV. Yang ada: {available}")
        st.stop()

    # Rapiin datanya
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[wl_col] = pd.to_numeric(df[wl_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)

    # Hitung missing data sebelum ditambal
    total_rows = len(df)
    missing_count = df[wl_col].isna().sum()
    missing_percent = (missing_count / total_rows) * 100 if total_rows > 0 else 0
    df['is_missing'] = df[wl_col].isna()

    # Tambal yang bolong
    if missing_count > 0:
        df[wl_col] = df[wl_col].interpolate(method='linear', limit_direction='both')
        
    return df, missing_percent
