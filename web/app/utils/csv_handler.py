import pandas as pd
import streamlit as st

def load_tide_data(filepath, time_col, wl_col):
    """Fungsi buat ngebaca dan ngerapiin CSV pasut kamu nih~"""
    df = pd.read_csv(filepath)
    
    # --- PENGAMAN BARU BIAR NGGAK ERROR ---
    # Kita cek dulu, beneran ada nggak kolomnya di dalem CSV?
    if time_col not in df.columns:
        st.error(f"Aduuuh! Kolom waktu '{time_col}' nggak ketemu di CSV kamu. Yang ada cuma kolom ini nih: {list(df.columns)}. Coba benerin ketikannya di sidebar yaa! 🥺")
        st.stop() # Berhentiin programnya biar nggak crash
        
    if wl_col not in df.columns:
        st.error(f"Ihhh, kolom elevasi air '{wl_col}' juga nggak ada! Cek lagi yaa~ 😭")
        st.stop()
    # --------------------------------------
    
    # Pastiin kolom waktu formatnya datetime yaa!
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Urutin datanya dari waktu terlama ke terbaru
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Cek ada data yang bolong (NaN) nggak, kalau ada kita interpolasi aja
    if df[wl_col].isnull().any():
        df[wl_col] = df[wl_col].interpolate(method='linear')
        
    return df
