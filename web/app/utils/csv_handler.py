import pandas as pd
import numpy as np
import streamlit as st
import csv

def load_tide_data(filepath, time_col, wl_col):
    """
    Membaca data pasang surut dari CSV dengan proteksi delimiter 
    dan menghitung persentase data yang hilang.
    """
    # --- 1. JURUS PENGAMAN PEMBACAAN FILE ---
    try:
        # Coba deteksi otomatis (sep=None)
        # Kita tambahin on_bad_lines biar kalau ada baris aneh gak langsung crash
        df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='warn')
    except Exception:
        # Kalau gagal, putar balik ke awal file
        filepath.seek(0)
        # Kita coba pakai titik koma (;) dulu karena khas Excel Indonesia
        try:
            df = pd.read_csv(filepath)
        except Exception:
            # Kalau masih gagal, putar balik lagi dan paksa koma (,)
            filepath.seek(0)
            df = pd.read_csv(filepath, sep=',')
    
    # --- 2. VALIDASI KOLOM (Case Sensitive Protection) ---
    # Kita rapihin dikit: hapus spasi di nama kolom biar gak typo
    df.columns = df.columns.str.strip()
    
    if time_col not in df.columns or wl_col not in df.columns:
        available = list(df.columns)
        st.error(f"Aduuuh Sayang, kolom '{time_col}' atau '{wl_col}' gak nemu! 🥺")
        st.info(f"Kolom yang terdeteksi di file kamu: {available}")
        st.stop()

    # --- 3. PEMBERSIHAN & STANDARISASI DATA ---
    # Konversi ke datetime (penting buat UTide)
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    # Konversi ke angka
    df[wl_col] = pd.to_numeric(df[wl_col], errors='coerce')
    
    # Buang baris yang waktunya beneran gak bisa dibaca
    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)

    # --- 4. ANALISIS DATA KOSONG (MISSING DATA) ---
    total_rows = len(df)
    # Catat posisi mana aja yang aslinya kosong (NaN)
    df['is_missing'] = df[wl_col].isna()
    missing_count = df['is_missing'].sum()
    
    # Hitung persentase buat statistik di dashboard nanti
    missing_percent = (missing_count / total_rows) * 100 if total_rows > 0 else 0

    # --- 5. PENAMBALAN DATA (INTERPOLASI) ---
    # UTide & FFT butuh data yang rapat tanpa bolong (NaN)
    if missing_count > 0:
        # Interpolasi linear buat nambal titik yang kosong
        df[wl_col] = df[wl_col].interpolate(method='linear', limit_direction='both')
        
    return df, missing_percent
