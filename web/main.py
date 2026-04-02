import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT SEMUA MODUL "OTAK" KITA ---
from app.utils.csv_handler import load_tide_data
from app.utils.outliers import apply_hampel_filter
from app.core.statistics import calculate_extremes
from app.core.timeseries import perform_fft, extract_utide
from app.core.smoothing import savitzky_golay_smoothing
from app.core.filtering import low_pass_filter
from app.core.regression import linear_fitting

# --- 1. SETUP HALAMAN ---
st.set_page_config(layout="wide", page_title="HydroTide Pro", page_icon="🌊")

# CSS sedikit biar tab-nya makin cantik dan rapi
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR (Pusat Kontrol) ---
with st.sidebar:
    st.header("⚙️ Station Settings")
    station_name = st.text_input("Station name", value="Teluk Kalabahi, Alor")
    
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=-8.330, format="%.3f")
    lon = col2.number_input("Longitude", value=124.500, format="%.3f")
    
    st.markdown("---")
    st.header("📂 Data & Columns")
    time_col = st.text_input("Time column", value="time")
    wl_col = st.text_input("Water level column", value="wl")
    sampling_dt = st.number_input("Sampling dt (hours)", value=1.00, step=0.50)
    
    st.markdown("---")
    st.header("🛠️ Processing Params")
    
    with st.expander("Outliers & Smoothing", expanded=False):
        hampel_window = st.number_input("Hampel window", value=9, step=2)
        hampel_sigma = st.number_input("Hampel n-sigma", value=3.00, step=0.10)
        sg_window = st.number_input("Savitzky-Golay Window", value=5, step=2)
        
    with st.expander("Low-Pass Filter", expanded=False):
        cutoff_freq = st.number_input("Cutoff Freq (cycles/hour)", value=0.05, step=0.01)
        
    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("🚀 Execute All Core Modules", type="primary", use_container_width=True)


# --- 3. HALAMAN UTAMA ---
st.title("🌊 HydroTide Pro — Integrated Analysis")
st.write("Platform komputasi oseanografi: Preprocessing, Spectral (FFT), Harmonik (UTide), & Regression.")

uploaded_file = st.file_uploader("Upload CSV Data Pasang Surut di sini yaa~", type=["csv"])

if process_btn:
    if uploaded_file is None:
        st.error("jangan lupa masukin data CSV-nya dulu ")
    else:
        with st.spinner("Mesin HydroTide sedang bekerja, tunggu sebentar... "):
            
            # ==========================================
            # EKSEKUSI SEMUA MODUL CORE
            # ==========================================
            # 1. Utils: Baca & Bersihin Spike
            df = load_tide_data(uploaded_file, time_col, wl_col)
            df['clean_wl'] = apply_hampel_filter(df[wl_col], int(hampel_window), float(hampel_sigma))
            
            # 2. Core: Smoothing & Filtering Lanjutan
            df['smooth_wl'] = savitzky_golay_smoothing(df['clean_wl'], int(sg_window), 2)
            # Asumsi sampling rate = 1/dt
            fs = 1.0 / sampling_dt 
            df['filtered_wl'] = low_pass_filter(df['smooth_wl'].dropna(), cutoff_freq, fs)
            
            # 3. Core: Statistics
            stats = calculate_extremes(df['clean_wl'])
            
            # 4. Core: Timeseries (FFT & UTide)
            # Pake dropna biar aman kalau ada sisa NaN
            valid_df = df.dropna(subset=[time_col, 'clean_wl']).copy()
            xf, yf = perform_fft(valid_df['clean_wl'], sampling_dt)
            
            # UTide butuh datetime mdates, kita panggil fungsinya
            try:
                utide_coef, utide_reconstruct = extract_utide(valid_df[time_col], valid_df['clean_wl'], lat)
                valid_df['utide_pred'] = utide_reconstruct
            except Exception as e:
                st.warning(f"UTide sedang bermasalah {e}")
                
            # 5. Core: Regression
            # Bikin waktu jadi array numerik buat X (contoh: urutan jam)
            x_num = np.arange(len(valid_df))
            trend_line, reg_coef = linear_fitting(x_num, valid_df['clean_wl'].to_numpy())
            
            st.success("Semua komputasi dari folder `app/core/` udah beres!")
            
            # ==========================================
            # UI TABS PRESENTATION
            # ==========================================
            tab1, tab2, tab3, tab4 = st.tabs([
                "🧹 1. Preprocessing & Stats", 
                "📈 2. Spectral (FFT)", 
                "🌀 3. Harmonic (UTide)", 
                "📏 4. Regression & Trends"
            ])
            
            # --- TAB 1: PREPROCESSING ---
            with tab1:
                st.subheader(f"📊 Summary Statistics - {station_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Sea Level (MSL)", f"{stats['MSL']:.3f} m")
                col2.metric("Highest High Water", f"{stats['HHWL']:.3f} m")
                col3.metric("Lowest Low Water", f"{stats['LLWL']:.3f} m")
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df[time_col], y=df[wl_col], mode='lines', name='Raw (Spikes)', line=dict(color='rgba(255, 99, 132, 0.4)')))
                fig1.add_trace(go.Scatter(x=df[time_col], y=df['clean_wl'], mode='lines', name='Cleaned (Hampel)', line=dict(color='#00b5ad')))
                fig1.add_trace(go.Scatter(x=df[time_col], y=df['filtered_wl'], mode='lines', name='Low-Pass Filter', line=dict(color='#FFD700', dash='dot')))
                
                fig1.update_layout(title="Elevasi Muka Air: Raw vs Preprocessed", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig1, use_container_width=True)

            # --- TAB 2: FFT SPECTRUM ---
            with tab2:
                st.subheader("⚡ Fast Fourier Transform (Energy Spectrum)")
                st.write("Mengubah sinyal domain waktu menjadi domain frekuensi untuk mencari periode puncak gelombang.")
                
                fig2 = go.Figure()
                # Hindari frekuensi 0 biar grafiknya bagus
                fig2.add_trace(go.Scatter(x=xf[1:], y=yf[1:], mode='lines', fill='tozeroy', line=dict(color='#8A2BE2')))
                fig2.update_layout(title="FFT Power Spectrum", xaxis_title="Frekuensi (cycles/hour)", yaxis_title="Amplitudo", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

            # --- TAB 3: UTIDE HARMONIC ---
            with tab3:
                st.subheader("🌊 UTide Reconstruction")
                if 'utide_pred' in valid_df.columns:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=valid_df[time_col], y=valid_df['clean_wl'], mode='lines', name='Observasi', line=dict(color='rgba(0, 181, 173, 0.5)')))
                    fig3.add_trace(go.Scatter(x=valid_df[time_col], y=valid_df['utide_pred'], mode='lines', name='UTide Prediksi', line=dict(color='#FF5733')))
                    
                    fig3.update_layout(title="Perbandingan Observasi vs Prediksi Pasang Surut", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Data UTide belum tersedia.")

            # --- TAB 4: REGRESSION ---
            with tab4:
                st.subheader("📏 Trend Analysis (Linear Regression)")
                st.write(f"Persamaan Garis: y = {reg_coef[0]:.6f}x + {reg_coef[1]:.6f}")
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=valid_df[time_col], y=valid_df['clean_wl'], mode='lines', name='Data Elevasi', line=dict(color='rgba(0, 181, 173, 0.5)')))
                fig4.add_trace(go.Scatter(x=valid_df[time_col], y=trend_line, mode='lines', name='Linear Trend', line=dict(color='#FF00FF', width=3)))
                
                fig4.update_layout(title="Analisis Tren Muka Air Jangka Panjang", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig4, use_container_width=True)

else:
    # Tampilan kosong kalau belum di-klik
    st.info("👈 Atur parameternya di sebelah kiri dulu, terus klik tombol Execute")
