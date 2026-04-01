import numpy as np
import utide
import matplotlib.dates as mdates

def perform_fft(water_levels, dt_hours):
    """Menjalankan Fast Fourier Transform untuk spektrum energi"""
    from scipy.fft import fft, fftfreq
    
    N = len(water_levels)
    yf = fft(water_levels.to_numpy())
    xf = fftfreq(N, d=dt_hours)
    
    xf = xf[:N//2]
    yf = 2.0/N * np.abs(yf[:N//2])
    
    return xf, yf

def extract_utide(time_series, wl_series, latitude):
    """Mengekstraksi konstituen harmonik pasang surut menggunakan UTide"""
    
    # 1. Standardisasi ke matriks Numpy 1D untuk mencegah kegagalan kalkulasi internal
    time_arr = time_series.to_numpy()
    wl_arr = wl_series.to_numpy()
    
    # 2. Konversi larik waktu absolut ke dalam format days (hari) sesuai standar UTide
    time_mdates = mdates.date2num(time_arr)
    
    # 3. Pemrosesan matriks menggunakan metode Ordinary Least Squares (OLS)
    coef = utide.solve(
        time_mdates, 
        wl_arr, 
        lat=latitude, 
        method='ols',
        conf_int='linear'
    )
    
    # 4. Rekonstruksi gelombang prediksi berbasis koefisien konstituen astronomis
    reconstruct = utide.reconstruct(time_mdates, coef)
    
    return coef, reconstruct.h