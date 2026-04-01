import pandas as pd
from scipy.signal import savgol_filter

def moving_average(series: pd.Series, window: int = 3):
    """Bikin datanya lebih halus pakai rata-rata bergerak~"""
    return series.rolling(window=window, center=True).mean()

def exponential_moving_average(series: pd.Series, span: int = 3):
    """Kalau EMA ini lebih peka sama data terbaru loh, makanya pake bobot!"""
    return series.ewm(span=span, adjust=False).mean()

def savitzky_golay_smoothing(series: pd.Series, window: int = 5, polyorder: int = 2):
    """Ini andalan banget buat ngilangin noise tanpa ngerusak puncak gelombang airnya!"""
    # Pastiin ukuran windownya ganjil yaa
    if window % 2 == 0:
        window += 1 
    
    smoothed = savgol_filter(series.dropna(), window_length=window, polyorder=polyorder)
    
    # Bikin jadi Pandas Series lagi biar rapi
    result = series.copy()
    result.loc[series.notna()] = smoothed
    return result   