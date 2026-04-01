import numpy as np
import pandas as pd

def apply_hampel_filter(data_series: pd.Series, window_size: int = 9, n_sigma: float = 3.0) -> pd.Series:
    """
    Fungsi buat ngebersihin 'spike' atau data loncat di rekaman pasut kamu nih!
    Pake algoritma Hampel Filter biar datanya makin mulus~ ✨
    """
    # Kita bikin salinan datanya dulu biar data aslinya aman
    clean_series = data_series.copy()
    
    # Hitung nilai median dalam jendela waktu (rolling window)
    rolling_median = clean_series.rolling(window=window_size, center=True).median()
    
    # Hitung Median Absolute Deviation (MAD) biar tau seberapa jauh data nyimpang
    # Rumusnya: median(|x_i - median(x)|)
    mad = clean_series.rolling(window=window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    
    # Ambang batas buat nentuin data itu 'nakal' atau nggak (biasanya dikali 1.4826)
    threshold = n_sigma * 1.4826 * mad
    
    # Cari index mana aja yang nilainya lebih gede dari ambang batas (outliers)
    outlier_idx = np.abs(clean_series - rolling_median) > threshold
    
    # Ganti data yang 'nakal' tadi pake nilai median-nya. Taraaa! Jadi bersih deh!
    clean_series[outlier_idx] = rolling_median[outlier_idx]
    
    return clean_series