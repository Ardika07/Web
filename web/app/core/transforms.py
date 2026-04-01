import numpy as np
import pandas as pd

def log_transform(series: pd.Series):
    """Biar sebaran datanya nggak terlalu jomplang, kita log-in aja!"""
    # Pake log1p (log(1+x)) biar aman kalau-kalau ada nilai 0 hihihi
    return np.log1p(series)

def min_max_normalize(series: pd.Series):
    """Jurus jitu biar semua nilainya cuma ada di rentang 0 sampai 1~"""
    return (series - series.min()) / (series.max() - series.min())

def differencing(series: pd.Series, periods: int = 1):
    """Buat ngilangin tren panjang biar datanya jadi stasioner nih!"""
    return series.diff(periods=periods)