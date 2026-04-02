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
    from utide import solve, reconstruct
    import numpy as np
    import pandas as pd
    t = pd.to_datetime(time_series).to_numpy() 
    u = wl_series.to_numpy(dtype=float)
    coef = solve(
        t, u,
        lat=latitude,
        method="ols",
        conf_int="linear",
        trend=False,        # Biar gak pusing sama kenaikan muka air rata-rata
        Rayleigh_min=1.0,   # Kunci biar gak 'empty'
    )

    # 3. Rekonstruksi
    rec = reconstruct(t, coef)
    
    return coef, rec.h
