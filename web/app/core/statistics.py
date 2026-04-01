import numpy as np
import pandas as pd

def calculate_extremes(data_series: pd.Series):
    """Ngitung nilai ekstrem muka air laut (MSL, HHWL, LLWL)"""
    msl = np.mean(data_series)
    hhwl = np.max(data_series)
    llwl = np.min(data_series)
    
    return {
        "MSL": msl,
        "HHWL": hhwl,
        "LLWL": llwl
    }