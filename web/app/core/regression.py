import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def linear_fitting(x, y):
    """Cari garis regresi linear lurus dari datamu~"""
    # Derajat 1 itu khusus buat garis lurus (linear)
    coef = np.polyfit(x, y, 1)
    poly_func = np.poly1d(coef)
    
    return poly_func(x), coef

def poly_fitting(x, y, degree=2):
    """Kalau tren elevasi airnya melengkung, kita pakai regresi polinomial yaa!"""
    coef = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coef)
    
    return poly_func(x), coef

def interpolate_missing(x, y, x_new, kind='linear'):
    """Kalau instrumennya sempet mati dan data bolong, kita tambal pakai interpolasi ini!"""
    f = interp1d(x, y, kind=kind, fill_value="extrapolate")
    return f(x_new)