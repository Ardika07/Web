from scipy.signal import butter, filtfilt

def low_pass_filter(data, cutoff_freq, sample_rate, order=5):
    """Nyaring sinyal frekuensi tinggi pakai Butterworth Low-Pass Filter"""
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyq
    
    # Bikin desain filter-nya
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Terapin ke datanya
    filtered_data = filtfilt(b, a, data)
    return filtered_data