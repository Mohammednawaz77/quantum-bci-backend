import numpy as np

def segment_signal(raw, segment_duration=2.0):
    sfreq = raw.info["sfreq"]
    segment_samples = int(segment_duration * sfreq)
    data = raw.get_data()
    segments = []
    for start in range(0, data.shape[1], segment_samples):
        end = start + segment_samples
        if end <= data.shape[1]:
            seg = raw.copy().crop(tmin=start/sfreq, tmax=end/sfreq)
            segments.append(seg)
    return segments

def extract_band_power(segment, sfreq):
    data = segment.get_data()
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30)
    }
    power_features = []
    for ch in data:
        ch_band_powers = []
        for band, (low, high) in bands.items():
            psd, freqs = segment.compute_psd(method='welch', fmin=low, fmax=high).get_data(return_freqs=True)
            band_power = np.mean(psd)
            ch_band_powers.append(band_power)
        power_features.append(ch_band_powers)
    return power_features
