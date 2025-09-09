# src/preprocessing.py
import mne
import numpy as np

def extract_features(file_path):
    """
    Read EDF, filter, and return channels, duration, sfreq and full features array.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    # band-pass same as before
    raw.filter(1., 50.)
    channels = raw.ch_names
    sfreq = raw.info.get("sfreq", None)
    duration = raw.n_times / sfreq if sfreq else raw.n_times
    data, _ = raw[:, :]  # numpy array shape (n_channels, n_samples)
    return {
        "channels": channels,
        "duration": duration,
        "sfreq": float(sfreq) if sfreq else None,
        "features": data
    }


async def stream_eeg_data(file_path, chunk_size=50, brain_state="Streaming"):
    """
    Async generator yielding consecutive chunks of the real EEG data.
    Yields JSON-serializable chunks (signals as lists).
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.filter(1., 50.)
    data, _ = raw[:, :]
    channels = raw.ch_names
    n_samples = data.shape[1]

    losses = []  # track simple epoch losses for display

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = data[:, start:end]

        slice_mean = np.mean(np.abs(chunk))
        losses.append(round(float(slice_mean), 6))

        mid = len(channels) // 2

        # use RMS for left/right (robust)
        left_val = float(np.sqrt(np.mean(np.square(chunk[:mid, :]))) if mid > 0 else 0.0)
        right_val = float(np.sqrt(np.mean(np.square(chunk[mid:, :]))) if (len(channels)-mid) > 0 else 0.0)

        focus = min(100, round(np.mean(np.abs(chunk)) * 1000, 2))
        stress = min(100, round(np.std(chunk) * 1000, 2))
        health = max(0, 100 - stress)

        yield {
            "channels": channels,
            "signals": chunk.tolist(),
            "brain_state": brain_state,
            "left_right": [left_val, right_val],
            "metrics": {"focus": focus, "stress": stress, "health": health},
            "epoch_losses": losses[-5:],
        }
