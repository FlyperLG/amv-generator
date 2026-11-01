from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True, slots=True)
class DenseFeatures:
    """Six curves describing a song, sampled at the same fixed frame rate (~43 Hz).

    Each row of values is one moment in the song (about 23 ms) and each column is one feature.
    values[i, j] is feature names[j] at time times[i]. Every value is relative to the same song:
    0 is typical for it, +1 is one IQR above typical and -1 is one IQR below.

    Attributes:
        times (np.ndarray): The centre of each frame in seconds, shape [T].
        values (np.ndarray): The normalised features, shape [T, F], where F = 6.
        names (tuple[str, ...]): The column names of values, length F:
            rms: Overall loudness.
            onset: How strongly new sounds start, on a perceptual (mel, dB) scale.
            flux: How much the raw spectrum gained since the previous frame, aligned with onset.
            low: Energy below low_band_hz (kick, bass).
            mid: Energy between the two cutoffs (vocals, melody).
            high: Energy above high_band_hz (hats, snare brightness)."""

    times: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]


def _robust_normalise(x: np.ndarray) -> np.ndarray:
    """Put each feature on a scale relative to its own typical value.

    Each column becomes (column - median) / IQR, where the IQR is the 75th minus the 25th
    percentile. A loud and a quiet song become comparable, and brief peaks barely affect the
    scale. A column with an IQR of 0 has no usable scale, even if it has spikes, and becomes
    all zeros.

    Args:
        x (np.ndarray): The raw features, shape [T, F]. Not modified.

    Returns:
        np.ndarray: The normalised features, shape [T, F]."""
    median = np.median(x, axis=0)
    percentile_25 = np.percentile(x, 25, axis=0)
    percentile_75 = np.percentile(x, 75, axis=0)
    iqr = percentile_75 - percentile_25
    safe_iqr = np.where(iqr == 0, 1, iqr)
    return np.where(iqr == 0, 0.0, (x - median) / safe_iqr)


def compute_dense_features(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    low_band_hz: float,
    high_band_hz: float,
) -> DenseFeatures:
    """Compute the normalised per-frame features of a song.

    Args:
        y (np.ndarray): The mono signal, shape [N].
        sample_rate (int): The sample rate of y in Hz.
        hop_length (int): The samples between two frames. Sets the frame rate.
        n_fft (int): The samples in each analysis window. Sets the frequency resolution.
        n_mels (int): The number of mel bands.
        low_band_hz (float): Mel bands centred below this count as low.
        high_band_hz (float): Mel bands centred above this count as high.

    Returns:
        DenseFeatures: The features, with T = 1 + N // hop_length frames."""
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    T = mag.shape[1]
    mel = librosa.feature.melspectrogram(S=mag**2, sr=sample_rate, n_mels=n_mels)
    rms = librosa.feature.rms(S=mag, frame_length=n_fft, hop_length=hop_length)
    rms = librosa.amplitude_to_db(rms).squeeze()
    onset = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel), sr=sample_rate, hop_length=hop_length, n_fft=n_fft
    )
    flux = np.diff(mag)
    flux = np.clip(flux, 0, None)
    flux = np.sum(flux, axis=0)
    flux = np.pad(flux, (1 + n_fft // (2 * hop_length), 0))[:T]

    centres = librosa.mel_frequencies(n_mels + 2, fmax=sample_rate / 2)[1:-1]
    is_low = centres < low_band_hz
    is_high = centres > high_band_hz
    is_mid = ~is_low & ~is_high
    low = librosa.power_to_db(mel[is_low].sum(axis=0))  # [bands, T] -> [T]
    mid = librosa.power_to_db(mel[is_mid].sum(axis=0))  # [bands, T] -> [T]
    high = librosa.power_to_db(mel[is_high].sum(axis=0))  # [bands, T] -> [T]

    times = librosa.frames_to_time(np.arange(T), sr=sample_rate, hop_length=hop_length)

    features = {"rms": rms, "onset": onset, "flux": flux, "low": low, "mid": mid, "high": high}
    raw = np.stack(list(features.values()), axis=1)

    return DenseFeatures(
        times=times,
        values=_robust_normalise(raw),
        names=tuple(features),
    )
