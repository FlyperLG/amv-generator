from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True, slots=True)
class DenseFeatures:
    """Curves describing a song, sampled at the same fixed frame rate (~43 Hz).

    Each row of values is one moment in the song (about 23 ms) and each column is one feature.
    values[i, j] is feature names[j] at time times[i]. Which features there are, and whether
    they are normalised, depends on the function that made them: compute_dense_features
    (the 6 energy curves, normalised) or compute_timbre_features (MFCC and chroma, raw).

    Attributes:
        times (np.ndarray): The centre of each frame in seconds, shape [T].
        values (np.ndarray): The features, shape [T, F].
        names (tuple[str, ...]): The column names of values, length F."""

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


_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def compute_timbre_features(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    n_mfcc: int,
) -> DenseFeatures:
    """Compute the per-frame timbre and harmony of a song, for finding its sections.

    Kept apart from compute_dense_features: only the section finder uses these, and it
    z-scores them per bar itself, so they are returned raw. The frames are the same as
    compute_dense_features', so the two share times.

    Args:
        y (np.ndarray): The mono signal, shape [N]. Not modified.
        sample_rate (int): The sample rate of y in Hz.
        hop_length (int): The samples between two frames. Sets the frame rate.
        n_fft (int): The samples in each analysis window.
        n_mels (int): The number of mel bands the MFCCs are computed from.
        n_mfcc (int): The number of MFCCs computed, including the dropped mfcc_0.

    Returns:
        DenseFeatures: T = 1 + N // hop_length frames and these columns, in order:
            mfcc_1 .. mfcc_{n_mfcc-1}: The shape of the spectrum (timbre), from the log-mel
                spectrogram. mfcc_0, the overall loudness, is dropped, so a quiet and a loud
                copy give the same values.
            chroma_C, chroma_C#, .. chroma_B: How strongly each of the 12 pitch classes
                sounds, octaves folded together (harmony). Each frame is scaled so its
                strongest pitch class is 1. Computed without a tuning estimate, which would
                cost most of the time and doesn't move section boundaries."""
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=sample_rate, n_mels=n_mels)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
    mfcc = mfcc[1:]  # drop row 0 = loudness
    chroma = librosa.feature.chroma_stft(S=power, sr=sample_rate, n_fft=n_fft, tuning=0.0)
    times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sample_rate, hop_length=hop_length)

    names = (*(f"mfcc_{k}" for k in range(1, n_mfcc)), *(f"chroma_{p}" for p in _PITCH_CLASSES))
    raw = np.concatenate([mfcc, chroma], axis=0).T

    return DenseFeatures(
        times=times,
        values=raw,
        names=names,
    )
