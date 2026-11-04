"""Tests for amv.music.dense (docs/02-music-features, task 3.3).

Most tests use synthetic signals whose answer is known: a tone in one band, a tone that
starts at a known time, a quiet then a loud passage. Normalisation is only checked through
compute_dense_features, so these tests don't care whether robust_normalise is public or
private. Names are compared as tuples, so names may be a list or a tuple.

SPECIFICATION
  Settings are required arguments (Decision: dense.py does not read amv.config; the entry
  point reads the values from config and passes them in).
  (config: HOP_LENGTH = 512, N_FFT = 2048, N_MELS = 64, LOW_BAND_HZ = 200, HIGH_BAND_HZ = 4000)

  DenseFeatures: frozen dataclass
    times   [T]     float  the centre of each frame in seconds, frame * hop_length / sr
    values  [T, 6]  float  the normalised features, one column per name
    names   6 str          rms, onset, flux, low, mid, high, in the order of the columns

  compute_dense_features(y, sample_rate, hop_length, n_fft, n_mels, low_band_hz, high_band_hz)
    -> DenseFeatures
  - T = 1 + len(y) // hop_length. y is not modified.
  - rms: loudness in dB. Stays high during a sustained sound.
  - onset: onset_strength on the log-mel spectrogram. Spikes when a sound starts, then falls.
  - flux: the positive frame-to-frame gain of the linear STFT magnitude, summed over
    frequency. Spikes when a sound starts, then falls. Shifted later by the same
    1 + n_fft // (2 * hop_length) frames as onset_strength, so the two line up in time.
  - low / mid / high: summed mel power in dB, of the mel bands centred below low_band_hz,
    between the cutoffs, and above high_band_hz.

  Normalisation, per column and per song
  - (column - median) / IQR, where IQR = 75th - 25th percentile. So every column has a
    median of 0 and an IQR of 1, and the song's overall level has no effect.
  - A column with an IQR of 0 becomes all zeros (Decision: its spikes can't be put on the
    scale, and they must not become inf or a huge number). This is not exotic: a mostly
    silent signal has an IQR of 0 in spite of its spikes. No numpy warning is raised.

  compute_timbre_features(y, sample_rate, hop_length, n_fft, n_mels, n_mfcc) -> DenseFeatures
    (config: N_MFCC = 13; task 3.5 pools these per bar to find section boundaries)
  - A separate DenseFeatures, not extra columns of compute_dense_features (Decision: only
    the section finder uses them, and as extra columns they would go through the robust
    normalisation, smoothing and beat table meant for the 6 energy curves).
  - Same frames as compute_dense_features: identical times, T = 1 + len(y) // hop_length.
  - values [T, (n_mfcc - 1) + 12], NOT normalised (3.5 z-scores them per bar).
  - names, in column order:
      mfcc_1 .. mfcc_{n_mfcc-1}   librosa.feature.mfcc of the log-mel spectrogram (n_mels
                                  bands), with coefficient 0 dropped (Decision: it is the
                                  overall loudness, which 3.5 handles through rms instead).
                                  Numbered like librosa's rows, so mfcc_1 is row 1.
      chroma_C, chroma_C#, chroma_D, chroma_D#, chroma_E, chroma_F, chroma_F#, chroma_G,
      chroma_G#, chroma_A, chroma_A#, chroma_B
                                  librosa.feature.chroma_stft of the power spectrogram, rows
                                  in librosa's order (row 0 is C). Each frame is scaled so
                                  its largest pitch class is 1, so the values are in [0, 1].
  - chroma uses tuning=0.0 (Decision: librosa's tuning estimate takes ~320 ms of a ~450 ms
    call on a 3-min song, and a song-wide tuning offset doesn't move section boundaries).
  - Because mfcc_0 is dropped and chroma is scaled per frame, a quiet and a loud copy give
    the same values. y is not modified.
"""

import dataclasses
import time
import warnings

import numpy as np
import pytest

from amv.music.audio import load_audio
from amv.music.dense import DenseFeatures, compute_dense_features, compute_timbre_features

SR = 22050
HOP = 512
N_FFT = 2048
N_MELS = 64
LOW_HZ = 200
HIGH_HZ = 4000
NAMES = ("rms", "onset", "flux", "low", "mid", "high")


def features(y, sample_rate=SR, hop_length=HOP, n_fft=N_FFT, n_mels=N_MELS, low=LOW_HZ, high=HIGH_HZ):
    return compute_dense_features(y, sample_rate, hop_length, n_fft, n_mels, low, high)


def column(dense, name):
    return dense.values[:, tuple(dense.names).index(name)]


def tone(freq, seconds=1.0, amplitude=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def mean_between(dense, name, start, end):
    """The mean of a feature between two times in seconds."""
    inside = (dense.times >= start) & (dense.times < end)
    return column(dense, name)[inside].mean()


# --- Output shape -----------------------------------------------------------------


def test_returns_dense_features(drum_loop):
    y, _ = drum_loop()

    assert isinstance(features(y), DenseFeatures)


def test_dense_features_is_frozen(drum_loop):
    y, _ = drum_loop()
    dense = features(y)

    with pytest.raises(dataclasses.FrozenInstanceError):
        dense.values = np.zeros(1)


def test_names_label_the_six_columns_in_order(drum_loop):
    y, _ = drum_loop()

    assert tuple(features(y).names) == NAMES


@pytest.mark.parametrize("n_samples", [SR, SR + 1, 3 * SR + 100])
def test_shapes_follow_the_signal_length(n_samples):
    y = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32)

    dense = features(y)

    expected_frames = 1 + n_samples // HOP
    assert dense.times.shape == (expected_frames,)
    assert dense.values.shape == (expected_frames, len(NAMES))


def test_hop_length_sets_the_number_of_frames():
    y = tone(440, seconds=2.0)

    assert features(y, hop_length=256).times.shape == (1 + len(y) // 256,)


def test_times_are_the_frame_centres():
    y = tone(440, seconds=2.0)

    dense = features(y)

    np.testing.assert_allclose(dense.times, np.arange(len(dense.times)) * HOP / SR)


def test_does_not_modify_the_signal(drum_loop):
    y, _ = drum_loop()
    original = y.copy()

    features(y)

    np.testing.assert_array_equal(y, original)


def test_every_column_is_a_different_feature(drum_loop):
    """Catches a feature computed from the wrong variable, such as high from the low mask."""
    y, _ = drum_loop()
    dense = features(y)

    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            assert not np.array_equal(dense.values[:, i], dense.values[:, j]), (NAMES[i], NAMES[j])


# --- Bands ------------------------------------------------------------------------


@pytest.fixture
def three_tones():
    """1 s at 100 Hz, then 1 s at 1 kHz, then 1 s at 8 kHz: one tone per band."""
    return features(np.concatenate([tone(100), tone(1000), tone(8000)]))


@pytest.mark.parametrize(
    ("name", "section"),
    [("low", 0), ("mid", 1), ("high", 2)],
)
def test_each_band_is_highest_during_its_own_tone(three_tones, name, section):
    means = [mean_between(three_tones, name, s + 0.2, s + 0.8) for s in range(3)]

    assert int(np.argmax(means)) == section, means


@pytest.mark.parametrize(
    ("low", "high", "band_with_tone"),
    [
        (1000, 4000, "low"),  # 500 Hz is below both cutoffs
        (200, 4000, "mid"),
        (100, 300, "high"),  # 500 Hz is above both cutoffs
    ],
)
def test_cutoffs_decide_the_band(low, high, band_with_tone):
    """2 s of a faint 6 kHz tone, then 2 s of a 500 Hz tone. Only one band rises most with it."""
    y = np.concatenate([tone(6000, seconds=2.0, amplitude=0.001), tone(500, seconds=2.0)])

    dense = features(y, low=low, high=high)

    rises = {
        name: mean_between(dense, name, 2.2, 3.8) - mean_between(dense, name, 0.2, 1.8) for name in NAMES[3:]
    }
    assert max(rises, key=rises.get) == band_with_tone, rises


# --- Loudness and attacks ---------------------------------------------------------


def test_rms_follows_loudness():
    y = np.concatenate([tone(440, amplitude=0.01), tone(440, amplitude=0.5)])

    dense = features(y)

    assert mean_between(dense, "rms", 1.2, 1.8) > mean_between(dense, "rms", 0.2, 0.8)


@pytest.fixture
def tone_starting_at_one_second():
    """1 s of silence, then a tone that lasts 2 s, then 1 s of silence."""
    silence = np.zeros(SR, np.float32)
    return features(np.concatenate([silence, tone(1000, seconds=2.0), silence]))


@pytest.mark.parametrize("name", ["onset", "flux"])
def test_attack_features_spike_when_the_sound_starts(tone_starting_at_one_second, name):
    dense = tone_starting_at_one_second

    peak_time = dense.times[np.argmax(column(dense, name))]

    assert peak_time == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("name", ["onset", "flux"])
def test_attack_features_fall_while_the_sound_lasts(tone_starting_at_one_second, name):
    dense = tone_starting_at_one_second

    sustained = mean_between(dense, name, 1.3, 2.7)

    assert sustained < 0.1 * column(dense, name).max()


def test_rms_stays_high_while_the_sound_lasts(tone_starting_at_one_second):
    dense = tone_starting_at_one_second

    assert mean_between(dense, "rms", 1.3, 2.7) > mean_between(dense, "rms", 0.2, 0.8)


def test_onset_peaks_land_on_the_beats(drum_loop):
    y, beats = drum_loop()
    dense = features(y)
    onset = column(dense, "onset")

    for beat in beats:
        near = np.abs(dense.times - beat) <= 0.05
        assert onset[near].max() > 1.0, beat


# --- Normalisation ----------------------------------------------------------------


def test_each_column_has_median_zero_and_iqr_one():
    y = np.random.default_rng(0).standard_normal(5 * SR).astype(np.float32)

    values = features(y).values

    np.testing.assert_allclose(np.median(values, axis=0), 0.0, atol=1e-5)
    iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    np.testing.assert_allclose(iqr, 1.0, atol=1e-5)


def test_a_quiet_and_a_loud_copy_give_the_same_features(drum_loop):
    """With a noise floor: librosa's absolute -100 dB floor shifts the digital silence of a quiet copy."""
    y, _ = drum_loop()
    y = y + 1e-3 * np.random.default_rng(0).standard_normal(len(y)).astype(np.float32)

    loud = features(y).values
    quiet = features(0.1 * y).values

    np.testing.assert_allclose(quiet, loud, atol=1e-2)


@pytest.fixture
def mostly_silent():
    """A single click in 10 s of digital silence, so most columns have an IQR of 0."""
    y = np.zeros(10 * SR, np.float32)
    y[5 * SR] = 1.0
    return y


def test_a_column_with_zero_iqr_becomes_all_zeros(mostly_silent):
    """The click is non-zero, so x / 0 gives inf here, not the nan of 0 / 0."""
    dense = features(mostly_silent)

    np.testing.assert_array_equal(column(dense, "flux"), 0.0)


def test_values_are_always_finite_and_on_scale(mostly_silent):
    values = features(mostly_silent).values

    assert np.isfinite(values).all()
    assert np.abs(values).max() < 1e6


def test_zero_iqr_raises_no_numpy_warning(mostly_silent):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        features(mostly_silent)


# --- Timbre features --------------------------------------------------------------

N_MFCC = 13
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TIMBRE_NAMES = (*(f"mfcc_{k}" for k in range(1, N_MFCC)), *(f"chroma_{p}" for p in PITCHES))


def timbre(y, sample_rate=SR, hop_length=HOP, n_fft=N_FFT, n_mels=N_MELS, n_mfcc=N_MFCC):
    return compute_timbre_features(y, sample_rate, hop_length, n_fft, n_mels, n_mfcc)


def chroma_of(feats):
    """[T, 12] chroma block, by name."""
    return np.stack([column(feats, f"chroma_{p}") for p in PITCHES], axis=1)


def test_timbre_returns_dense_features(drum_loop):
    y, _ = drum_loop()

    assert isinstance(timbre(y), DenseFeatures)


def test_timbre_names_are_mfcc_1_up_then_the_twelve_pitch_classes(drum_loop):
    y, _ = drum_loop()

    assert tuple(timbre(y).names) == TIMBRE_NAMES


def test_timbre_n_mfcc_sets_the_number_of_mfcc_columns(drum_loop):
    y, _ = drum_loop()

    feats = timbre(y, n_mfcc=20)

    assert tuple(feats.names)[:19] == tuple(f"mfcc_{k}" for k in range(1, 20))
    assert feats.values.shape[1] == 19 + 12


def test_timbre_uses_the_same_frames_as_the_dense_features(drum_loop):
    y, _ = drum_loop()

    feats = timbre(y)

    np.testing.assert_array_equal(feats.times, features(y).times)
    assert feats.values.shape == (1 + len(y) // HOP, len(TIMBRE_NAMES))


def test_timbre_does_not_modify_the_signal(drum_loop):
    y, _ = drum_loop()
    original = y.copy()

    timbre(y)

    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize(
    ("freq", "pitch"),
    [(261.63, "C"), (329.63, "E"), (440.0, "A"), (493.88, "B"), (880.0, "A")],
)
def test_timbre_chroma_names_the_pitch_of_a_tone(freq, pitch):
    """880 Hz is A an octave up: chroma folds octaves together."""
    feats = timbre(tone(freq, seconds=2.0))

    middle = chroma_of(feats)[20:-20].mean(axis=0)

    assert PITCHES[int(np.argmax(middle))] == pitch, dict(zip(PITCHES, middle.round(2)))


def test_timbre_chroma_is_scaled_per_frame_not_normalised():
    """Raw chroma_stft: every frame's largest pitch class is 1. A robust-normalised column
    would have median 0 and go negative."""
    y = np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32)

    chroma = chroma_of(timbre(y))

    assert chroma.min() >= 0.0
    np.testing.assert_allclose(chroma.max(axis=1), 1.0, atol=1e-6)


def test_timbre_mfcc_1_tells_a_low_from_a_high_sound():
    """mfcc_1 measures spectral tilt: high for a dark sound, low for a bright one."""
    feats = timbre(np.concatenate([tone(100, seconds=1.0), tone(6000, seconds=1.0)]))

    assert mean_between(feats, "mfcc_1", 0.2, 0.8) > mean_between(feats, "mfcc_1", 1.2, 1.8)


def test_timbre_a_quiet_and_a_loud_copy_give_the_same_values(drum_loop):
    """Only mfcc_0 depends on the gain, and it is dropped. Noise floor: see the dense test."""
    y, _ = drum_loop()
    y = y + 1e-3 * np.random.default_rng(0).standard_normal(len(y)).astype(np.float32)

    loud = timbre(y).values
    quiet = timbre(0.1 * y).values

    np.testing.assert_allclose(quiet, loud, atol=1e-2)


def test_timbre_values_are_finite(mostly_silent):
    assert np.isfinite(timbre(mostly_silent).values).all()


def test_timbre_skips_the_tuning_estimate():
    """With tuning=0.0, MFCC + chroma cost little next to the STFT itself; librosa's tuning
    estimate alone takes ~3x the STFT. Best of 3, so a busy machine doesn't fail it."""
    import librosa

    y = np.random.default_rng(0).standard_normal(120 * SR).astype(np.float32)
    timbre(y)  # warm-up: librosa compiles and caches on the first call

    def best_of_3(f):
        best = np.inf
        for _ in range(3):
            start = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - start)
        return best

    stft = best_of_3(lambda: librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    full = best_of_3(lambda: timbre(y))

    assert full < 3 * stft, f"timbre {full * 1000:.0f} ms vs stft {stft * 1000:.0f} ms"


# --- Config -----------------------------------------------------------------------


def test_config_has_the_planned_settings():
    import amv.config as config

    assert config.HOP_LENGTH == HOP
    assert config.N_FFT == N_FFT
    assert config.N_MELS == N_MELS
    assert config.LOW_BAND_HZ == LOW_HZ
    assert config.HIGH_BAND_HZ == HIGH_HZ
    assert config.N_MFCC == N_MFCC


# --- Real song (integration) ------------------------------------------------------


@pytest.mark.integration
def test_bluelock_features_are_well_formed(bluelock_wav):
    y, sr = load_audio(bluelock_wav, SR)

    dense = features(y, sample_rate=sr)

    assert dense.values.shape == (1 + len(y) // HOP, len(NAMES))
    assert np.isfinite(dense.values).all()
    np.testing.assert_allclose(np.median(dense.values, axis=0), 0.0, atol=1e-5)


@pytest.mark.integration
def test_bluelock_onset_and_flux_line_up_in_time(bluelock_wav):
    """Unshifted, flux correlates best with onset 2 frames (46 ms) away. A synthetic loop can't show this."""
    y, sr = load_audio(bluelock_wav, SR)

    dense = features(y, sample_rate=sr)
    onset, flux = column(dense, "onset"), column(dense, "flux")

    aligned = np.corrcoef(onset, flux)[0, 1]
    flux_later = np.corrcoef(onset[2:], flux[:-2])[0, 1]
    flux_earlier = np.corrcoef(onset[:-2], flux[2:])[0, 1]
    assert aligned > flux_later
    assert aligned > flux_earlier
