import os
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | os.PathLike, sample_rate: int) -> tuple[np.ndarray, int]:
    """Read a mono audio proxy and peak-normalise it in memory.

    Only proxies are accepted: resampling and down-mixing are step 01's job.

    Args:
        path (str | os.PathLike): The audio file.
        sample_rate (int): The sample rate the file must have, in Hz.

    Returns:
        tuple[np.ndarray, int]: The 1-D float32 signal with a peak of 1.0, and its sample rate.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has another sample rate, more than one channel, no samples,
            or only silence."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    audio_data, file_rate = sf.read(path, dtype="float32")
    if file_rate != sample_rate:
        raise ValueError(
            f"Invalid sample rate received expected {sample_rate} got {file_rate} for file: {path}."
        )
    if len(audio_data.shape) > 1:
        raise ValueError(
            f"Invalid channel numbers received expected 1 got {audio_data.shape[1]} for file: {path}."
        )
    if audio_data.size == 0:
        raise ValueError(f"Empty audio file received: {path}.")
    peak = np.max(np.abs(audio_data))
    if peak == 0:
        raise ValueError(f"The provided file is silent: {path}")
    scaled_audio_data = audio_data / peak
    return scaled_audio_data, file_rate
