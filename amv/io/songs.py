import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from amv.config import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from amv.io.ffmpeg import generate_file_id, make_audio_proxy, probe
from amv.io.media import AudioInfo

logger = logging.getLogger(__name__)

COLUMNS = [
    "song_id",
    "kind",
    "path",
    "name",
    "audio_proxy_path",
    "duration",
    "sample_rate",
    "channels",
    "codec",
]
type SongKind = Literal["amv", "song"]


@dataclass(frozen=True, slots=True)
class AudioFile:
    """An AMV or song found and checked by discovery, ready to be turned into a wav."""

    path: Path  # resolved, absolute
    song_id: str  # generate_file_id(path)
    audio: AudioInfo  # probe(path).audio, i.e. of the original
    kind: SongKind


def _discover_valid_files(
    amv_dir: Path,
    song_dir: Path,
) -> list[AudioFile]:
    """Find, check and probe every AMV below amv_dir and every song below song_dir.

    Runs before any ffmpeg work, so that a bad file stops the run before any audio is
    extracted instead of halfway through. Each folder has its own suffixes, matched
    case-insensitively at any depth: VIDEO_EXTENSIONS in amv_dir, AUDIO_EXTENSIONS in
    song_dir. Everything else is ignored (an .mp3 lying in amv_dir is not an AMV, an
    .mp4 lying in song_dir is not a song).

    Args:
        amv_dir (Path): The folder with the reference AMVs.
        song_dir (Path): The folder with the input songs.

    Returns:
        list[AudioFile]: All AMVs, then all songs, each sorted by path.

    Raises:
        ValueError: If a file has no audio stream.
        RuntimeError: If ffprobe fails on a file, e.g. one that isn't media at all."""
    valid_files: list[AudioFile] = []
    targets: list[tuple[Path, SongKind, frozenset[str]]] = [
        (amv_dir, "amv", VIDEO_EXTENSIONS),
        (song_dir, "song", AUDIO_EXTENSIONS),
    ]
    for path, kind, extensions in targets:
        for file in tqdm(
            sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions),
            desc=f"Probing ({kind})",
            unit="file",
            dynamic_ncols=True,
            disable=None,
        ):
            media = probe(file)
            if media.audio is None:
                raise ValueError(f"Got file with no audio stream: {file}")
            song_id = generate_file_id(file)
            valid_files.append(AudioFile(file.resolve(), song_id, media.audio, kind))
    return valid_files


def build_song_table(
    amv_dir: str | os.PathLike,
    song_dir: str | os.PathLike,
    proxy_dir: str | os.PathLike,
    out_path: str | os.PathLike,
) -> pd.DataFrame:
    """Make the audio proxy of every AMV and song and write the song table (step 01, task 3.7).

    Every input gets a 22050 Hz mono wav, the only audio format step 02 reads. Work
    already done is skipped, so a re-run only extracts new or changed files. Files are
    processed one at a time with a progress bar. Step 02 iterates over this table and
    never lists proxy_dir itself. It does not read videos.parquet: an AMV's song_id
    equals its video_id there, because both come from generate_file_id.

    The table has one row per input and these columns (COLUMNS):
        song_id, kind, path, name: identity of the original (path is resolved);
            kind is "amv" or "song", from the folder the file was found in
        audio_proxy_path: the cached wav
        duration, sample_rate, channels, codec: of the original's audio stream, not
            of the wav (which is always 22050 Hz mono pcm)

    Args:
        amv_dir (str | os.PathLike): The folder with the reference AMVs.
        song_dir (str | os.PathLike): The folder with the input songs.
        proxy_dir (str | os.PathLike): The proxy cache; created if missing.
        out_path (str | os.PathLike): The parquet file to write; its folder is created if
            missing. It is written to a temp name first, so it is never left half-written.

    Returns:
        pd.DataFrame: The table, exactly as written to out_path.

    Raises:
        FileNotFoundError: If amv_dir or song_dir does not exist.
        ValueError: If a file has no audio stream. Raised before any extraction.
        RuntimeError: If ffmpeg or ffprobe fails on a file."""
    amv_dir = Path(amv_dir)
    song_dir = Path(song_dir)
    proxy_dir = Path(proxy_dir)
    out_path = Path(out_path)

    if not amv_dir.exists():
        raise FileNotFoundError(f"Input amv directory does not exist: {amv_dir}")
    if not song_dir.exists():
        raise FileNotFoundError(f"Input song directory does not exist: {song_dir}")

    valid_files = _discover_valid_files(amv_dir, song_dir)
    data: list[dict[str, Any]] = []
    with logging_redirect_tqdm():
        bar = tqdm(
            enumerate(valid_files, start=1),
            total=len(valid_files),
            desc="Building proxies",
            unit="file",
            dynamic_ncols=True,
            disable=None,
        )
        for i, file in bar:
            bar.set_postfix_str(file.song_id[:8])
            audio_proxy_path = make_audio_proxy(file.path, proxy_dir)

            row = {
                "song_id": file.song_id,
                "kind": file.kind,
                "path": str(file.path),
                "name": file.path.name,
                "audio_proxy_path": str(audio_proxy_path),
                "duration": file.audio.duration,
                "sample_rate": file.audio.sample_rate,
                "channels": file.audio.channels,
                "codec": file.audio.codec,
            }
            data.append(row)
            logger.debug(
                "[%d/%d] File %s was successfully processed and added to the song parquet table.",
                i,
                len(valid_files),
                file.path.name,
            )

    df = pd.DataFrame(
        data,
        columns=COLUMNS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)
    logger.info("Wrote %d songs to %s.", len(df), out_path)
    return df
