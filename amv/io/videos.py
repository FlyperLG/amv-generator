import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from amv.config import VIDEO_EXTENSIONS
from amv.io.ffmpeg import generate_file_id, make_audio_proxy, make_video_proxy, probe
from amv.io.media import VideoInfo

logger = logging.getLogger(__name__)

COLUMNS = [
    "video_id",
    "kind",
    "series",
    "path",
    "name",
    "video_proxy_path",
    "audio_proxy_path",
    "duration",
    "fps",
    "is_vfr",
    "width",
    "height",
    "n_frames",
    "codec",
    "proxy_width",
    "proxy_height",
]
type Kind = Literal["amv", "library"]


@dataclass(frozen=True, slots=True)
class VideoFile:
    """A source video found and checked by discovery, ready to be turned into proxies."""

    path: Path  # resolved, absolute
    video_id: str  # generate_file_id(path)
    video: VideoInfo  # probe(path).video, i.e. of the original
    kind: Kind
    series: str | None  # first folder below library_dir; None for AMVs


def _discover_valid_files(
    amv_dir: Path,
    library_dir: Path,
) -> list[VideoFile]:
    """Find, check and probe every video below amv_dir and library_dir.

    Runs before any ffmpeg work, so that a bad file stops the run before hours of
    transcoding instead of after them. Files whose suffix is in VIDEO_EXTENSIONS
    (case-insensitive) are found at any depth; everything else is ignored.

    Args:
        amv_dir (Path): The folder with the reference AMVs.
        library_dir (Path): The folder with the source footage, one sub-folder per series.

    Returns:
        list[VideoFile]: All AMVs, then all library videos, each sorted by path.

    Raises:
        ValueError: If a library video is not inside a series folder, or a file has no
            video stream.
        RuntimeError: If ffprobe fails on a file."""
    valid_files: list[VideoFile] = []
    targets: list[tuple[Path, Kind]] = [(amv_dir, "amv"), (library_dir, "library")]
    for path, kind in targets:
        for file in tqdm(
            sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
            desc=f"Probing ({kind})",
            unit="file",
            dynamic_ncols=True,
            disable=None,
        ):
            series: str | None = None
            if kind == "library":
                relative_file_parts = file.relative_to(library_dir).parts
                if len(relative_file_parts) <= 1:
                    raise ValueError(f"Library file is not inside a series folder: {file}")
                series = relative_file_parts[0]
            media = probe(file)
            if media.video is None:
                raise ValueError(f"Got file with no video stream: {file}")
            video_id = generate_file_id(file)
            valid_files.append(VideoFile(file.resolve(), video_id, media.video, kind, series))
    return valid_files


def build_video_table(
    amv_dir: str | os.PathLike,
    library_dir: str | os.PathLike,
    proxy_dir: str | os.PathLike,
    out_path: str | os.PathLike,
) -> pd.DataFrame:
    """Make the proxies of every source video and write the video table (step 01, task 3.6).

    Every video gets a video proxy; AMVs also get a wav (library videos don't need one).
    Work already done is skipped, so a re-run only transcodes new or changed files.
    Videos are processed one at a time with a progress bar. Later steps select videos
    through this table (kind == "amv" / "library") and never list proxy_dir themselves.

    The table has one row per video and these columns (COLUMNS):
        video_id, kind, series, path, name: identity of the original (path is resolved)
        video_proxy_path, audio_proxy_path: the cached proxies (audio is None for library)
        duration, fps, is_vfr, width, height, n_frames, codec: of the original, not the
            proxy; fps is an exact fraction string ("30000/1001"), n_frames a nullable Int64
        proxy_width, proxy_height: the proxy's size, as iter_frames needs it

    Args:
        amv_dir (str | os.PathLike): The folder with the reference AMVs.
        library_dir (str | os.PathLike): The folder with the source footage, one
            sub-folder per series (e.g. library_dir/Blue Lock/S01E01.mkv).
        proxy_dir (str | os.PathLike): The proxy cache; created if missing.
        out_path (str | os.PathLike): The parquet file to write; its folder is created if
            missing. It is written to a temp name first, so it is never left half-written.

    Returns:
        pd.DataFrame: The table, exactly as written to out_path.

    Raises:
        FileNotFoundError: If amv_dir or library_dir does not exist.
        ValueError: If a library video is not inside a series folder, or a file has no
            video stream. Raised before any transcoding.
        RuntimeError: If ffmpeg or ffprobe fails on a file."""
    amv_dir = Path(amv_dir)
    library_dir = Path(library_dir)
    proxy_dir = Path(proxy_dir)
    out_path = Path(out_path)

    if not amv_dir.exists():
        raise FileNotFoundError(f"Input amv directory does not exist: {amv_dir}")
    if not library_dir.exists():
        raise FileNotFoundError(f"Input library directory does not exist: {library_dir}")

    valid_files = _discover_valid_files(amv_dir, library_dir)
    data: list[dict[str, Any]] = []
    with logging_redirect_tqdm():
        bar = tqdm(
            enumerate(valid_files, start=1),
            total=len(valid_files),
            desc="Building proxies",
            unit="video",
            dynamic_ncols=True,
            disable=None,
        )
        for i, file in bar:
            bar.set_postfix_str(file.video_id[:8])
            video_proxy_path = make_video_proxy(file.path, proxy_dir)
            audio_proxy_path = make_audio_proxy(file.path, proxy_dir) if file.kind == "amv" else None

            proxy_media = probe(video_proxy_path)
            assert proxy_media.video is not None  # Should never be None
            row = {
                "video_id": file.video_id,
                "kind": file.kind,
                "series": file.series,
                "path": str(file.path),
                "name": file.path.name,
                "video_proxy_path": str(video_proxy_path),
                "audio_proxy_path": str(audio_proxy_path) if audio_proxy_path else None,
                "duration": file.video.duration,
                "fps": str(file.video.fps),
                "is_vfr": file.video.is_vfr,
                "width": file.video.width,
                "height": file.video.height,
                "n_frames": file.video.n_frames,
                "codec": file.video.codec,
                "proxy_width": proxy_media.video.width,
                "proxy_height": proxy_media.video.height,
            }
            data.append(row)
            logger.debug(
                "[%d/%d] File %s was successfully processed and added to the parquet table.",
                i,
                len(valid_files),
                file.path.name,
            )

    df = pd.DataFrame(
        data,
        columns=COLUMNS,
    )
    df["n_frames"] = df["n_frames"].astype("Int64")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)
    logger.info("Wrote %d videos to %s.", len(df), out_path)
    return df
