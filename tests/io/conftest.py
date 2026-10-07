"""Fixtures for amv.io tests: recorded ffprobe outputs.

Each fixture returns a fresh deep copy, so tests can mutate it freely.
To record a new one: run ffprobe -print_format json -show_streams -show_format
on the file, replace format.filename with a plain name, save under
tests/fixtures/ffprobe/.
"""

import copy
from typing import Any

import pytest

from tests.helpers import load_json_fixture


@pytest.fixture(scope="session")
def _probe_cache() -> dict[str, dict[str, Any]]:
    return {
        # MP4, 1080p h264 + AAC (the real Blue Lock AMV)
        "bluelock": load_json_fixture("ffprobe/bluelock_mp4.json"),
        # MP3, audio only
        "song_mp3": load_json_fixture("ffprobe/song_mp3.json"),
        # MKV, video only, no stream duration, no nb_frames
        "episode_mkv": load_json_fixture("ffprobe/episode_video_only_mkv.json"),
        # MKV, video + audio, no stream durations (typical anime episode)
        "episode_av_mkv": load_json_fixture("ffprobe/episode_av_mkv.json"),
    }


@pytest.fixture
def bluelock_probe(_probe_cache) -> dict[str, Any]:
    return copy.deepcopy(_probe_cache["bluelock"])


@pytest.fixture
def song_probe(_probe_cache) -> dict[str, Any]:
    return copy.deepcopy(_probe_cache["song_mp3"])


@pytest.fixture
def mkv_probe(_probe_cache) -> dict[str, Any]:
    return copy.deepcopy(_probe_cache["episode_mkv"])


@pytest.fixture
def mkv_av_probe(_probe_cache) -> dict[str, Any]:
    return copy.deepcopy(_probe_cache["episode_av_mkv"])
