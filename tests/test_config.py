"""Tests for amv.config: the layout every step's settings follow.

SPECIFICATION
  - Each section of settings is a frozen dataclass whose defaults are the configured
    values, plus one instance: PATHS (PathSettings), DECODING (DecodingSettings, step 01),
    MUSIC (MusicSettings, step 02), SHOTS (ShotSettings, step 03). Library code takes them
    as arguments; only entry points import the instances. (Decision: one typed section per
    step, like an options class in C#, instead of loose module constants.)
  - Besides the instances, the module only exports the anchors ROOT and DATA_DIR: no loose
    UPPER_CASE settings.
  - config.py is a leaf: it imports nothing from amv, so importing it for typing can never
    cause an import cycle. It also doesn't need ffmpeg (amv.io.ffmpeg finds the binaries).
  - config.py is the config of the pipeline and the model only. Settings of side tools
    (benchmarks, manual checks) live in their own folder (e.g. benchmarks/settings.py), so
    no path in PATHS points into data/out/, where those tools write.
  - Values more than one step must agree on have one source: MUSIC.sample_rate defaults to
    DECODING.audio_proxy.sample_rate.
  - DECODING.end_card (task 3.8) is an EndCardSettings, or None to turn end-card detection
    off. (Decision: None instead of an `enabled` flag, so "off" is one state.)
"""

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

import amv.config as config
from amv.config import (
    DECODING,
    MUSIC,
    PATHS,
    SHOTS,
    AudioProxySettings,
    DecodingSettings,
    EndCardSettings,
    MusicSettings,
    PathSettings,
    ShotSettings,
    VideoProxySettings,
)

SECTIONS = [
    (PATHS, PathSettings),
    (DECODING, DecodingSettings),
    (DECODING.video_proxy, VideoProxySettings),
    (DECODING.audio_proxy, AudioProxySettings),
    (DECODING.end_card, EndCardSettings),
    (MUSIC, MusicSettings),
    (SHOTS, ShotSettings),
]


@pytest.mark.parametrize(("instance", "cls"), SECTIONS, ids=lambda x: type(x).__name__)
def test_each_section_is_the_default_instance_of_its_class(instance, cls):
    assert type(instance) is cls
    assert instance == cls()


@pytest.mark.parametrize(("instance", "cls"), SECTIONS, ids=lambda x: type(x).__name__)
def test_each_section_is_frozen(instance, cls):
    name = dataclasses.fields(cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, getattr(instance, name))


def test_only_the_anchors_and_the_sections_are_upper_case():
    upper = {name for name in vars(config) if name.isupper()}

    assert upper == {"ROOT", "DATA_DIR", "PATHS", "DECODING", "MUSIC", "SHOTS"}


def test_config_imports_nothing_from_amv():
    tree = ast.parse(Path(config.__file__).read_text(encoding="utf-8"))
    modules = [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

    assert not [m for m in modules if m == "amv" or m.startswith("amv.")]


def test_config_imports_without_ffmpeg_on_the_path():
    code = "import os; os.environ['PATH'] = ''; import amv.config"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_paths_live_below_the_data_dir():
    for f in dataclasses.fields(PathSettings):
        path = getattr(PATHS, f.name)
        assert isinstance(path, Path), f.name
        assert path.is_relative_to(config.DATA_DIR), f.name


def test_paths_hold_no_side_tool_output():
    for f in dataclasses.fields(PathSettings):
        assert not getattr(PATHS, f.name).is_relative_to(config.DATA_DIR / "out"), f.name


def test_decoding_has_the_planned_values():
    assert DECODING.video_extensions == frozenset({".mp4", ".mkv"})
    assert DECODING.audio_extensions == frozenset({".mp3", ".wav", ".flac"})
    assert DECODING.probe_timeout == 10
    assert DECODING.video_proxy == VideoProxySettings(width=640, preset="ultrafast", crf=28, timeout=300)
    assert DECODING.audio_proxy == AudioProxySettings(sample_rate=22050, channels=1, timeout=60)
    assert DECODING.end_card == EndCardSettings(
        scan=30,
        fps=4,
        width=64,
        height=36,
        threshold=2.0,
        smooth=1.0,
        min_tail=3.0,
        lookback=6.0,
        audio_rate=8000,
        audio_window=0.05,
        silence_db=-60.0,
        min_gap=0.25,
        min_dip=18.0,
    )


def test_end_card_detection_can_be_turned_off():
    off = dataclasses.replace(DECODING, end_card=None)

    assert off.end_card is None
    assert off.video_proxy == DECODING.video_proxy


def test_music_reads_the_wavs_step_01_writes():
    assert MUSIC.sample_rate == DECODING.audio_proxy.sample_rate
