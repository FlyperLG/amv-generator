"""Shared pytest fixtures for the whole test suite.

pytest discovers conftest.py files automatically; every fixture defined here can be
requested by any test simply by naming it as a function argument. Fixtures only one
area needs live in that area's own conftest.py (e.g. tests/io/conftest.py).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.helpers import DATA_DIR

# --- Real media files ---------------------------------------------------------


@pytest.fixture
def bluelock_amv() -> Path:
    """Path to the real Blue Lock AMV; skips the test if it is not present."""
    matches = sorted((DATA_DIR / "amvs").glob("*Blue Lock*.mp4"))
    if not matches:
        pytest.skip("Blue Lock AMV not found in data/amvs/")
    return matches[0]


@pytest.fixture
def make_media(tmp_path):
    """Factory: generate a tiny synthetic media file with ffmpeg's built-in sources.

    Usage in a test:  path = make_media("song.mp3", audio=True, video=False)
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")

    def _make(name: str, *, video: bool, audio: bool, duration: float = 2.0) -> Path:
        out = tmp_path / name
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
        if video:
            cmd += ["-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=24000/1001:duration={duration}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}"]
        if video:
            cmd += ["-c:v", "libx264", "-preset", "ultrafast"]
        cmd.append(str(out))
        subprocess.run(cmd, check=True)
        return out

    return _make
