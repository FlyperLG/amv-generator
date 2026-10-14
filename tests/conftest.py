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

    `name` may contain sub-directories ("a/clip.mp4"); they are created.
    `vfr=True` makes a variable-frame-rate video: the first second plays at `rate`,
    the rest at half of it (frame timestamps are stretched, nothing is dropped).
    Audio is a 440 Hz sine at 44.1 kHz; `channels=2` makes it stereo (same tone on both).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")

    def _make(
        name: str,
        *,
        video: bool,
        audio: bool,
        duration: float = 2.0,
        size: str = "320x180",
        rate: str = "24000/1001",
        vfr: bool = False,
        channels: int = 1,
    ) -> Path:
        out = tmp_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
        if video:
            cmd += ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={duration}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}"]
        if video and vfr:
            # frame N -> timestamp N/rate for the first second, then twice as far apart
            cmd += ["-vf", f"setpts='if(lt(N,{rate}),N,{rate}+(N-{rate})*2)/({rate})/TB'", "-fps_mode", "vfr"]
        if video:
            cmd += ["-c:v", "libx264", "-preset", "ultrafast"]
        if audio:
            cmd += ["-ac", str(channels)]
        cmd.append(str(out))
        subprocess.run(cmd, check=True)
        return out

    return _make
