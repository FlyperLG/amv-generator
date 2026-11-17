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


@pytest.fixture
def make_amv_with_end_card(tmp_path):
    """Factory: an AMV-like clip whose footage ends in a static end card.

    Usage in a test:  path = make_amv_with_end_card("amvs/a.mp4", footage=4.0, card=3.0)

    The footage is full-frame noise (every frame differs, ~15 mean luma change at 64x36), the
    card is one flat colour (0 change), so the card starts exactly at `footage` seconds.
    Audio: by default a 440 Hz sine runs under both. With `song_end` (< footage), the sine
    stops there, digital silence follows until the card, and an 880 Hz "jingle" plays under
    the card, like a real outro: the song ends before the card fades in. `audio=False`
    makes a video-only file.
    Two more knobs for outros without real silence: `outro_start` (default: footage) starts
    the jingle earlier, under the still moving footage, as when a title card animates in;
    `gap_db` replaces the silence between song_end and outro_start with the song's sine
    turned down by that many dB (e.g. -25), a quiet valley instead of digital silence.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")

    def _make(
        name: str,
        *,
        footage: float = 4.0,
        card: float = 3.0,
        song_end: float | None = None,
        outro_start: float | None = None,
        gap_db: float | None = None,
        audio: bool = True,
        rate: str = "24000/1001",
    ) -> Path:
        out = tmp_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        size = "320x180"
        if not audio:
            audio_inputs, audio_graph = [], ""
        elif song_end is None:
            audio_inputs = [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=44100:duration={footage + card}",
            ]
            audio_graph = ";[2:a]anull[a]"
        else:
            outro = footage if outro_start is None else outro_start
            gap = (
                f"anullsrc=r=44100:cl=mono:d={outro - song_end}"
                if gap_db is None
                else f"sine=frequency=440:sample_rate=44100:duration={outro - song_end},volume={gap_db}dB"
            )
            jingle = f"sine=frequency=880:sample_rate=44100:duration={footage + card - outro}"
            audio_inputs = [
                "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={song_end}",
                "-f", "lavfi", "-i", gap,
                "-f", "lavfi", "-i", jingle,
            ]  # fmt: skip
            audio_graph = ";[2:a][3:a][4:a]concat=n=3:v=0:a=1[a]"
        noise = f"nullsrc=s={size}:r={rate}:d={footage},geq=lum='random(1)*255':cb=128:cr=128"
        cmd = [
            ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", noise,
            "-f", "lavfi", "-i", f"color=c=0x305080:s={size}:r={rate}:d={card}",
            *audio_inputs,
            "-filter_complex", f"[0:v][1:v]concat=n=2:v=1:a=0[v]{audio_graph}",
            "-map", "[v]", *(["-map", "[a]", "-ac", "1"] if audio else []),
            "-c:v", "libx264", "-preset", "ultrafast",
            str(out),
        ]  # fmt: skip
        subprocess.run(cmd, check=True)
        return out

    return _make
