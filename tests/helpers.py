"""Plain helper functions for tests (not fixtures).

Import from here instead of from conftest.py: conftest files are loaded by pytest
itself, and importing them as regular modules can load them twice.
"""

import json
from pathlib import Path
from typing import Any

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
DATA_DIR = TESTS_DIR.parent / "data"


def load_json_fixture(relative_path: str) -> dict[str, Any]:
    """Load a JSON file from tests/fixtures/, e.g. load_json_fixture("ffprobe/song_mp3.json")."""
    with open(FIXTURES_DIR / relative_path, encoding="utf-8") as f:
        return json.load(f)
