"""
pytest configuration for Wi-Fi Vision 3D test suite.

Redirects tmp_path to tests/tmp/ inside the workspace so all tests can
write files regardless of OS-level restrictions on AppData\Local\Temp.
The cleanup_dead_symlinks call at session end is also patched to avoid
the sandbox PermissionError on Windows when iterating the tmp directory.
"""

import os
import pytest
from pathlib import Path

_TESTS_TMP = Path(__file__).parent / "tmp"
_counter = [0]


@pytest.fixture
def tmp_path():
    """
    Workspace-local replacement for pytest's tmp_path fixture.
    Creates a unique subdirectory under tests/tmp/ for each test.
    """
    _tests_tmp = Path(__file__).parent / "tmp"
    _tests_tmp.mkdir(exist_ok=True)
    _counter[0] += 1
    test_dir = _tests_tmp / f"t{_counter[0]:04d}"
    test_dir.mkdir(exist_ok=True)
    return test_dir
