from pathlib import Path

import pytest


@pytest.fixture
def temp_course_dir(tmp_path: Path) -> Path:
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    return course_dir
