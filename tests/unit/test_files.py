from pathlib import Path, PosixPath

import pytest

from daidai.files import _compute_target_path


@pytest.mark.parametrize(
    ("protocole", "source_path", "target_path", "is_file", "expected"),
    [
        (
            "file",
            "/path/to/file.txt",
            Path("/cache"),
            True,
            (
                PosixPath(
                    "/cache/file/path/to"
                ),  # target_dir / protocole / source_path [:-1]
                PosixPath("/cache/file/path/to/file.txt"),
                "file:///path/to/file.txt",
            ),
        ),
    ],
)
def test_compute_target_path(protocole, source_path, target_path, is_file, expected):
    assert (
        _compute_target_path(protocole, source_path, target_path, is_file) == expected
    )
