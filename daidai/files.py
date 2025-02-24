import contextlib
import threading
import typing
import urllib.parse
from collections.abc import Generator
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO
from weakref import WeakValueDictionary

import fsspec.utils

from daidai.config import CONFIG
from daidai.logs import get_logger
from daidai.types import (
    VALID_FORMAT_TYPES,
    FileDependencyCacheStrategy,
    FileDependencyParams,
)

logger = get_logger(__name__)


_file_locks = WeakValueDictionary()
_file_locks_lock = threading.RLock()  # Lock for the locks themselves


def get_file_lock(file_path: str) -> threading.RLock:
    with _file_locks_lock:
        return _file_locks.setdefault(file_path, threading.RLock())


def _deserialize_local_file(
    path: str, open_options: dict[str, Any], format: VALID_FORMAT_TYPES
) -> str | bytes | Path | BinaryIO | TextIO | Generator[str] | Generator[bytes]:
    path: Path = Path(path).expanduser().resolve()
    if format is Path:
        return path
    if format is bytes:
        return path.read_bytes()
    if format is str:
        return path.read_text()
    if format is BinaryIO:
        return path.open(**({"mode": "rb"} | open_options))
    if format is TextIO:
        return path.open(**({"mode": "r"} | open_options))
    if (typing.get_origin(format) or format) is Generator:

        def _stream(mode: Literal["r", "rb"]):
            with path.open(**({"mode": mode} | open_options)) as stream:
                yield from stream

        format_arg = typing.get_args(format)
        if format_arg and format_arg[0] is str:
            return _stream("r")
        if format_arg and format_arg[0] is bytes:
            return _stream("rb")
        if format_arg:
            raise ValueError(
                f"Generator format should be 'str' or 'bytes', not {format_arg[0]!s}"
            )
        if open_options.get("mode") == "r":
            return _stream("r")
        if open_options.get("mode") == "rb":
            return _stream("rb")
        if open_options.get("mode"):
            raise ValueError(
                f"Generator mode should be 'r' or 'rb', not {open_options['mode']!s}"
            )
        raise ValueError(
            "Generator should send type: 'str' or 'bytes', i.e.: Generator[str] or Generator[bytes]"
        )
    raise ValueError(f"Unsupported deserialization format {format}")


def _compute_target_path(
    protocol: str, source_uri: str, destination_dir: Path, is_file: bool
) -> tuple[Path, Path, str]:
    if protocol == "file":
        abs_path = Path(source_uri).expanduser().resolve()
        source_uri = abs_path.as_uri()
        parts = abs_path.parts[1:]
    else:
        parsed = urllib.parse.urlparse(source_uri)
        parts = [p for p in parsed.path.split("/") if p]
        if is_file and parsed.query:
            parts[-1] += f"?{parsed.query}"

    target_dir = destination_dir / protocol / Path(*parts[:-1])
    target = target_dir / parts[-1]
    return target_dir, target, source_uri


def load_file_dependency(
    uri: str, files_params: FileDependencyParams
) -> str | bytes | Path | BinaryIO | TextIO | Generator[str] | Generator[bytes]:
    options = fsspec.utils.infer_storage_options(uri)
    protocol = options.get("protocol", "file")
    raw_path = options.get("path") or uri  # Fall back to the full URI if needed

    lock_key = f"{protocol}://{raw_path}"
    file_lock = get_file_lock(lock_key)
    with file_lock:
        return _load_file_dependency_impl(uri, files_params, protocol, raw_path)


def _load_file_dependency_impl(
    uri: str, files_params: FileDependencyParams, protocol: str, raw_path: str
) -> str | bytes | Path | BinaryIO | TextIO | Generator[str] | Generator[bytes]:
    fs = fsspec.filesystem(protocol, **files_params["storage_options"])
    open_options = files_params["open_options"]
    deserialization = files_params["deserialization"]
    is_dir = fs.isdir(raw_path) if hasattr(fs, "isdir") else False
    is_file = fs.isfile(raw_path) if hasattr(fs, "isfile") else not is_dir
    if is_dir and is_file:
        raise ValueError(f"Cannot determine if {uri} is a file or a directory")
    if is_dir and (open_options.get("mode") or deserialization["format"] is not Path):
        raise ValueError(
            f"Cannot specify read mode or format for directories: {uri} is a directory"
        )
    if files_params["cache_strategy"] in (
        FileDependencyCacheStrategy.ON_DISK,
        FileDependencyCacheStrategy.ON_DISK_TEMP,
    ):
        return _handle_disk_cache(
            protocol, raw_path, fs, files_params, is_file, open_options, deserialization
        )

    if files_params["cache_strategy"] == FileDependencyCacheStrategy.NO_CACHE:
        return _handle_no_cache(protocol, raw_path, open_options, deserialization)

    logger.error(
        "Feature not yet implemented",
        uri=uri,
        cache_strategy=files_params["cache_strategy"],
        files_params=files_params,
    )
    raise NotImplementedError("Feature not yet implemented")


def _handle_disk_cache(
    protocol: str,
    raw_path: str,
    fs,
    files_params: FileDependencyParams,
    is_file: bool,
    open_options: dict,
    deserialization: dict,
) -> Any:
    cache_dir: Path = (
        CONFIG.cache_dir
        if files_params["cache_strategy"] == FileDependencyCacheStrategy.ON_DISK
        else CONFIG.cache_dir_tmp
    )
    target_dir, target, source_uri = _compute_target_path(
        protocol, raw_path, cache_dir, is_file
    )
    success = target.with_suffix(target.suffix + ".SUCCESS")
    need_download = (
        files_params["force_download"] or not success.exists() or not target.exists()
    )
    try:
        if not need_download:
            logger.debug("Cache hit, using cached file", target=target)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            if is_file:
                fs.cp(source_uri, str(target))
            else:
                fs.cp(source_uri, str(target_dir), recursive=True)
            success.touch()
        return _deserialize_local_file(target, open_options, deserialization["format"])
    except Exception as e:
        with contextlib.suppress(Exception):
            success.unlink(missing_ok=True)
        logger.error(
            "Failed to copy file(s)",
            source=source_uri,
            target=target,
            error=str(e),
            error_type=e.__class__.__name__,
        )
        raise


def _handle_no_cache(
    protocol: str, raw_path: str, open_options: dict, deserialization: dict
) -> Any:
    """Handle no-cache strategy."""
    if protocol == "file":
        return _deserialize_local_file(
            raw_path, open_options, deserialization["format"]
        )

    if deserialization["format"] is not str and deserialization["format"] is not bytes:
        raise ValueError(
            "Cannot use NO_CACHE strategy with non-str or non-bytes deserialization when the file is remote"
        )

    with fsspec.open(raw_path, **open_options) as f:
        return f.read()
