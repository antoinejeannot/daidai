import enum
import os
import shutil
import typing
import urllib
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, TextIO, TypedDict

import fsspec

from daidai.logs import get_logger

logger = get_logger(__name__)


class FileDependencyCacheStrategy(enum.Enum):
    ON_DISK: Annotated[str, "Fetch and store on permanently on disk"] = "on_disk"
    ON_DISK_TEMP: Annotated[str, "Fetch and temporarily store on disk"] = (
        "on_disk_temporary"
    )
    NO_CACHE: Annotated[str, "Do not cache the file"] = "no_cache"


class FileDependencyParams(TypedDict):
    storage_options: Annotated[
        dict[str, Any], "see fsspec storage options for more details"
    ]
    open_options: Annotated[dict[str, Any], "see fsspec open options for more details"]
    deserialization: Annotated[dict[str, Any], "deserialization options for the file"]
    cache_strategy: Annotated[FileDependencyCacheStrategy, "cache strategy to use"]


VALID_TYPES = (
    Path,
    bytes,
    str,
    TextIO,
    BinaryIO,
)  # + (Generator[str], Generator[bytes])
VALID_FORMAT_TYPES = (
    type[Path]
    | type[bytes]
    | type[str]
    | type[TextIO]
    | type[BinaryIO]
    | type[Generator[str]]
    | type[Generator[bytes]]
)


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
    protocol: str, source_uri: str, destination_dir: str, is_file: bool
) -> tuple[str, str, str]:
    destination_dir = str(Path(destination_dir).expanduser().resolve())

    if protocol == "file":
        abs_path = Path(source_uri).resolve()
        source_uri = abs_path.as_uri()
        parts = abs_path.parts[1:]
    else:
        parsed = urllib.parse.urlparse(source_uri)
        parts = [p for p in parsed.path.split("/") if p]
        if is_file and parsed.query:
            parts[-1] += f"?{parsed.query}"

    target_dir = os.path.join(
        destination_dir, protocol, *parts[:-1] if is_file else parts
    )
    target = os.path.join(target_dir, parts[-1]) if is_file else target_dir
    return target_dir, target, source_uri


def load_file_dependency(
    uri: str, files_params: FileDependencyParams
) -> str | bytes | Path | BinaryIO | TextIO | Generator[str] | Generator[bytes]:
    options = fsspec.utils.infer_storage_options(uri)
    protocol = options.get("protocol", "file")
    raw_path = options.get("path") or uri  # Fall back to the full URI if needed
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
        if files_params[
            "cache_strategy"
        ] is FileDependencyCacheStrategy.ON_DISK_TEMP and deserialization["format"] in (
            Path,
            TextIO,
            BinaryIO,
        ):
            raise ValueError(
                "Cannot use temporary cache strategy with Path, TextIO or BinaryIO deserialization"
            )
        cache_dir = (
            "~/.lightkit_cache/"
            if files_params["cache_strategy"] == FileDependencyCacheStrategy.ON_DISK
            else "~/.lightkit_cache_temp/"
        )
        target_dir, target, source_uri = _compute_target_path(
            protocol, raw_path, cache_dir, is_file
        )
        os.makedirs(target_dir, exist_ok=True)
        try:
            fs.cp(source_uri, target, recursive=not is_file)
            return _deserialize_local_file(
                target, open_options, deserialization["format"]
            )
        except Exception as e:
            logger.error(
                "Failed to copy file",
                source=source_uri,
                target=target,
                error=str(e),
                error_type=e.__class__.__name__,
            )
            raise
        finally:
            if (
                files_params["cache_strategy"]
                == FileDependencyCacheStrategy.ON_DISK_TEMP
            ):
                shutil.rmtree(target_dir, ignore_errors=True)
    if files_params["cache_strategy"] == FileDependencyCacheStrategy.NO_CACHE:
        if protocol == "file":
            return _deserialize_local_file(
                raw_path, open_options, deserialization["format"]
            )
        if (
            deserialization["format"] is not str
            and deserialization["format"] is not bytes
        ):
            raise ValueError(
                "Cannot use NO_CACHE strategy with non-str or non-bytes deserialization when the file is remote"
            )
        with fsspec.open(raw_path, **open_options) as f:
            return f.read()
    logger.error(
        "Feature not yet implemented",
        uri=uri,
        cache_strategy=files_params["cache_strategy"],
        files_params=files_params,
    )
    raise NotImplementedError("Feature not yet implemented")
