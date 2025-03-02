import enum
import typing
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, TextIO, TypedDict

from daidai.logs import get_logger

logger = get_logger(__name__)


class ComponentType(enum.Enum):
    PREDICTOR = "predictor"
    ARTIFACT = "artifact"
    FILE = "file"


class Metadata(typing.TypedDict):
    type: ComponentType
    dependencies: list[
        tuple[str, typing.Callable, dict[str, typing.Any]]
    ]  # (param_name, dep_func, dep_func_args)
    files: list[tuple[str, str, dict[str, typing.Any]]]
    # (param_name, files_uri, files_args)
    function: typing.Callable


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
    force_download: Annotated[bool, "force download the file(s)"]


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


class Deserialization(TypedDict, total=False):
    format: VALID_FORMAT_TYPES


class OpenOptions(TypedDict, total=False):
    mode: Literal["r", "rb"]


class DaiDaiError(Exception):
    """Base exception for DaiDai errors"""

    ...


class ModelManagerError(DaiDaiError):
    """Base exception for ModelManager errors"""

    ...


class ComponentLoadError(ModelManagerError):
    """Raised when component loading fails"""

    ...
