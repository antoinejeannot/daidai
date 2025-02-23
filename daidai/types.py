import enum
import typing


class ComponentType(enum.Enum):
    PREDICTOR = "predictor"
    ARTIFACT = "artifact"


class Metadata(typing.TypedDict):
    type: ComponentType
    dependencies: list[
        tuple[str, typing.Callable, dict[str, typing.Any]]
    ]  # (param_name, dep_func, dep_func_args)
    files: list[tuple[str, str, dict[str, typing.Any]]]
    # (param_name, files_uri, files_args)
    function: typing.Callable
    clean_up: (
        typing.Generator | None
    )  # Only for artifacts using generators for init & cleanup


class ModelManagerError(Exception):
    """Base exception for ModelManager errors"""

    ...


class ComponentLoadError(ModelManagerError):
    """Raised when component loading fails"""

    ...
