import functools
import inspect
import typing
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Annotated, Any, BinaryIO, TextIO

from daidai.config import CONFIG
from daidai.logs import get_logger
from daidai.managers import (
    Metadata,
    _current_namespace,
    _functions,
    _load_one_artifact_or_predictor,
    _namespaces,
)
from daidai.types import (
    VALID_TYPES,
    ComponentType,
    Deserialization,
    FileDependencyCacheStrategy,
    OpenOptions,
)

logger = get_logger(__name__)

P = typing.ParamSpec("P")
R = typing.TypeVar("R")


class Artifact:
    def __init__(
        self,
        fn: Callable[P, R],
        *,
        force_download: bool = CONFIG.force_download,
        cache_strategy: str | FileDependencyCacheStrategy = CONFIG.cache_strategy,
        cache_directory: str | Path = CONFIG.cache_dir,
        storage_options: dict[str, Any] | None = None,
        open_options: dict | None = None,
        deserialization: dict | None = None,
        **fn_params,
    ) -> R:
        self.fn = fn
        self.fn_params = fn_params
        self.force_download = force_download
        self.cache_strategy = (
            cache_strategy
            if isinstance(cache_strategy, FileDependencyCacheStrategy)
            else FileDependencyCacheStrategy(cache_strategy)
        )
        self.cache_directory = (
            cache_directory
            if isinstance(cache_directory, Path)
            else Path(cache_directory)
        )
        self.storage_options = storage_options or {}
        self.open_options = open_options or OpenOptions()
        self.deserialization = deserialization or Deserialization()

        if not inspect.isfunction(self.fn):
            logger.error(f"fn must be a user-defined function, got {type(self.fn)}")
            raise TypeError(f"fn must be a user-defined function, got {type(self.fn)}")

        if self.fn.__name__ not in _functions:
            logger.error(
                f"fn {self.fn.__name__} is not registered, register it with @artifact"
            )
            raise ValueError(
                f"fn {self.fn.__name__} is not registered, register it with @artifact"
            )

        if _functions[self.fn.__name__]["type"] != ComponentType.ARTIFACT:
            logger.error(f"fn {self.fn.__name__} is not an artifact")
            raise TypeError(f"fn {self.fn.__name__} is not an artifact")

        if not isinstance(self.cache_strategy, FileDependencyCacheStrategy):
            logger.error(
                f"cache_strategy must be FileDependencyCacheStrategy, got {type(self.cache_strategy)}"
            )
            raise TypeError(
                f"cache_strategy must be FileDependencyCacheStrategy, got {type(self.cache_strategy)}"
            )

        if self.open_options.get("mode") not in ("r", "rb", None):
            raise ValueError(
                f"open_options mode must be 'r' or 'rb', got '{self.open_options['mode']}'"
            )

    def get_args(self) -> dict[str, Any]:
        # TODO: pass the cache_strategy, cache_directory, storage_options, open_options, deserialization
        # to the underlying File dependencies and return them here
        return self.fn_params


class Predictor:
    def __init__(
        self,
        fn: Callable,
        **fn_params,
    ) -> None:
        self.fn = fn
        self.fn_params = fn_params

        if not inspect.isfunction(self.fn):
            logger.error(f"fn must be a user-defined function, got {type(self.fn)}")
            raise TypeError(f"fn must be a user-defined function, got {type(self.fn)}")

        if self.fn.__name__ not in _functions:
            logger.error(
                f"fn {self.fn.__name__} is not registered, register it with @predictor"
            )
            raise ValueError(
                f"fn {self.fn.__name__} is not registered, register it with @predictor"
            )

        if _functions[self.fn.__name__]["type"] != ComponentType.PREDICTOR:
            logger.error(f"fn {self.fn.__name__} is not a predictor")
            raise TypeError(f"fn {self.fn.__name__} is not a predictor")

    def get_args(self) -> dict[str, Any]:
        return self.fn_params


def component_decorator(
    component_type: ComponentType,
):
    if component_type not in (ComponentType.ARTIFACT, ComponentType.PREDICTOR):
        raise ValueError(
            f"Invalid component type {component_type}. "
            f"Must be one of {ComponentType.ARTIFACT}, {ComponentType.PREDICTOR}"
        )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _functions[func.__name__] = Metadata(
            dependencies=[],
            type=component_type,
            function=func,
            files=[],
        )
        hints = typing.get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)

        for param_name in sig.parameters:
            if param_name not in hints:
                continue
            annotation = hints[param_name]
            if typing.get_origin(annotation) is not Annotated:
                continue
            typing_args = typing.get_args(annotation)
            if len(typing_args) < 2:
                raise TypeError(
                    f"Annotated type hint must have at least 2 arguments, got {len(typing_args)}"
                )
            origin_type = typing.get_origin(typing_args[0]) or typing_args[0]
            if (
                typing_args[0] in VALID_TYPES or origin_type is Generator
            ) and isinstance(typing_args[1], str):
                # File dependency, e.g. Annotated[Path, "s3://bucket/file"]
                # only check for a string as the second argument
                # TODO: narrow down the check to a valid URI
                files_uri = typing_args[1]
                files_params = typing_args[2] if len(typing_args) > 2 else {}
                open_options = files_params.setdefault("open_options", {})
                deserialization = files_params.setdefault("deserialization", {})
                if deserialization.get("format") not in (None, typing_args[0]):
                    raise TypeError(
                        f"Deserialization format {deserialization.get('format')} "
                        f"does not match the expected type {typing_args[0]}"
                    )
                deserialization["format"] = typing_args[0]
                if typing_args[0] is Path and open_options.get("mode"):
                    raise ValueError(
                        "Cannot specify mode for Path objects. Use 'str' or 'bytes' instead."
                    )
                if typing_args[0] is bytes or typing_args[0] is BinaryIO:
                    open_options.setdefault("mode", "rb")
                    if open_options["mode"] != "rb":
                        raise ValueError(
                            "Cannot read bytes in text mode. Use 'rb' instead."
                        )
                    open_options["mode"] = "rb"
                elif typing_args[0] is str or typing_args[0] is TextIO:
                    open_options.setdefault("mode", "r")
                    if open_options["mode"] != "r":
                        raise ValueError(
                            "Cannot read text in binary mode. Use 'r' instead."
                        )
                files_params["cache_strategy"] = (
                    FileDependencyCacheStrategy(files_params["cache_strategy"])
                    if files_params.get("cache_strategy")
                    else CONFIG.cache_strategy
                )
                files_params["storage_options"] = (
                    files_params.get("storage_options") or {}
                )
                files_params["open_options"] = files_params.get("open_options") or {}
                files_params["force_download"] = (
                    files_params.get("force_download") or CONFIG.force_download
                )
                _functions[func.__name__]["files"].append(
                    (param_name, files_uri, files_params)
                )
                continue
            dependency = typing_args[1:]
            if not (
                inspect.isfunction(dependency[0])
                or isinstance(dependency[0], Artifact | Predictor)
            ):
                continue
            elif inspect.isfunction(dependency[0]):
                dep_func: Callable = dependency[0]
                def_sig = inspect.signature(dep_func)
                dep_defaults = {
                    k: v.default
                    for k, v in def_sig.parameters.items()
                    if v.default is not inspect.Parameter.empty
                }
                dep_func_args: dict[str, Any] = (
                    dependency[1]
                    if len(dependency) > 1 and isinstance(dependency[1], dict)
                    else {}
                )
                _functions[func.__name__]["dependencies"].append(
                    (param_name, dep_func, dep_defaults | dep_func_args)
                )
            else:
                dep: Artifact | Predictor = dependency[0]
                def_sig = inspect.signature(dep.fn)
                dep_defaults = {
                    k: v.default
                    for k, v in def_sig.parameters.items()
                    if v.default is not inspect.Parameter.empty
                }
                _functions[func.__name__]["dependencies"].append(
                    (param_name, dep.fn, dep_defaults | dep.get_args())
                )

        @functools.wraps(
            func, assigned=(*functools.WRAPPER_ASSIGNMENTS, "__signature__")
        )
        def wrapper(*args, **kwargs):
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            config = dict(bound_args.arguments)
            current_namespace = _current_namespace.get()
            result = _load_one_artifact_or_predictor(
                _namespaces[current_namespace],
                func,
                config,
            )
            return result() if component_type == ComponentType.PREDICTOR else result

        wrapper.__wrapped_component__ = True
        return wrapper

    return decorator


artifact = component_decorator(ComponentType.ARTIFACT)
predictor = component_decorator(ComponentType.PREDICTOR)
