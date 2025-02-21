import collections
import contextlib
import contextvars
import enum
import functools
import inspect
import os
import shutil
import threading
import time
import typing
import urllib
import urllib.parse
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, TextIO

import fsspec
import pympler.asizeof
import structlog

logger = structlog.get_logger(__name__)

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


def compute_target_path(
    protocol: str, source_uri: str, destination_dir: str, is_file: bool
) -> tuple[str, str, str]:
    """
    Compute the local cache directory, final target path, and source URI for the given raw path.
    """
    if protocol == "file":
        # For local files, make sure we have an absolute path
        abs_path = Path(source_uri).absolute()
        destination_dir = str(Path(destination_dir).expanduser().resolve())
        # Use parts (skip the root on Unix; adjust if needed for Windows)
        parts = abs_path.parts[1:]
        if is_file:
            target_dir = os.path.join(destination_dir, "local", *parts[:-1])
            target = os.path.join(target_dir, abs_path.name)
            source_uri = abs_path.as_uri()
        else:
            target_dir = os.path.join(destination_dir, "local", *parts)
            target = os.path.join(target_dir, "")
            source_uri = abs_path.as_uri() + "/"
        return target_dir, target, source_uri

    # For remote files, use urllib to parse and then rebuild a cache path.
    parsed = urllib.parse.urlparse(source_uri)
    # Split the path and remove empty parts
    path_parts = [p for p in parsed.path.split("/") if p]
    # If there are query parameters and this is a file, append them to the filename.
    if parsed.query and is_file:
        path_parts[-1] = f"{path_parts[-1]}?{parsed.query}"
    if is_file:
        target_dir = os.path.join(destination_dir, protocol, *path_parts[:-1])
        target = os.path.join(target_dir, path_parts[-1])
    else:
        target_dir = os.path.join(destination_dir, protocol, *path_parts)
        target = os.path.join(target_dir, "")
    # Rebuild the source URI (adding a trailing slash for directories)
    return target_dir, target, source_uri


class ComponentType(enum.Enum):
    PREDICTOR = "predictor"
    ARTIFACT = "artifact"


class FileDependencyCacheStrategy(enum.Enum):
    ON_DISK: Annotated[str, "Fetch and store on permanently on disk"] = "on_disk"
    ON_DISK_TEMP: Annotated[str, "Fetch and temporarily store on disk"] = (
        "on_disk_temporary"
    )
    NO_CACHE: Annotated[str, "Do not cache the file"] = "no_cache"


class FileDependencyParams(typing.TypedDict):
    storage_options: Annotated[
        dict[str, Any], "see fsspec storage options for more details"
    ]
    open_options: Annotated[dict[str, Any], "see fsspec open options for more details"]
    deserialization: Annotated[dict[str, Any], "deserialization options for the file"]
    cache_strategy: Annotated[FileDependencyCacheStrategy, "cache strategy to use"]


CURRENT_NAMESPACE = contextvars.ContextVar("CURRENT_NAMESPACE", default="global")


class Metadata(typing.TypedDict):
    kind: ComponentType
    dependencies: list[
        tuple[str, Callable, dict[str, typing.Any]]
    ]  # (param_name, dep_func, dep_func_args)
    files: list[tuple[str, str, dict[str, typing.Any]]]
    # (param_name, files_uri, files_args)
    function: Callable


class MetaModelManager:
    _global_lock: threading.RLock = threading.RLock()
    namespaces: typing.ClassVar = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    functions: typing.ClassVar[dict[str, Metadata]] = {}  # Using function names as keys


def create_cache_key(args: dict[str, Any] | None) -> frozenset:
    """Create a hashable key for caching, making all values hashable."""
    if not args:
        return frozenset()
    hashable_items = []
    for k, v in args.items():
        if isinstance(v, Callable):
            continue
        # Convert mutable types to immutable
        if isinstance(v, dict):
            v = frozenset((k2, v2) for k2, v2 in v.items())
        elif isinstance(v, list):
            v = tuple(v)
        elif isinstance(v, set):
            v = frozenset(v)
        hashable_items.append((k, v))
    return frozenset(hashable_items)


P = typing.ParamSpec("P")
R = typing.TypeVar("R")


class ModelManager:
    def __init__(
        self,
        preload: dict[Callable, dict[str, Any] | None]
        | Iterable[Callable | Generator]
        | None = None,
        namespace: str | None = None,
    ):
        self.namespace = namespace or CURRENT_NAMESPACE.get()
        self.namespace_token = CURRENT_NAMESPACE.set(self.namespace)
        self._exit_stack = contextlib.ExitStack()
        if not isinstance(preload, dict):
            preload = dict.fromkeys(preload or [], None)
        self.artifacts_or_predictors = preload
        if preload:
            self._load()

    @staticmethod
    def _get_from_cache(
        namespace: dict[str, dict[frozenset, Any]],
        func_name: str,
        cache_key: frozenset,
    ) -> Any | None:
        """Try to get a value from cache."""
        return namespace.get(func_name, {}).get(cache_key)

    @staticmethod
    def _cache_value(
        namespace: dict[str, dict[frozenset, Any]],
        func_name: str,
        cache_key: frozenset,
        value: Any,
    ) -> None:
        """Cache a value."""
        namespace.setdefault(func_name, {})[cache_key] = value

    @functools.singledispatchmethod
    @staticmethod
    def load(*artifacts_or_predictors: Callable | Generator):
        """Load multiple artifacts or predictors without configs."""
        return ModelManager.load(dict.fromkeys(artifacts_or_predictors, None))

    @load.register(Iterable)
    @staticmethod
    def _(
        artifacts_or_predictors: Iterable[Callable | Generator],
    ):
        """Load multiple artifacts or predictors without configs."""
        return ModelManager.load(dict.fromkeys(artifacts_or_predictors, None))

    @load.register(dict)
    @staticmethod
    def _(artifacts_or_predictors: dict[Callable, dict[str, Any] | None]):
        """Load multiple artifacts or predictors with configs."""
        current_namespace = CURRENT_NAMESPACE.get()
        logger.debug(
            "Loading model components",
            artifacts=[
                f
                for f in artifacts_or_predictors
                if MetaModelManager.functions[f.__name__]["kind"]
                == ComponentType.ARTIFACT
            ],
            predictors=[
                f.__name__
                for f in artifacts_or_predictors
                if MetaModelManager.functions[f.__name__]["kind"]
                == ComponentType.PREDICTOR
            ],
            namespace=current_namespace,
        )
        for artifact_or_predictor, config in artifacts_or_predictors.items():
            ModelManager._load_artifact_or_predictor(
                MetaModelManager.namespaces[current_namespace],
                artifact_or_predictor,
                config,
            )

    def _load(self):
        def _fill_exit_stack():
            for func_name, meta in MetaModelManager.functions.items():
                if "clean_up" in meta:
                    self._exit_stack.callback(
                        lambda m=meta, fn=func_name: self._cleanup_artifact(m, fn)
                    )

        for func_name, meta in MetaModelManager.functions.items():
            if "clean_up" in meta:
                self._exit_stack.callback(
                    lambda m=meta, fn=func_name: self._cleanup_artifact(m, fn)
                )
        try:
            self.load(self.artifacts_or_predictors)
        except Exception as e:
            logger.error("Error during loading components", error=str(e))
            _fill_exit_stack()
            self._exit_stack.close()
            raise
        _fill_exit_stack()
        return self

    def _close(self):
        logger.debug("Closing model manager", namespace=self.namespace)
        CURRENT_NAMESPACE.reset(self.namespace_token)
        try:
            self._exit_stack.close()
        except Exception as cleanup_error:
            logger.error("Error during cleanup", error=str(cleanup_error))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        if exc_type:
            logger.error("Exiting due to exception", error=str(exc_val))

    @staticmethod
    def _cleanup_artifact(meta, func_name):
        try:
            logger.debug("Tearing down component", component=func_name)
            if "clean_up" in meta:
                next(meta["clean_up"])
        except StopIteration:
            pass
        except Exception as e:
            logger.error(
                "Failed to clean up component",
                name=func_name,
                error=str(e),
                error_type=e.__class__.__name__,
            )

    @staticmethod
    def close(
        namespace: dict[str, dict[frozenset, Any]] | None = None,
    ):
        current_namespace = CURRENT_NAMESPACE.get()
        namespace = namespace or MetaModelManager.namespaces[current_namespace]
        for func_name in list(MetaModelManager.functions):
            if "clean_up" not in MetaModelManager.functions.get(func_name, {}):
                continue
            try:
                ModelManager._cleanup_artifact(
                    MetaModelManager.functions[func_name], func_name
                )
            except StopIteration:
                pass
            except Exception as e:
                logger.error(
                    "Failed to clean up component",
                    name=func_name,
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            MetaModelManager.functions.pop(func_name, None)
            namespace.pop(func_name, None)
        namespace.clear()

    @staticmethod
    def _deserialize_local_file(
        path: str, open_options: dict[str, Any], format: VALID_FORMAT_TYPES
    ) -> str | bytes | Path | BinaryIO | TextIO | Generator[str] | Generator[bytes]:
        path = Path(path).expanduser().resolve()
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

    @staticmethod
    def _load_file_dependency(
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
        if is_dir and (
            open_options.get("mode") or deserialization["format"] is not Path
        ):
            raise ValueError(
                f"Cannot specify read mode or format for directories: {uri} is a directory"
            )
        if files_params["cache_strategy"] in (
            FileDependencyCacheStrategy.ON_DISK,
            FileDependencyCacheStrategy.ON_DISK_TEMP,
        ):
            if files_params[
                "cache_strategy"
            ] is FileDependencyCacheStrategy.ON_DISK_TEMP and deserialization[
                "format"
            ] in (Path, TextIO, BinaryIO):
                raise ValueError(
                    "Cannot use temporary cache strategy with Path, TextIO or BinaryIO deserialization"
                )
            cache_dir = (
                "~/.lightkit_cache/"
                if files_params["cache_strategy"] == FileDependencyCacheStrategy.ON_DISK
                else "~/.lightkit_cache_temp/"
            )
            target_dir, target, source_uri = compute_target_path(
                protocol, raw_path, cache_dir, is_file
            )
            os.makedirs(target_dir, exist_ok=True)
            try:
                fs.cp(source_uri, target, recursive=not is_file)
                return ModelManager._deserialize_local_file(
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
                return ModelManager._deserialize_local_file(
                    raw_path, open_options, deserialization["format"]
                )
            if (
                deserialization["format"] is not str
                and deserialization["format"] is not bytes
            ):
                raise ValueError(
                    "Cannot use NO_CACHE strategy with non-str or non-bytes deserialization when the file is remote"
                )
            with fsspec.open(source_uri, **open_options) as f:
                return f.read()
        logger.error(
            "Feature not yet implemented",
            uri=uri,
            cache_strategy=files_params["cache_strategy"],
            files_params=files_params,
        )
        raise NotImplementedError("Feature not yet implemented")

    @staticmethod
    def _load_artifact_or_predictor(
        namespace: dict[str, dict[frozenset, Any]],
        func: Callable | Generator,
        config: dict[str, Any] | None = None,
    ) -> Callable | Generator:
        t0 = time.perf_counter()
        func_name = func.__name__
        kind = MetaModelManager.functions[func_name]["kind"]
        prepared_args = {}
        config = config or {}
        config_cache_key = create_cache_key(config)
        if cached := ModelManager._get_from_cache(
            namespace, func_name, config_cache_key
        ):
            logger.debug(
                "Using cached component",
                kind=kind.value,
                name=func_name,
                cache_key=str(config_cache_key),
                elapsed=round(time.perf_counter() - t0, 9),
            )
            return cached
        logger.debug(
            "Loading component", name=func_name, kind=kind.value, config=config
        )
        # whether the function is an artifact or a predictor, it can have files dependencies
        files = MetaModelManager.functions[func_name]["files"]
        for param_name, uri, files_params in files:
            logger.debug(
                "Processing files dependency",
                component=func_name,
                param_name=param_name,
                dependency=uri,
                params=files_params,
            )
            files_params["cache_strategy"] = (
                FileDependencyCacheStrategy(files_params["cache_strategy"])
                if "cache_strategy" in files_params
                else FileDependencyCacheStrategy.ON_DISK
            )
            files_params["storage_options"] = files_params.get("storage_options") or {}
            files_params["open_options"] = files_params.get("open_options") or {}
            cache_key = create_cache_key(files_params)
            if file_dependency := ModelManager._get_from_cache(
                namespace,
                "file/" + uri,
                cache_key,  # file/ to avoid collision with function names
            ):
                logger.debug(
                    "Using cached file",
                    name="file/" + uri,
                    cache_key=str(cache_key),
                    elapsed=round(time.perf_counter() - t0, 9),
                )
            else:
                file_dependency = ModelManager._load_file_dependency(uri, files_params)
                ModelManager._cache_value(
                    namespace, "file/" + uri, cache_key, file_dependency
                )
            prepared_args[param_name] = file_dependency
        # For predictors, we don't cache the function itself, just its artifact dependencies
        if kind == ComponentType.PREDICTOR:
            dependencies = MetaModelManager.functions[func_name]["dependencies"]
            logger.debug(
                "Dependency resolution status",
                predictor=func_name,
                resolved_count=len(dependencies),
                resolved_names=[name for name, _, _ in dependencies],
            )
            for param_name, dep_func, dep_func_args in dependencies:
                if param_name in config:
                    logger.debug(
                        "Skipping dependency resolution",
                        component=func_name,
                        dependency=dep_func.__name__,
                        cause="dependency passed in config",
                    )
                    continue
                logger.debug(
                    "Processing dependency",
                    component=func_name,
                    dependency=dep_func.__name__,
                )
                dep_result = ModelManager._load_artifact_or_predictor(
                    namespace, dep_func, dep_func_args
                )
                prepared_args[param_name] = dep_result

            logger.debug("Prepared predictor", name=func_name, args=prepared_args)
            prepared_predictor = functools.partial(
                func, **(prepared_args | (config or {}))
            )
            ModelManager._cache_value(
                namespace, func_name, config_cache_key, prepared_predictor
            )
            return prepared_predictor

        if kind != ComponentType.ARTIFACT:
            raise ValueError(f"Invalid kind {kind}")
        dependencies = MetaModelManager.functions[func_name]["dependencies"]
        logger.debug(
            "Dependency resolution status",
            predictor=func_name,
            resolved_count=len(dependencies),
            resolved_names=[name for name, _, _ in dependencies],
        )
        for param_name, dep_func, dep_func_args in dependencies:
            if param_name in config:
                logger.debug(
                    "Skipping dependency resolution",
                    component=func_name,
                    dependency=dep_func.__name__,
                    cause="dependency passed in config",
                )
                continue

            logger.debug(
                "Processing dependency",
                component=func_name,
                dependency=dep_func.__name__,
            )
            dep_result = ModelManager._load_artifact_or_predictor(
                namespace, dep_func, dep_func_args
            )
            prepared_args[param_name] = dep_result

        final_args = prepared_args | (config or {})
        logger.debug("Computing artifact", name=func_name, args=final_args)
        try:
            result = (
                func.__wrapped__(**final_args)
                if hasattr(func, "__wrapped_component__")
                else func(**final_args)
            )
            if isinstance(result, Generator):
                MetaModelManager.functions[func_name]["clean_up"] = result
                result = next(result)
            # ModelManager._cache_value(namespace, func_name, cache_key, result)
            ModelManager._cache_value(namespace, func_name, config_cache_key, result)
            logger.debug(
                "Component loaded",
                name=func_name,
                kind=kind.value,
                elapsed=round(time.perf_counter() - t0, 9),
                size_mb=round(pympler.asizeof.asizeof(result) / (1024 * 1024), 9),
            )
            return result

        except Exception as e:
            logger.error(
                "Failed to load component",
                name=func_name,
                kind=kind.value,
                error=str(e),
                error_type=e.__class__.__name__,
                config=config,
            )
            raise


def component_decorator(kind: ComponentType):
    """
    A common decorator factory for both artifacts and predictors.

    Args:
        kind: The kind of component (Kind.ARTIFACT or Kind.PREDICTOR).
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Register the function and its metadata.
        MetaModelManager.functions[func.__name__] = Metadata(
            dependencies=[],
            kind=kind,
            function=func,
            files=[],
        )
        hints = typing.get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)

        # Process parameters to set up dependency resolution.
        for param_name in sig.parameters:
            if param_name not in hints:
                continue
            annotation = hints[param_name]
            if typing.get_origin(annotation) is not Annotated:
                continue
            typing_args = typing.get_args(annotation)
            if len(typing_args) < 2:
                raise ValueError(
                    f"Missing dependency configuration for parameter {param_name}"
                )
            origin_type = typing.get_origin(typing_args[0]) or typing_args[0]
            if (
                typing_args[0] in VALID_TYPES or origin_type is Generator
            ) and isinstance(typing_args[1], str):
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
                MetaModelManager.functions[func.__name__]["files"].append(
                    (param_name, files_uri, files_params)
                )
                continue

            dependency = typing_args[1:]
            dep_func: Callable = dependency[0]
            def_sig = inspect.signature(dep_func)
            dep_defaults = {
                k: v.default
                for k, v in def_sig.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            dep_func_args: dict[str, Any] = dependency[1] if len(dependency) > 1 else {}
            MetaModelManager.functions[func.__name__]["dependencies"].append(
                (param_name, dep_func, dep_defaults | dep_func_args)
            )

        @functools.wraps(
            func, assigned=(*functools.WRAPPER_ASSIGNMENTS, "__signature__")
        )
        def wrapper(*args, **kwargs):
            # Build the configuration dictionary from bound arguments.
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            config = dict(bound_args.arguments)
            current_namespace = CURRENT_NAMESPACE.get()
            result = ModelManager._load_artifact_or_predictor(
                MetaModelManager.namespaces[current_namespace],
                func,
                config,
            )
            # For predictors, call the returned partial immediately.
            return result() if kind == ComponentType.PREDICTOR else result

        wrapper.__wrapped_component__ = True
        return wrapper

    return decorator


artifact = component_decorator(ComponentType.ARTIFACT)
predictor = component_decorator(ComponentType.PREDICTOR)
