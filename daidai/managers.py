import collections
import contextlib
import contextvars
import functools
import time
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any

from daidai.artifacts import load_artifact
from daidai.logs import get_logger
from daidai.types import (
    ArtifactCacheStrategy,
    ComponentLoadError,
    ComponentType,
    Metadata,
)

logger = get_logger(__name__)

try:
    import pympler.asizeof

    has_pympler = True
except ImportError:
    has_pympler = False
    logger.info("pympler is not installed, memory usage will not be logged")

_current_namespace = contextvars.ContextVar("CURRENT_NAMESPACE", default="global")
_namespaces = collections.defaultdict(
    lambda: collections.defaultdict(dict)
)  # id -> func_name -> cache_key -> value
_functions: dict[str, Metadata] = {}  # func_name -> metadata


def _create_cache_key(args: dict[str, Any] | None) -> frozenset:
    if not args:
        return frozenset()

    def make_hashable(value):
        if isinstance(value, dict):
            return frozenset((k, make_hashable(v)) for k, v in value.items())
        elif isinstance(value, list | tuple):
            return tuple(make_hashable(item) for item in value)
        elif isinstance(value, set):
            return frozenset(make_hashable(item) for item in value)
        elif isinstance(value, int | float | str | bool | type(None)):
            return value
        else:
            # Fall back to string representation for non-hashable types
            return f"__custom__{type(value).__name__}_{id(value)}"

    hashable_items = []
    for k, v in args.items():
        if callable(v):
            continue
        hashable_items.append((k, make_hashable(v)))
    return frozenset(hashable_items)


def _get_from_cache(
    namespace: dict[str, dict[frozenset, Any]],
    func_name: str,
    cache_key: frozenset,
) -> Any | None:
    return namespace.get(func_name, {}).get(cache_key)


def _cache_value(
    namespace: dict[str, dict[frozenset, Any]],
    func_name: str,
    cache_key: frozenset,
    value: Any,
) -> None:
    namespace.setdefault(func_name, {})[cache_key] = value


class ModelManager:
    def __init__(
        self,
        preload: dict[Callable, dict[str, Any] | None]
        | Iterable[Callable | Generator]
        | None = None,
        namespace: str | None = None,
        force_download: bool | None = None,
        cache_strategy: str | ArtifactCacheStrategy | None = None,
        cache_directory: str | Path | None = None,
        storage_options: dict[str, Any] | None = None,
    ):
        if namespace == "global":
            raise ValueError("Cannot use 'global' as a namespace")
        self.namespace = namespace or str(id(self))
        self._namespace_token = _current_namespace.set(self.namespace)
        self._exit_stack = contextlib.ExitStack()
        if isinstance(preload, dict):
            pass
        elif isinstance(preload, Generator | Callable):
            preload = {preload: None}
        elif preload is None or (isinstance(preload, Iterable)):
            preload = dict.fromkeys(preload or [], None)
        else:
            raise TypeError(f"Invalid type for assets_or_predictors: {type(preload)}")
        self.assets_or_predictors = preload
        self.artifact_config = {}
        if force_download is not None:
            self.artifact_config["force_download"] = force_download
        if cache_strategy is not None:
            self.artifact_config["cache_strategy"] = (
                cache_strategy
                if isinstance(cache_strategy, ArtifactCacheStrategy)
                else ArtifactCacheStrategy(cache_strategy)
            )
        if cache_directory is not None:
            self.artifact_config["cache_directory"] = (
                cache_directory
                if isinstance(cache_directory, Path)
                else Path(cache_directory)
            )
        if storage_options is not None:
            self.artifact_config["storage_options"] = storage_options
        if preload:
            self._load()

    @property
    def _namespace(self) -> dict[str, dict[frozenset, Any]]:
        return _namespaces[self.namespace]

    def load(
        self,
        assets_or_predictors: Callable
        | Generator
        | dict[Callable, dict[str, Any] | None]
        | Iterable[Callable | Generator],
        force_download: bool | None = None,
        cache_strategy: str | ArtifactCacheStrategy | None = None,
        cache_directory: str | Path | None = None,
        storage_options: dict[str, Any] | None = None,
    ):
        artifact_config = self.artifact_config.copy()
        if force_download is not None:
            artifact_config["force_download"] = force_download
        if cache_strategy is not None:
            artifact_config["cache_strategy"] = (
                cache_strategy
                if isinstance(cache_strategy, ArtifactCacheStrategy)
                else ArtifactCacheStrategy(cache_strategy)
            )
        if cache_directory is not None:
            artifact_config["cache_directory"] = (
                cache_directory
                if isinstance(cache_directory, Path)
                else Path(cache_directory)
            )
        if storage_options is not None:
            artifact_config["storage_options"] = storage_options

        if isinstance(assets_or_predictors, dict):
            return _load_many_assets_or_predictors(
                self._namespace, assets_or_predictors, artifact_config
            )
        if isinstance(assets_or_predictors, Iterable):
            return _load_many_assets_or_predictors(
                self._namespace,
                dict.fromkeys(assets_or_predictors, None),
                artifact_config,
            )
        if isinstance(assets_or_predictors, Callable | Generator):
            return _load_one_asset_or_predictor(
                self._namespace, assets_or_predictors, None, artifact_config
            )

        raise TypeError(
            f"Invalid type for assets_or_predictors: {type(assets_or_predictors)}"
        )

    def _load(self):
        try:
            self.load(self.assets_or_predictors)
            self._exit_stack = _register_cleanup_functions(self._namespace)
        except Exception as e:
            logger.error("Error during loading components", error=str(e))
            # If an error occurs during loading, we still need to clean up the loaded components
            self._exit_stack = _register_cleanup_functions(self._namespace)
            self.close()
            raise

    def close(self):
        logger.debug("Closing model manager", namespace=self.namespace)
        try:
            self._exit_stack.close()
        except Exception as cleanup_error:
            logger.error("Error during cleanup", error=str(cleanup_error))
            raise
        finally:
            _namespaces.pop(self.namespace)
            _current_namespace.reset(self._namespace_token)
            logger.debug("Model manager closed", namespace=self.namespace)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type:
            logger.error("Exiting due to exception", error=str(exc_val))


def _load_one_asset_or_predictor(
    namespace: dict[str, dict[frozenset, Any]],
    func: Callable | Generator,
    config: dict[str, Any] | None = None,
    parent_artifact_config: dict[str, Any] | None = None,
) -> Callable | Generator:
    t0 = time.perf_counter()
    component_type = _functions[func.__name__]["type"]
    prepared_args = {}
    config = config or {}
    parent_artifact_config = parent_artifact_config or {}
    current_artifact_config = {
        k: config.pop(k)
        for k in (
            "force_download",
            "cache_strategy",
            "cache_directory",
            "storage_options",
        )
        if k in config
    }
    effective_artifact_config = current_artifact_config | parent_artifact_config
    config_cache_key = _create_cache_key(config)
    if cached := _get_from_cache(namespace, func.__name__, config_cache_key):
        logger.debug(
            "Using cached component",
            type=component_type.value,
            name=func.__name__,
            cache_key=str(config_cache_key),
            elapsed=round(time.perf_counter() - t0, 9),
        )
        return cached
    logger.debug(
        "Loading component",
        name=func.__name__,
        type=component_type.value,
        config=config,
        artifact_config=effective_artifact_config,
    )
    # whether the function is an asset or a predictor, it can have artifacts
    artifacts = _functions[func.__name__]["artifacts"]
    for param_name, uri, artifact_params in artifacts:
        if param_name in config:
            logger.debug(
                "Skipping artifact dependency resolution",
                component=func.__name__,
                dependency=uri,
                cause="dependency passed in config",
            )
            continue

        merged_params = artifact_params | effective_artifact_config
        logger.debug(
            "Processing artifact",
            component=func.__name__,
            param_name=param_name,
            dependency=uri,
            params=merged_params,
        )
        cache_key = _create_cache_key(merged_params)
        artifact = _get_from_cache(
            namespace, "artifact/" + uri, cache_key
        )  # artifact/ to avoid collision with function names
        if artifact:
            logger.debug(
                "Using cached artifact",
                name="artifact/" + uri,
                cache_key=str(cache_key),
                elapsed=round(time.perf_counter() - t0, 9),
            )
        else:
            artifact = load_artifact(uri, merged_params)
            _cache_value(namespace, "artifact/" + uri, cache_key, artifact)
        prepared_args[param_name] = artifact
    # For predictors, we don't cache the function itself, just its asset dependencies
    if component_type == ComponentType.PREDICTOR:
        dependencies = _functions[func.__name__]["dependencies"]
        logger.debug(
            "Dependency resolution status",
            predictor=func.__name__,
            resolved_count=len(dependencies),
            resolved_names=[name for name, _, _ in dependencies],
        )
        for param_name, dep_func, dep_func_args in dependencies:
            if param_name in config:
                logger.debug(
                    "Skipping dependency resolution",
                    component=func.__name__,
                    dependency=dep_func.__name__,
                    cause="dependency passed in config",
                )
                continue
            logger.debug(
                "Processing dependency",
                component=func.__name__,
                dependency=dep_func.__name__,
                param_name=param_name,
                params=dep_func_args,
                artifact_params=effective_artifact_config,
            )
            dep_result = _load_one_asset_or_predictor(
                namespace, dep_func, dep_func_args, effective_artifact_config
            )
            prepared_args[param_name] = dep_result

        logger.debug("Prepared predictor", name=func.__name__, args=prepared_args)
        prepared_predictor = functools.partial(func, **(prepared_args | (config or {})))
        _cache_value(namespace, func.__name__, config_cache_key, prepared_predictor)
        return prepared_predictor

    if component_type != ComponentType.ASSET:
        logger.error("Invalid component type", component_type=component_type)
        raise ValueError(f"Invalid component type: {component_type}")
    dependencies = _functions[func.__name__]["dependencies"]
    logger.debug(
        "Dependency resolution status",
        predictor=func.__name__,
        resolved_count=len(dependencies),
        resolved_names=[name for name, _, _ in dependencies],
    )
    for param_name, dep_func, dep_func_args in dependencies:
        if param_name in config:
            logger.debug(
                "Skipping dependency resolution",
                component=func.__name__,
                dependency=dep_func.__name__,
                cause="dependency passed in config",
            )
            continue

        logger.debug(
            "Processing dependency",
            component=func.__name__,
            dependency=dep_func.__name__,
        )
        dep_result = _load_one_asset_or_predictor(
            namespace, dep_func, dep_func_args, effective_artifact_config
        )
        prepared_args[param_name] = dep_result

    final_args = prepared_args | (config or {})
    logger.debug("Computing asset", name=func.__name__, args=final_args)
    try:
        result = (
            func.__wrapped__(**final_args)
            if hasattr(func, "__wrapped_component__")
            else func(**final_args)
        )
        if isinstance(result, Generator):
            namespace.setdefault("__cleanup__", {})[func.__name__] = result
            result = next(result)
        _cache_value(namespace, func.__name__, config_cache_key, result)
        logger.debug(
            "Component loaded",
            name=func.__name__,
            type=component_type.value,
            elapsed=round(time.perf_counter() - t0, 9),
            size_mb=round(pympler.asizeof.asizeof(result) / (1024 * 1024), 9)
            if has_pympler
            else None,
        )
        return result

    except Exception as e:
        logger.error(
            "Failed to load component",
            name=func.__name__,
            type=component_type.value,
            error=str(e),
            error_type=e.__class__.__name__,
            config=config,
        )
        raise ComponentLoadError("Failed to load component") from e


def _load_many_assets_or_predictors(
    namespace: dict[str, dict[frozenset, Any]],
    assets_or_predictors: dict[Callable, dict[str, Any] | None],
    parent_artifact_config: dict[str, Any] | None = None,
) -> None:
    logger.debug(
        "Loading model components",
        assets=[
            f
            for f in assets_or_predictors
            if _functions[f.__name__]["type"] == ComponentType.ASSET
        ],
        predictors=[
            f.__name__
            for f in assets_or_predictors
            if _functions[f.__name__]["type"] == ComponentType.PREDICTOR
        ],
        namespace=namespace,
    )
    for asset_or_predictor, config in assets_or_predictors.items():
        _load_one_asset_or_predictor(
            namespace,
            asset_or_predictor,
            config,
            parent_artifact_config,
        )


def _cleanup_asset_namespace(name, generator):
    try:
        logger.debug("Cleaning up asset", name=name)
        next(generator)
    except StopIteration:
        pass
    except Exception as e:
        logger.error(
            "Failed to clean up component",
            name=name,
            error=str(e),
            error_type=e.__class__.__name__,
        )
        raise


def _register_cleanup_functions(
    namespace: dict[str, dict[frozenset, Any]],
) -> contextlib.ExitStack:
    exit_stack = contextlib.ExitStack()
    for func_name, generator in namespace.get("__cleanup__", {}).items():
        exit_stack.callback(
            lambda name=func_name, gen=generator: _cleanup_asset_namespace(name, gen)
        )
    return exit_stack


def extract_dependencies_to_load(
    assets_or_predictors: dict[Callable, dict[str, Any] | None],
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """
    Extract all artifacts from components in a flat structure for parallel resolution.
    """
    visited = set()
    flattened_predictors = []
    flattened_assets = []
    flattened_artifact_deps = []
    queue = collections.deque(assets_or_predictors.items())
    while queue:
        func, config = queue.popleft()
        config = config or {}
        cache_key = _create_cache_key(config | {"__func_name__": func.__name__})
        if cache_key in visited:
            continue
        visited.add(cache_key)
        dependencies = _functions[func.__name__]["dependencies"]
        for param_name, dep_func, dep_func_args in dependencies:
            if param_name in config:
                # If the dependency is passed in the config, we don't need to resolve it
                continue
            queue.append((dep_func, dep_func_args))
        if _functions[func.__name__]["type"] == ComponentType.ASSET:
            flattened_assets.append((func.__name__, config))
        else:
            flattened_predictors.append((func.__name__, config))
        artifacts = _functions[func.__name__]["artifacts"]
        for param_name, uri, artifact_params in artifacts:
            if param_name in config:
                # If the dependency is passed in the config, we don't need to resolve
                continue
            cache_key = _create_cache_key(artifact_params | {"uri": uri})
            if cache_key in visited:
                continue
            visited.add(cache_key)
            flattened_artifact_deps.append((uri, artifact_params))

    return {
        "predictors": flattened_predictors,
        "assets": flattened_assets,
        "artifacts": flattened_artifact_deps,
    }
