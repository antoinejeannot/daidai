import collections
import contextlib
import contextvars
import functools
import threading
import time
import typing
from collections.abc import Callable, Generator, Iterable
from typing import Any, Final

from daidai.config import CONFIG
from daidai.files import FileDependencyCacheStrategy, load_file_dependency
from daidai.logs import get_logger
from daidai.types import ComponentLoadError, ComponentType, Metadata

logger = get_logger(__name__)

try:
    import pympler.asizeof

    has_pympler = True
except ImportError:
    has_pympler = False
    logger.info("pympler is not installed, memory usage will not be logged")

GLOBAL_LOCK: Final = threading.RLock()
CURRENT_NAMESPACE = contextvars.ContextVar("CURRENT_NAMESPACE", default="global")


class MetaModelManager:
    namespaces: typing.ClassVar = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    functions: typing.ClassVar[dict[str, Metadata]] = {}


def _create_cache_key(args: dict[str, Any] | None) -> frozenset:
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


def _get_from_cache(
    namespace: dict[str, dict[frozenset, Any]],
    func_name: str,
    cache_key: frozenset,
) -> Any | None:
    with GLOBAL_LOCK:
        return namespace.get(func_name, {}).get(cache_key)


def _cache_value(
    namespace: dict[str, dict[frozenset, Any]],
    func_name: str,
    cache_key: frozenset,
    value: Any,
) -> None:
    with GLOBAL_LOCK:
        namespace.setdefault(func_name, {})[cache_key] = value


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
            with GLOBAL_LOCK:
                cleanup_funcs = [
                    (meta, func_name)
                    for func_name, meta in MetaModelManager.functions.items()
                    if meta.get("clean_up")
                ]
                for func_name, meta in cleanup_funcs:
                    if meta.get("clean_up"):
                        self._exit_stack.callback(
                            lambda m=meta, fn=func_name: self._cleanup_artifact(m, fn)
                        )

        with GLOBAL_LOCK:
            for func_name, meta in MetaModelManager.functions.items():
                if meta.get("clean_up"):
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
            if clean_up := meta.get("clean_up"):
                next(clean_up)
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
        with GLOBAL_LOCK:
            namespace = (
                namespace or MetaModelManager.namespaces[CURRENT_NAMESPACE.get()]
            )
            cleanup_items = [
                func_name
                for func_name, metadata in MetaModelManager.functions.items()
                if metadata.get("clean_up")
            ]
        for func_name in cleanup_items:
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
        with GLOBAL_LOCK:
            for func_name, _ in cleanup_items:
                MetaModelManager.functions.pop(func_name, None)
            namespace.clear()

    @staticmethod
    def _load_artifact_or_predictor(
        namespace: dict[str, dict[frozenset, Any]],
        func: Callable | Generator,
        config: dict[str, Any] | None = None,
    ) -> Callable | Generator:
        t0 = time.perf_counter()
        with GLOBAL_LOCK:
            kind = MetaModelManager.functions[func.__name__]["kind"]
        prepared_args = {}
        config = config or {}
        config_cache_key = _create_cache_key(config)
        with GLOBAL_LOCK:
            if cached := _get_from_cache(namespace, func.__name__, config_cache_key):
                logger.debug(
                    "Using cached component",
                    kind=kind.value,
                    name=func.__name__,
                    cache_key=str(config_cache_key),
                    elapsed=round(time.perf_counter() - t0, 9),
                )
                return cached
        logger.debug(
            "Loading component", name=func.__name__, kind=kind.value, config=config
        )
        # whether the function is an artifact or a predictor, it can have files dependencies
        with GLOBAL_LOCK:
            files = MetaModelManager.functions[func.__name__]["files"]
        for param_name, uri, files_params in files:
            logger.debug(
                "Processing files dependency",
                component=func.__name__,
                param_name=param_name,
                dependency=uri,
                params=files_params,
            )
            with GLOBAL_LOCK:
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
            cache_key = _create_cache_key(files_params)
            with GLOBAL_LOCK:
                file_dependency = _get_from_cache(
                    namespace, "file/" + uri, cache_key
                )  # file/ to avoid collision with function names
            if file_dependency:
                logger.debug(
                    "Using cached file",
                    name="file/" + uri,
                    cache_key=str(cache_key),
                    elapsed=round(time.perf_counter() - t0, 9),
                )
            else:
                file_dependency = load_file_dependency(uri, files_params)
                _cache_value(namespace, "file/" + uri, cache_key, file_dependency)
            prepared_args[param_name] = file_dependency
        # For predictors, we don't cache the function itself, just its artifact dependencies
        if kind == ComponentType.PREDICTOR:
            with GLOBAL_LOCK:
                dependencies = MetaModelManager.functions[func.__name__]["dependencies"]
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
                dep_result = ModelManager._load_artifact_or_predictor(
                    namespace, dep_func, dep_func_args
                )
                prepared_args[param_name] = dep_result

            logger.debug("Prepared predictor", name=func.__name__, args=prepared_args)
            prepared_predictor = functools.partial(
                func, **(prepared_args | (config or {}))
            )
            _cache_value(namespace, func.__name__, config_cache_key, prepared_predictor)
            return prepared_predictor

        if kind != ComponentType.ARTIFACT:
            raise ValueError(f"Invalid kind {kind}")
        dependencies = MetaModelManager.functions[func.__name__]["dependencies"]
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
            dep_result = ModelManager._load_artifact_or_predictor(
                namespace, dep_func, dep_func_args
            )
            prepared_args[param_name] = dep_result

        final_args = prepared_args | (config or {})
        logger.debug("Computing artifact", name=func.__name__, args=final_args)
        try:
            result = (
                func.__wrapped__(**final_args)
                if hasattr(func, "__wrapped_component__")
                else func(**final_args)
            )
            if isinstance(result, Generator):
                with GLOBAL_LOCK:
                    MetaModelManager.functions[func.__name__]["clean_up"] = result
                result = next(result)
            # ModelManager._cache_value(namespace, func_name, cache_key, result)
            _cache_value(namespace, func.__name__, config_cache_key, result)
            logger.debug(
                "Component loaded",
                name=func.__name__,
                kind=kind.value,
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
                kind=kind.value,
                error=str(e),
                error_type=e.__class__.__name__,
                config=config,
            )
            raise ComponentLoadError("Failed to load component") from e
