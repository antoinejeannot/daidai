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
    _load_one_asset_or_predictor,
    _namespaces,
)
from daidai.types import (
    VALID_TYPES,
    ArtifactCacheStrategy,
    ComponentType,
    Deserialization,
    OpenOptions,
)

logger = get_logger(__name__)

P = typing.ParamSpec("P")
R = typing.TypeVar("R")


class Artifact:
    def __init__(
        self,
        uri: str,
        *,
        force_download: bool | None = CONFIG.force_download,
        cache_strategy: str | ArtifactCacheStrategy = CONFIG.cache_strategy,
        cache_directory: str | Path = CONFIG.cache_dir,
        cache_directory_tmp: str | Path = CONFIG.cache_dir_tmp,
        open_options: dict[str, Any] | None = None,
        deserialization: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
        expected_type: type | None = None,
    ) -> None:
        self.uri = uri
        self.cache_strategy = (
            cache_strategy
            if isinstance(cache_strategy, ArtifactCacheStrategy)
            else ArtifactCacheStrategy(cache_strategy)
        )
        self.cache_directory = (
            cache_directory
            if isinstance(cache_directory, Path)
            else Path(cache_directory)
        )
        self.cache_directory_tmp = (
            cache_directory_tmp
            if isinstance(cache_directory_tmp, Path)
            else Path(cache_directory_tmp)
        )
        self.open_options = open_options or OpenOptions()
        self.deserialization = deserialization or Deserialization()
        self.storage_options = storage_options or {}
        self.force_download = force_download
        self.expected_type = expected_type
        self._validated = False

    def validate(self) -> "Artifact":
        if self.expected_type not in VALID_TYPES:
            raise TypeError(
                f"Expected type {self.expected_type} is not a valid type. "
                f"Must be one of {VALID_TYPES}"
            )
        if self.deserialization.get("format") not in (None, self.expected_type):
            raise TypeError(
                f"Deserialization format {self.deserialization.get('format')}"
                f"does not match the expected type {self.expected_type}"
            )
        self.deserialization["format"] = self.expected_type
        if self.expected_type is Path and self.open_options.get("mode"):
            raise ValueError(
                "Cannot specify mode for Path objects. Use 'str' or 'bytes' instead."
            )
        if self.expected_type is bytes or self.expected_type is BinaryIO:
            self.open_options.setdefault("mode", "rb")
            if self.open_options["mode"] != "rb":
                raise ValueError("Cannot read bytes in text mode. Use 'rb' instead.")
            self.open_options["mode"] = "rb"
        elif self.expected_type is str or self.expected_type is TextIO:
            self.open_options.setdefault("mode", "r")
            if self.open_options["mode"] != "r":
                raise ValueError("Cannot read text in binary mode. Use 'r' instead.")
        if self.open_options.get("mode") not in ("r", "rb", None):
            raise ValueError(
                f"open_options mode must be 'r' or 'rb', got '{self.open_options['mode']}'"
            )
        self._validated = True
        return self

    @property
    def config(self) -> dict[str, Any]:
        if not self._validated:
            raise ValueError("Artifact must be validated before accessing its config")
        return {
            "cache_strategy": self.cache_strategy,
            "cache_directory": self.cache_directory,
            "cache_directory_tmp": self.cache_directory_tmp,
            "open_options": self.open_options,
            "deserialization": self.deserialization,
            "storage_options": self.storage_options,
            "force_download": self.force_download,
        }


class Depends:
    def __init__(
        self,
        asset_or_predictor: Callable,
        *,
        force_download: bool | None = None,
        cache_strategy: str | ArtifactCacheStrategy | None = None,
        cache_directory: str | Path | None = None,
        storage_options: dict[str, Any] | None = None,
        **asset_or_predictor_params,
    ) -> None:
        self.fn = asset_or_predictor
        self._artifact_config = {}
        if force_download:
            self._artifact_config["force_download"] = force_download
        if cache_strategy:
            self._artifact_config["cache_strategy"] = (
                cache_strategy
                if isinstance(cache_strategy, ArtifactCacheStrategy)
                else ArtifactCacheStrategy(cache_strategy)
            )
        if cache_directory:
            self._artifact_config["cache_directory"] = (
                cache_directory
                if isinstance(cache_directory, Path)
                else Path(cache_directory)
            )
        if storage_options:
            self._artifact_config["storage_options"] = storage_options
        fn_defaults = {
            k: v.default
            for k, v in inspect.signature(asset_or_predictor).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        # function defaults (fn_defaults) are overriden by the user-defined defaults (asset_fn_params)
        self._function_config = fn_defaults | asset_or_predictor_params
        self._validated = False

    def validate(self) -> "Depends":
        if not inspect.isfunction(self.fn):
            logger.error(
                f"{self.fn.__name__} must be a user-defined function, got {type(self.fn)}"
            )
            raise TypeError(
                f"{self.fn.__name__} must be a user-defined function, got {type(self.fn)}"
            )

        if self.fn.__name__ not in _functions:
            logger.error(
                f"{self.fn.__name__} is not registered, register it with @asset or @predictor"
            )
            raise ValueError(
                f"{self.fn.__name__} is not registered, register it with @asset or @predictor"
            )
        self._validated = True
        return self

    @property
    def artifact_config(self) -> dict[str, Any]:
        if not self._validated:
            raise ValueError(
                "Depends must be validated before accessing its artifact config"
            )
        return self._artifact_config

    @property
    def function_config(self) -> dict[str, Any]:
        if not self._validated:
            raise ValueError(
                "Depends must be validated before accessing its function config"
            )
        return self._function_config


class Asset(Depends):
    def __init__(
        self,
        asset_fn: Callable,
        *,
        force_download: bool | None = None,
        cache_strategy: str | ArtifactCacheStrategy | None = None,
        cache_directory: str | Path | None = None,
        storage_options: dict[str, Any] | None = None,
        **predictor_fn_params,
    ) -> None:
        super().__init__(
            asset_fn,
            force_download=force_download,
            cache_strategy=cache_strategy,
            cache_directory=cache_directory,
            storage_options=storage_options,
            **predictor_fn_params,
        )

    def validate(self):
        super().validate()
        if _functions[self.fn.__name__]["type"] != ComponentType.ASSET:
            logger.error(
                f"{self.fn.__name__} is not an Asset, but an {_functions[self.fn.__name__]['type'].value.capitalize()}"
            )
            raise TypeError(
                f"{self.fn.__name__} is not an Asset, but an {_functions[self.fn.__name__]['type'].value.capitalize()}"
            )
        return self


class Predictor(Depends):
    def __init__(
        self,
        predictor_fn: Callable,
        *,
        force_download: bool | None = None,
        cache_strategy: str | ArtifactCacheStrategy | None = None,
        cache_directory: str | Path | None = None,
        storage_options: dict[str, Any] | None = None,
        **predictor_fn_params,
    ) -> None:
        super().__init__(
            predictor_fn,
            force_download=force_download,
            cache_strategy=cache_strategy,
            cache_directory=cache_directory,
            storage_options=storage_options,
            **predictor_fn_params,
        )

    def validate(self):
        super().validate()
        if _functions[self.fn.__name__]["type"] != ComponentType.PREDICTOR:
            logger.error(
                f"{self.fn.__name__} is not a Predictor, but an {_functions[self.fn.__name__]['type'].value.capitalize()}"
            )
            raise TypeError(
                f"{self.fn.__name__} is not a Predictor, but an {_functions[self.fn.__name__]['type'].value.capitalize()}"
            )
        return self


def component_decorator(
    component_type: ComponentType,
):
    if component_type not in (ComponentType.ASSET, ComponentType.PREDICTOR):
        raise ValueError(
            f"Invalid component type {component_type}. "
            f"Must be one of {ComponentType.ASSET}, {ComponentType.PREDICTOR}"
        )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _functions[func.__name__] = Metadata(
            dependencies=[],
            type=component_type,
            function=func,
            artifacts=[],
        )
        hints = typing.get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)

        for param_name in sig.parameters:
            if param_name not in hints:
                continue
            annotation = hints[param_name]
            if typing.get_origin(annotation) is not Annotated:
                continue
            typing_args = typing.get_args(annotation)  # at least 2 args
            origin_type = typing.get_origin(typing_args[0]) or typing_args[0]
            dependency = typing_args[1:]
            if (
                typing_args[0] in VALID_TYPES
                or (
                    origin_type is Generator
                    and typing.get_args(typing_args[0])[0] in VALID_TYPES
                )
            ) and isinstance(dependency[0], str | Artifact):
                # potential Artifact spotted
                fn_params = (typing_args[2] or {}) if len(typing_args) > 2 else {}
                artifact = (
                    Artifact(
                        uri=typing_args[1],
                        **fn_params,
                    )
                    if isinstance(typing_args[1], str)
                    else typing_args[1]
                )
                if not artifact.expected_type:
                    artifact.expected_type = (
                        typing_args[0]
                        if origin_type is not Generator
                        else typing.get_args(typing_args[0])[0]
                    )
                artifact.validate()
                _functions[func.__name__]["artifacts"].append(
                    (param_name, artifact.uri, artifact.config)
                )
            elif inspect.isfunction(dependency[0]):
                # Asset or Predictor function spotted passed as function
                dep_func: Callable = dependency[0]
                component_class = (
                    Asset
                    if _functions[dep_func.__name__]["type"] == ComponentType.ASSET
                    else Predictor
                )
                component = component_class(
                    dep_func,
                    **(dependency[1] or {} if len(dependency) > 1 else {}),
                ).validate()
                _functions[func.__name__]["dependencies"].append(
                    (param_name, dep_func, component.function_config)
                )
            elif isinstance(dependency[0], Depends):
                # Depends, Asset or Predictor function spotted passed as instance
                dep: Asset | Predictor = dependency[0].validate()
                def_sig = inspect.signature(dep.fn)
                dep_defaults = {
                    k: v.default
                    for k, v in def_sig.parameters.items()
                    if v.default is not inspect.Parameter.empty
                }
                _functions[func.__name__]["dependencies"].append(
                    (param_name, dep.fn, dep_defaults | dep.function_config)
                )

        @functools.wraps(
            func, assigned=(*functools.WRAPPER_ASSIGNMENTS, "__signature__")
        )
        def wrapper(*args, **kwargs):
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            current_namespace = _current_namespace.get()
            result = _load_one_asset_or_predictor(
                _namespaces[current_namespace],
                func,
                bound_args.arguments,
            )
            return result() if component_type == ComponentType.PREDICTOR else result

        wrapper.__wrapped_component__ = True
        return wrapper

    return decorator


asset = component_decorator(ComponentType.ASSET)
predictor = component_decorator(ComponentType.PREDICTOR)
