import functools
import inspect
import typing
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Annotated, Any, BinaryIO, TextIO

from daidai.logs import get_logger
from daidai.managers import (
    Metadata,
    _current_namespace,
    _functions,
    _load_one_artifact_or_predictor,
    _namespaces,
)
from daidai.types import VALID_TYPES, ComponentType

logger = get_logger(__name__)

P = typing.ParamSpec("P")
R = typing.TypeVar("R")


def component_decorator(kind: ComponentType):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _functions[func.__name__] = Metadata(
            dependencies=[],
            kind=kind,
            function=func,
            files=[],
            clean_up=None,
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
                _functions[func.__name__]["files"].append(
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
            _functions[func.__name__]["dependencies"].append(
                (param_name, dep_func, dep_defaults | dep_func_args)
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
            return result() if kind == ComponentType.PREDICTOR else result

        wrapper.__wrapped_component__ = True
        return wrapper

    return decorator


artifact = component_decorator(ComponentType.ARTIFACT)
predictor = component_decorator(ComponentType.PREDICTOR)
