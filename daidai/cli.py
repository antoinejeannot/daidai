import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Literal

from daidai.logs import get_logger
from daidai.managers import _functions
from daidai.types import ComponentType, FileDependencyCacheStrategy, Metadata

logger = get_logger(__name__)

try:
    import click
    from rich.console import Console
    from rich.tree import Tree

except ImportError as e:
    missing_package = str(e).split("'")[1] if "'" in str(e) else "required package"
    logger.warning(
        f"The `{missing_package}` package is not installed. "
        "Please install `daidai[cli]` to use the CLI."
    )
    raise ImportError(
        f"Missing required package: {missing_package}. "
        "Please install `daidai[cli]` to use the CLI."
    ) from e


def import_module_from_path(module_path: str) -> None:
    """Import a module from a file path to register decorators."""
    if module_path.endswith(".py"):
        module_name = Path(module_path).stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}") from e


def collect_components(
    functions: dict[str, Metadata],
) -> dict[str, list[dict[str, Any]]]:
    """Collect all components of the specified type."""
    result = {
        "artifacts": [],
        "predictors": [],
        "files": [],
    }

    for name, metadata in functions.items():
        component_info = {
            "name": name,
            "type": metadata["type"].value,
            "dependencies": [],
            "files": [],
        }
        if metadata["dependencies"]:
            for param_name, dep_func, args in metadata["dependencies"]:
                component_info["dependencies"].append(
                    {
                        "name": dep_func.__name__,
                        "param_name": param_name,
                        "type": functions[dep_func.__name__]["type"].value,
                        "config": args,
                    }
                )
        if metadata["files"]:
            for param_name, uri, args in metadata["files"]:
                component_info["files"].append(
                    {"uri": uri, "param_name": param_name, "config": args}
                )

        if metadata["type"] == ComponentType.ARTIFACT:
            result["artifacts"].append(component_info)
        elif metadata["type"] == ComponentType.PREDICTOR:
            result["predictors"].append(component_info)
        elif metadata["type"] == ComponentType.FILE:
            result["files"].append(component_info)
        else:
            raise ValueError(f"Unknown component type: {metadata}")

    return result


def render_rich_components(
    data: dict[str, list[dict[str, Any]]],
    component_type: ComponentType | Literal["all"],
    cache_strategy: FileDependencyCacheStrategy | None,
) -> None:
    """Render components using rich for terminal display."""
    console = Console()
    components_tree = Tree("📦 Daidai Components")
    all_files = {}
    for component_type_s in data:
        for component in data[component_type_s]:
            for file in component["files"]:
                file_uri = file["uri"]
                if (
                    cache_strategy
                    and cache_strategy != file["config"]["cache_strategy"]
                ):
                    continue
                if file_uri not in all_files:
                    all_files[file_uri] = {
                        "uri": file_uri,
                        "used_by": [],
                        "cache_strategies": set(),
                    }

                all_files[file_uri]["used_by"].append(
                    {
                        "component_type": component_type_s.rstrip("s"),
                        "component_name": component["name"],
                        "param_name": file["param_name"],
                    }
                )
                all_files[file_uri]["cache_strategies"].add(
                    file["config"]["cache_strategy"].value
                )

    # Files (new top-level node)
    if component_type in (ComponentType.FILE, "all"):
        files_tree = components_tree.add("📄 Files")
        for file_uri, file_info in all_files.items():
            file_node = files_tree.add(f"[bold cyan]{file_uri}[/]")

            # Show cache strategies
            strategies_str = ", ".join(file_info["cache_strategies"])
            file_node.add(f"Cache strategies: {strategies_str}")

            # Show which components use this file
            usage_node = file_node.add("Used by:")
            for usage in file_info["used_by"]:
                usage_node.add(
                    f"[{'green' if usage['component_type'] == 'artifact' else 'magenta'}]{usage['component_name']}[/] "
                    f"({usage['component_type']}) as [yellow]{usage['param_name']}[/]"
                )

    if component_type in (ComponentType.ARTIFACT, "all"):
        artifacts_tree = components_tree.add("🧩 Artifacts")
        for artifact in data["artifacts"]:
            artifact_node = artifacts_tree.add(f"[bold green]{artifact['name']}[/]")

            if artifact["dependencies"]:
                deps_node = artifact_node.add("Dependencies")
                for dep in artifact["dependencies"]:
                    config_str = (
                        ", ".join(f"{k}={v}" for k, v in dep["config"].items())
                        if dep["config"]
                        else "default"
                    )
                    deps_node.add(
                        f"[yellow]{dep['param_name']}[/]: [green]{dep['name']}[/] ({dep['type']}) - {config_str}"
                    )

            if artifact["files"]:
                files_node = artifact_node.add("Files")
                for file in artifact["files"]:
                    if (
                        cache_strategy
                        and cache_strategy != file["config"]["cache_strategy"]
                    ):
                        continue
                    files_node.add(
                        f"[yellow]{file['param_name']}[/]: [blue]{file['uri']}[/] - Cache: {file['config']['cache_strategy'].value}"
                    )

    if component_type in (ComponentType.PREDICTOR, "all"):
        predictors_tree = components_tree.add("🔮 Predictors")
        for predictor in data["predictors"]:
            predictor_node = predictors_tree.add(
                f"[bold magenta]{predictor['name']}[/]"
            )

            if predictor["dependencies"]:
                deps_node = predictor_node.add("Dependencies")
                for dep in predictor["dependencies"]:
                    config_str = (
                        ", ".join(f"{k}={v}" for k, v in dep["config"].items())
                        if dep["config"]
                        else "default"
                    )
                    deps_node.add(
                        f"[yellow]{dep['param_name']}[/]: [green]{dep['name']}[/] ({dep['type']}) - {config_str}"
                    )

            if predictor["files"]:
                files_node = predictor_node.add("Files")
                for file in predictor["files"]:
                    cache_strategy = file["config"]["cache_strategy"]
                    strat_value = (
                        cache_strategy.value
                        if hasattr(cache_strategy, "value")
                        else cache_strategy
                    )
                    files_node.add(
                        f"[yellow]{file['param_name']}[/]: [blue]{file['uri']}[/] - Cache: {strat_value}"
                    )

    console.print(components_tree)


@click.group()
def cli(): ...


@cli.command()
@click.option("-m", "--module", required=True, help="Python module or file to analyze")
@click.argument(
    "component_type",
    type=click.Choice(["artifacts", "predictors", "files", "all"]),
    default="all",
    required=False,
)
@click.option(
    "-c",
    "--cache-strategy",
    type=click.Choice([s.value for s in FileDependencyCacheStrategy]),
    default=None,
    help="Filter files by cache strategy",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["text", "rich"]),
    default="rich",
    help="Output format",
)
def list(module, component_type, cache_strategy, format):
    """List all components (predictors, artifacts and files) in the module.

    Supports multiple output formats including text and Markdown.
    """
    import_module_from_path(module)
    cache_strategy = (
        FileDependencyCacheStrategy(cache_strategy) if cache_strategy else None
    )
    component_type = (
        ComponentType(component_type.rstrip("s")) if component_type != "all" else "all"
    )
    components = collect_components(_functions)
    if format == "rich":
        render_rich_components(components, component_type, cache_strategy)
        return


if __name__ == "__main__":
    cli()
