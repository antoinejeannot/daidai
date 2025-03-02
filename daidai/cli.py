import collections
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Literal

from daidai.logs import get_logger
from daidai.managers import _functions
from daidai.types import ArtifactCacheStrategy, ComponentType, Metadata

logger = get_logger(__name__)

try:
    import click
    from rich.console import Console
    from rich.text import Text
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
        "assets": [],
        "predictors": [],
        "artifacts": [],
    }

    for name, metadata in functions.items():
        component_info = {
            "name": name,
            "type": metadata["type"].value,
            "dependencies": [],
            "artifacts": [],
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
        if metadata["artifacts"]:
            for param_name, uri, args in metadata["artifacts"]:
                component_info["artifacts"].append(
                    {"uri": uri, "param_name": param_name, "config": args}
                )

        if metadata["type"] == ComponentType.ASSET:
            result["assets"].append(component_info)
        elif metadata["type"] == ComponentType.PREDICTOR:
            result["predictors"].append(component_info)
        elif metadata["type"] == ComponentType.ARTIFACT:
            result["artifacts"].append(component_info)
        else:
            raise ValueError(f"Unknown component type: {metadata}")

    return result


def render_rich_components(
    data: dict[str, list[dict[str, Any]]],
    component_type: ComponentType | Literal["all"],
    cache_strategy: ArtifactCacheStrategy | None,
) -> None:
    """Render components using rich for terminal display."""
    console = Console()
    components_tree = Tree("📦 Daidai Components")
    all_artifacts = {}
    for component_type_s in data:
        for component in data[component_type_s]:
            for artifact in component["artifacts"]:
                artifact_uri = artifact["uri"]
                if (
                    cache_strategy
                    and cache_strategy != artifact["config"]["cache_strategy"]
                ):
                    continue
                if artifact_uri not in all_artifacts:
                    all_artifacts[artifact_uri] = {
                        "uri": artifact_uri,
                        "used_by": [],
                        "cache_strategies": set(),
                    }

                all_artifacts[artifact_uri]["used_by"].append(
                    {
                        "component_type": component_type_s.rstrip("s"),
                        "component_name": component["name"],
                        "param_name": artifact["param_name"],
                    }
                )
                all_artifacts[artifact_uri]["cache_strategies"].add(
                    artifact["config"]["cache_strategy"].value
                )

    if component_type in (ComponentType.ARTIFACT, "all"):
        artifacts_tree = components_tree.add("📄 Artifacts")
        for artifact_uri, artifact_info in all_artifacts.items():
            artifact_node = artifacts_tree.add(f"[bold cyan]{artifact_uri}[/]")
            strategies_str = ", ".join(artifact_info["cache_strategies"])
            artifact_node.add(f"Cache strategies: {strategies_str}")
            usage_node = artifact_node.add("Used by:")
            for usage in artifact_info["used_by"]:
                usage_node.add(
                    f"[{'green' if usage['component_type'] == 'asset' else 'magenta'}]{usage['component_name']}[/] "
                    f"({usage['component_type']}) as [yellow]{usage['param_name']}[/]"
                )

    if component_type in (ComponentType.ASSET, "all"):
        assets_tree = components_tree.add("🧩 Assets")
        for asset in data["assets"]:
            asset_node = assets_tree.add(f"[bold green]{asset['name']}[/]")

            if asset["dependencies"]:
                deps_node = asset_node.add("Dependencies")
                for dep in asset["dependencies"]:
                    config_str = (
                        ", ".join(f"{k}={v}" for k, v in dep["config"].items())
                        if dep["config"]
                        else "default"
                    )
                    deps_node.add(
                        f"[yellow]{dep['param_name']}[/]: [green]{dep['name']}[/] ({dep['type']}) - {config_str}"
                    )

            if asset["artifacts"]:
                artifacts_node = asset_node.add("Artifacts")
                for artifact in asset["artifacts"]:
                    if (
                        cache_strategy
                        and cache_strategy != artifact["config"]["cache_strategy"]
                    ):
                        continue
                    artifacts_node.add(
                        f"[yellow]{artifact['param_name']}[/]: [blue]{artifact['uri']}[/] - Cache: {artifact['config']['cache_strategy'].value}"
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

            if predictor["artifacts"]:
                artifacts_node = predictor_node.add("Artifacts")
                for artifact in predictor["artifacts"]:
                    cache_strategy = artifact["config"]["cache_strategy"]
                    strat_value = (
                        cache_strategy.value
                        if hasattr(cache_strategy, "value")
                        else cache_strategy
                    )
                    artifacts_node.add(
                        f"[yellow]{artifact['param_name']}[/]: [blue]{artifact['uri']}[/] - Cache: {strat_value}"
                    )

    console.print(components_tree)


def render_raw_components(
    data: dict[str, list[dict[str, Any]]],
    component_type: ComponentType | Literal["all"],
    cache_strategy: ArtifactCacheStrategy | None,
) -> None:
    console = Console()
    output_sections = collections.defaultdict(set)

    if component_type in (ComponentType.ARTIFACT, "all"):
        for component_type_s in data:
            for component in data[component_type_s]:
                for artifact in component["artifacts"]:
                    if (
                        cache_strategy
                        and "cache_strategy" in artifact["config"]
                        and cache_strategy != artifact["config"]["cache_strategy"]
                    ):
                        continue
                    output_sections["artifacts"].add(artifact["uri"])

    if component_type in (ComponentType.ASSET, "all"):
        for asset in data["assets"]:
            output_sections["assets"].add(asset["name"])

    if component_type in (ComponentType.PREDICTOR, "all"):
        for predictor in data["predictors"]:
            output_sections["predictors"].add(predictor["name"])

    for i, (section, items) in enumerate(output_sections.items()):
        if i > 0:
            console.print()

        if component_type == "all":
            section_header = Text()
            section_header.append("[", style="bold white")
            section_header.append(section, style="bold cyan")
            section_header.append("]", style="bold white")
            console.print(section_header)

        for item in sorted(items):
            if section == "artifacts":
                console.print(item, style="cyan")
            elif section == "assets":
                console.print(item, style="green")
            elif section == "predictors":
                console.print(item, style="magenta")


@click.group()
def cli(): ...


@cli.command()
@click.option("-m", "--module", required=True, help="Python module or file to analyze")
@click.argument(
    "component_type",
    type=click.Choice(["assets", "predictors", "artifacts", "all"]),
    default="all",
    required=False,
)
@click.option(
    "-c",
    "--cache-strategy",
    type=click.Choice([s.value for s in ArtifactCacheStrategy]),
    default=None,
    help="Filter artifacts by cache strategy",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["raw", "tree"]),
    default="tree",
    help="Output format",
)
def list(module, component_type, cache_strategy, format):
    """List all components (predictors, assets and artifacts) in the module.

    Supports multiple output formats including text and Markdown.
    """
    import_module_from_path(module)
    cache_strategy = ArtifactCacheStrategy(cache_strategy) if cache_strategy else None
    component_type = (
        ComponentType(component_type.rstrip("s")) if component_type != "all" else "all"
    )
    components = collect_components(_functions)
    match format:
        case "tree":
            render_rich_components(components, component_type, cache_strategy)
        case "raw":
            render_raw_components(components, component_type, cache_strategy)
        case _:
            raise ValueError(f"Unknown output format: {format}")


if __name__ == "__main__":
    cli()
