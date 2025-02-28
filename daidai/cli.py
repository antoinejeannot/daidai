import importlib
import logging
import sys
from pathlib import Path

from daidai.logs import get_logger
from daidai.managers import _functions

logger = get_logger(__name__)
try:
    import click

    print("hello")
except ImportError as e:
    has_click = False
    logger.warning(
        "The `click` package is not installed. "
        "Please install `daidai[cli]` to use the CLI."
    )
    raise ImportError(
        "The `click` package is not installed. "
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


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
def cli(debug):
    """DaiDai CLI tool for managing model artifacts and dependencies."""
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.group()
def list():
    """List various components of the DaiDai system."""
    pass


@list.command("components")
@click.option("-m", "--module", required=True, help="Python module or file to analyze")
@click.option(
    "--type",
    "component_type",
    type=click.Choice(["artifacts", "predictors", "files", "all"]),
    default="all",
    help="Type of components to list",
)
def list_components(module, component_type):
    """List all components (artifacts and predictors) in the module."""
    try:
        import_module_from_path(module)

        for name, metadata in _functions.items():
            if component_type == "all" or metadata["type"].value == component_type:
                click.echo(f"{metadata['type'].value.capitalize()}: {name}")

                # Show dependencies if any
                if metadata["dependencies"]:
                    click.echo("  Dependencies:")
                    for param_name, dep_func, _ in metadata["dependencies"]:
                        click.echo(f"    {param_name}: {dep_func.__name__}")

                # Show file dependencies if any
                if metadata["files"]:
                    click.echo("  File dependencies:")
                    for param_name, uri, _ in metadata["files"]:
                        click.echo(f"    {param_name}: {uri}")

                click.echo("")  # Empty line for readability
    except Exception as e:
        logger.error(f"Error listing components: {e!s}")
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
