<p align="center">
    <img src="https://raw.githubusercontent.com/antoinejeannot/daidai/assets/logo.svg" alt="daidai logo" width="200px">
</p>
<h1 align="center"> daidai 🍊</h1>
<p align="center">
  <em>Modern dependency & assets management library for MLOps</em>
</p>

<p align="center">
<a href="https://github.com/antoinejeannot/daidai/actions/workflows/tests.yml"><img src="https://github.com/antoinejeannot/daidai/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
<a href="https://pypi.org/project/daidai/"><img src="https://img.shields.io/pypi/v/daidai.svg" alt="PyPI version"></a>
<a href="https://pypi.org/project/daidai/"><img src="https://img.shields.io/pypi/pyversions/daidai.svg" alt="Python Versions"></a>
<a href="https://github.com/antoinejeannot/daidai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/antoinejeannot/daidai.svg" alt="License"></a>
<img alt="AI MLOps" src="https://img.shields.io/badge/AI-MLOps-purple">
</p>

**daidai 🍊** is a minimalist, type-safe dependency management system for AI/ML components that streamlines workflow development with dependency injection, intelligent caching and seamless file handling.

🚧 **daidai** is still very much a work in progress and is definitely not prod-ready. It is currently developed as a _[selfish software](https://every.to/source-code/selfish-software)_ to become my personal go-to MLOps library, but feel free to give it a try :) 🚧

## Why daidai?

Built for both rapid prototyping and production ML workflows, **daidai 🍊**:

- 🚀 **Accelerates Development** - Reduces iteration cycles with zero-config caching
- 🧩 **Simplifies Architecture** - Define reusable components with clear dependencies
- 🔌 **Works Anywhere** - Seamless integration with cloud/local storage via fsspec
- 🧠 **Stays Out of Your Way** - Type-hint based DI means minimal boilerplate
- 🧹 **Manages Resources** - Automatic cleanup prevents leaks and wasted compute
- 🛡️ **Prioritizes Safety** - Strong typing catches issues at compile time, not runtime
- 🧪 **Enables Testing** - Inject mock dependencies with ease for robust unit testing
- 🎯 **Principle of Least Surprise** - Intuitive API that behaves exactly as you think it should work

> **daidai** is named after the Japanese word for "orange" 🍊, a fruit that is both sweet and sour, just like the experience of managing dependencies in ML projects. <br/>It is being developed with user happiness in mind, while providing great flexibility and minimal boilerplate. It has been inspired by [pytest](https://github.com/pytest-dev/pytest), [modelkit](https://github.com/Cornerstone-OnDemand/modelkit), [dependency injection & testing](https://antoinejeannot.github.io/nuggets/dependency_injection_and_testing.html) principles and functional programming.

## Installation

```bash
pip install daidai
```

## Quick Start

```python
import base64
from typing import Annotated, Any

import openai

from daidai import ModelManager, artifact, predictor

# Define artifacts which are long-lived objects
# that can be used by multiple predictors, or other artifacts


@artifact
def openai_client(**configuration: dict[str, Any]) -> openai.OpenAI:
    return openai.OpenAI(**configuration)


# Fetch a distant file from HTTPS, but it can be from any source: local, S3, GCS, Azure, FTP, HF Hub, etc.
@artifact
def dogo_picture(
    picture: Annotated[
        bytes,
        "https://images.pexels.com/photos/220938/pexels-photo-220938.jpeg",
        {"cache_strategy": "no_cache"},
    ],
) -> str:
    return base64.b64encode(picture).decode("utf-8")


# Define a predictor that depends on the previous artifacts
# which are automatically loaded and passed as an argument


@predictor
def ask(
    message: str,
    dogo_picture: Annotated[str, dogo_picture],
    client: Annotated[openai.OpenAI, openai_client, {"timeout": 5}],
    model: str = "gpt-4o-mini",
) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{dogo_picture}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        model=model,
    )
    return response.choices[0].message.content


# daidai takes care of loading dependencies & injecting artifacts!
print(ask("Hello, what's in the picture ?"))
# >>> The picture features a dog with a black and white coat.

# Or manage lifecycle with context manager for production usage
# all predictors, artifacts and files are automatically loaded and cleaned up
with ModelManager(preload=[ask]):
    print(ask("Hello, what's in the picture ?"))

# or manually pass dependencies
my_other_openai_client = openai.OpenAI(timeout=0.1)
print(ask("Hello, what's in the picture ?", client=my_other_openai_client))
# >>> openai.APITimeoutError: Request timed out.
# OOOPS, the new client timed out, of course :-)
```

## Roadmap

- [ ] Add tests (unit, integration, e2e)
- [ ] Clean things up now that the UX has landed
- [ ] Add docs
- [ ] Handle multi-threading & concurrency with locks
- [ ] Add support for async components
- [ ] Add a cookbook with common patterns & recipes
- [ ] Enjoy the fruits of my labor 🍊

<!--

## Core Concepts

### Components

#### Artifacts

Long-lived objects (models, embeddings, tokenizers) that are:

- Computed once and cached
- Automatically cleaned up when no longer needed
- Can have file dependencies and other artifacts as dependencies

#### Predictors

Functions that:

- Use artifacts as dependencies
- Are not cached themselves
- Can be called repeatedly with different inputs

### File Dependencies

Support for multiple file sources and caching strategies:

```python
@artifact
def load_embeddings(
    # Load from S3, keep on disk permanently
    embeddings: Annotated[
        Path,
        "s3://bucket/embeddings.npy",
        {"strategy": "on_disk"}
    ],
    # Load text file into memory as string
    vocab: Annotated[
        str,
        "gs://bucket/vocab.txt",
        {"strategy": "in_memory"}
    ]
):
    return {"embeddings": np.load(embeddings), "vocab": vocab.split()}
```

Available strategies:

- `on_disk` - Download and keep locally
- `on_disk_temporary` - Download temporarily
- `in_memory` - Load file contents into RAM
- `in_memory_stream` - Stream file contents via a generator

### Dependency Resolution

Components can depend on each other with clean syntax:

```python
@artifact
def tokenizer(vocab_file: Annotated[Path, "s3://bucket/vocab.txt"]):
    return Tokenizer.from_file(vocab_file)

@artifact
def embeddings(
    embedding_file: Annotated[Path, "s3://bucket/embeddings.npy"],
    tokenizer=tokenizer  # Automatically resolved
):
    # tokenizer is automatically loaded
    return Embeddings(embedding_file, tokenizer)

@predictor
def embed_text(
    text: str,
    embeddings=embeddings  # Automatically resolved
):
    return embeddings.embed(text)
```

### Namespace Management

```python
# Load components in different namespaces
with ModelManager([model_a], namespace="prod"):
    with ModelManager([model_b], namespace="staging"):
        # Use both without conflicts
        prod_result = predict_a("test")
        staging_result = predict_b("test")
```

## Advanced Usage

### Custom Configuration

```python
# Override default parameters
result = predict("input", model=load_model(model_path="local/path/model.pkl"))

# Or with ModelManager
with ModelManager({load_model: {"model_path": "local/path/model.pkl"}}):
    result = predict("input")  # Uses custom model path
```

### Generator-based Cleanup

```python
@artifact
def database_connection(url: str):
    conn = create_connection(url)
    try:
        yield conn  # Return the connection
    finally:
        conn.close()  # Automatically called during cleanup
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](https://github.com/daidai-project/daidai/blob/main/CONTRIBUTING.md).

## License

MIT -->
