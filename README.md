<center>
<img src="https://raw.githubusercontent.com/antoinejeannot/daidai/assets/logo.svg" alt="daidai logo" width="200px">
</center>

# daidai

**daidai** is a minimalist, type-safe dependency management system for AI/ML components that streamlines workflow development with intelligent caching and seamless file handling.

Still very much WIP - please check back soon for updates! 🚧

## Why daidai?

Built for both rapid prototyping and production ML workflows, daidai:

- 🚀 **Accelerates Development** - Reduces iteration cycles with zero-config caching
- 🧩 **Simplifies Architecture** - Define reusable components with clear dependencies
- 🔌 **Works Anywhere** - Seamless integration with cloud/local storage via fsspec
- 🧠 **Stays Out of Your Way** - Type-hint based DI means minimal boilerplate
- 🧹 **Manages Resources** - Automatic cleanup prevents leaks and wasted compute
- 🛡️ **Prioritizes Safety** - Strong typing catches issues at compile time, not runtime
- 🧪 **Enables Testing** - Inject mock dependencies with ease for robust unit testing
- 🎯 **Principle of Least Surprise** - Intuitive API that behaves exactly as you think it should work

## Installation

```bash
pip install daidai
```

## Quick Start

```python
from daidai import artifact, predictor, ModelManager
from typing import Annotated
from pathlib import Path
import pickle

# Define an artifact with a file dependency
# The file will be automatically downloaded and provided as a Path

@artifact
def my_model_pkl(
    model_path: Annotated[Path, "s3://my-bucket/model.pkl"]
):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Define a predictor that depends on the previous artifact
# which is automatically loaded and passed as an argument

@predictor
def predict(text: str, my_model_pkl):
    return my_model_pkl.predict(text)

# Use directly, daidai takes care of loading dependencies & injecting artifacts!
result = predict("Hello world")

# Or manage lifecycle with context manager for production usage
with ModelManager([predict]):
    result1 = predict("First prediction")
    result2 = predict("Second prediction")

# or manually pass dependencies
model = my_model_pkl(model_path="local/path/model.pkl")
result3 = predict("Third prediction", my_model_pkl=model)
```

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
