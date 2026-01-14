# Paritea

A collection of tools to handle and manage faults in Clifford ZX diagrams, as well as compute related properties.

Currently supported features:
- Computing **Pauli webs**, with options to accelerate the computation by providing hints about the structure of the diagram
- Specification of **noise models** as a collection of independent error sources
- Extraction of **`stim` DEM-like objects** from noise models that describe the interaction of faults with detectors and diagram boundaries
- Checking **fault equivalence** of two noise models on semantically equivalent diagrams

## Installation
Using your package manager of choice, add the repository as a dependency to your project. For example with `uv`:
```shell
uv add git+https://github.com/paritea/paritea
```
The project does not have any releases yet, so this remains sole way of installation. For options on how to install different versions based on git refs, see [the corresponding `uv` documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/#git).

## Developer Setup
The project uses the package manager [`uv`](https://github.com/astral-sh/uv) for managing dependencies.
Thus, **`uv` must be installed** (see for example [*their installation guide*](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)).

> First time users of `uv` virtual environments may find ["Using a virtual environment"](https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment) a useful guide.
In particular, a virtual environment must be active to use the packages contained within, but can remain inactive when managing dependencies.

Developers should then, after cloning this repository, install all required dependencies via:
```shell
uv sync --group all
```
This will by default install developer dependencies as well, such as test and code style dependencies.

### Linting and code formatting
The project uses [`ruff`](https://github.com/astral-sh/ruff) for both linting and formatting with (so far) minimal configuration changes.

After setting up your environment and activating it (see [*above*](#developer-setup)), you can check linting standards with:
```shell
ruff check
```
and formatting (including auto-fixes) with:
```shell
ruff format
```

### Running tests
The project uses [`pytest`](https://pytest.org) for testing.

Within an active virtual environment, simply run:
```shell
pytest
```
Common needs include:
- increasing verbosity: use `-v` flag
- restricting test runs to a pattern: use `-k <glob pattern>` flag
- allowing standard output (e.g. `print`): use `-s` flag

For more information refer to the [*pytest config documentation*](https://docs.pytest.org/en/stable/reference/customize.html).
