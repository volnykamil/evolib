# evolib

Modular Python framework for evolutionary algorithms (genetic algorithms & genetic programming).

## Installation

Install the library (core runtime dependencies):

```bash
pip install .
```

For development (includes linting, typing, tests):

```bash
pip install .[dev]
```

## Tooling

The project enforces code quality via:

* Ruff (format + lint) – configured in `pyproject.toml`.
* Mypy (static typing) – permissive base config, can be tightened over time.
* Pytest + Coverage.

### Running locally

```bash
ruff format .          # auto-format
ruff check .           # lint
mypy src/evolib        # type checking
pytest -vv             # tests with coverage (see pyproject addopts)
```

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/PR to `main`:

1. Install `.[dev]` dependencies
2. Ruff format check & lint
3. Mypy type checking
4. Pytest with coverage artifact upload

## Typing

`py.typed` marker is included for PEP 561 so downstream consumers get type info.

## Roadmap Ideas

* RNG injection for reproducibility
* Additional operators (distance metrics, advanced crossovers)
* Engine orchestration & integration tests
* Stricter mypy settings once annotations mature

---
MIT License.
