# Copilot Instructions for evolib

## Project Overview
- **evolib** is a modular Python framework for genetic algorithms and genetic programming.
- Main components are organized under `src/evolib/`:
  - `core/`: Genotype representations (`genotype.py`), with support for binary, real, integer, and permutation types.
  - `operators/`: Evolutionary operators (mutation, crossover, selection, replacement). Only `mutation.py` is currently implemented.
  - `engine.py`: Intended for orchestration, currently empty.

## Key Patterns & Conventions
- **Genotype Classes**: All genotypes inherit from `Genotype` (ABC). Each has a `random` constructor, `copy`, `__len__`, and `as_array` methods.
- **Mutation Operators**: Inherit from `MutationOperator` (ABC). Each implements a `mutate(genotype)` method, type-checked for the correct genotype.
- **Testing**: Unit tests are in `tests/unit/core/` and `tests/unit/operators/`, using `pytest`. Tests validate type assertions, random generation, and copying.
- **Type Safety**: Constructors assert correct numpy dtypes and value constraints (e.g., permutation validity).
- **Documentation**: Each module and class is documented with docstrings describing purpose and parameters.

## Developer Workflows
- **Build/Install**: Standard Python packaging via `pyproject.toml` (setuptools/wheel).
- **Testing**: Run tests with coverage:
  ```sh
  pytest -vv --cov=evolib --cov-report=term-missing --cov-config=pyproject.toml
  ```
- **Linting/Formatting**: Use `black`, `ruff`, and `mypy` (see `[project.optional-dependencies]` in `pyproject.toml`).
- **Python Path**: Tests use `src` as the Python path (see `pyproject.toml`).

## Integration Points
- **Dependencies**: Relies on `numpy` for genotype data and randomization.
- **Extensibility**: New genotype types or operators should inherit from the relevant ABC and follow the established method signatures.
- **Empty Files**: Some files (e.g., `engine.py`, `crossover.py`, `selection.py`, `individual.py`, `fitness.py`) are placeholders for future expansion.

## Examples
- To add a new mutation operator, inherit from `MutationOperator` and implement `mutate` for the target genotype.
- To add a new genotype, inherit from `Genotype` and implement required methods (`copy`, `__len__`, `as_array`).

## File References
- `src/evolib/core/genotype.py`: Genotype base and implementations
- `src/evolib/operators/mutation.py`: Mutation operators
- `tests/unit/core/test_genotype.py`: Genotype tests

---
_If any section is unclear or missing important details, please provide feedback to improve these instructions._
