# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cleanDE is a collection of single-file, opinionated, highly-efficient implementations of common data engineering patterns. Python 3.12+, managed with uv.

## Design Principles

- **Folder per pattern, file per technology.** Each pattern (e.g., `bitemporal_diff/`) is a folder. Each technology implementation (e.g., `pandas_impl.py`, `pyarrow_impl.py`) is a standalone file within it. No shared base classes, no internal imports between files.
- **Opinionated over flexible.** One clear implementation per pattern. No plugin systems, no strategy toggles, no configuration matrices.
- **Performance by default.** Prefer vectorized operations, minimal allocations, and batch processing. Avoid row-by-row loops when a set-based approach exists.
- **Pure functions over frameworks.** Patterns take data in, return data out. No ORMs, no pipeline frameworks, no decorators-on-decorators.
- **Correct first, then fast.** Every pattern must handle edge cases (nulls, empty sets, out-of-order arrivals) before optimizing for throughput.
- **No global variables.** All state lives inside functions. Constants, sentinels, and configuration are passed as arguments or defined locally.
- **Pandas + PyArrow only.** All Python implementations use pandas and pyarrow. No polars, no numpy, no delta, no iceberg.
- **Minimal dependencies.** Standard library + pandas + pyarrow. No utility packages for things Python already does.
- **Document every parameter.** Every function parameter gets a clear docstring entry explaining what it is, its expected type, and any defaults or sentinels. The caller should never have to read the function body to understand what to pass.
- **Purely functional.** Functions must not reference external state — no module-level variables, no environment reads, no file I/O inside business logic. Every piece of data a function needs comes in through its parameters. Side effects (if any) happen at the call site, not inside the pattern.
- **Contract first, then tests, then code.** Every new pattern starts by defining the public API — function signatures, parameter types, return types — before any implementation. Next, write a complete set of unit tests covering normal cases, edge cases (nulls, empty inputs, duplicates, out-of-order data), and expected errors. Only then write the implementation to make the tests pass. The tests are the spec; the implementation follows from them.
- **Self-documenting code.** Clear variable names and type hints over docstring novels. A reader should understand the pattern from the code itself.

## Repository Structure

```
cleanDE/
  <pattern_name>/
    pandas_impl.py
    pyarrow_impl.py
    test/
      pandas_test.py
      pyarrow_test.py
```

Python files use the `_impl` suffix to avoid shadowing their library imports. Test files use the `_test` suffix and live in a `test/` subfolder within each pattern.

Not every pattern needs every technology — only add implementations that make sense for the pattern.

## Commands

- **Run the app:** `uv run hello.py`
- **Add a dependency:** `uv add <package>`
- **Sync environment:** `uv sync`
