# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cleanDE is a collection of single-file, opinionated, highly-efficient implementations of common data engineering patterns. Python 3.12+, managed with uv.

## Design Principles

- **Folder per pattern, file per technology.** Each pattern (e.g., `bitemporal_diff/`) is a folder. Each technology implementation (e.g., `polars.py`, `pandas.py`, `postgres.sql`) is a standalone file within it. No shared base classes, no internal imports between files.
- **Opinionated over flexible.** One clear implementation per pattern. No plugin systems, no strategy toggles, no configuration matrices.
- **Performance by default.** Prefer vectorized operations, minimal allocations, and batch processing. Avoid row-by-row loops when a set-based approach exists.
- **Pure functions over frameworks.** Patterns take data in, return data out. No ORMs, no pipeline frameworks, no decorators-on-decorators.
- **Correct first, then fast.** Every pattern must handle edge cases (nulls, empty sets, out-of-order arrivals) before optimizing for throughput.
- **No global variables.** All state lives inside functions. Constants, sentinels, and configuration are passed as arguments or defined locally.
- **Pandas + PyArrow only.** All Python implementations use pandas and pyarrow. No polars, no numpy-only files.
- **Minimal dependencies.** Standard library + pandas + pyarrow. No utility packages for things Python already does.
- **Self-documenting code.** Clear variable names and type hints over docstring novels. A reader should understand the pattern from the code itself.

## Repository Structure

```
<pattern_name>/
  pandas_impl.py
  pyarrow_impl.py
  postgres.sql
  mysql.sql
  sqlserver.sql
  delta_impl.py
  parquet_impl.py
  iceberg_impl.py
```

Python files use the `_impl` suffix to avoid shadowing their library imports.

Not every pattern needs every technology — only add implementations that make sense for the pattern.

## Commands

- **Run the app:** `uv run hello.py`
- **Add a dependency:** `uv add <package>`
- **Sync environment:** `uv sync`
