# CleanDE

Single-file, self-contained implementations of common data engineering patterns in pandas and PyArrow. No frameworks, no ORMs, no pipeline DSLs — just pure functions that take data in and return data out.

## Why

Most data engineering libraries bury core logic behind layers of abstraction. CleanDE takes the opposite approach: each pattern is a single file you can read top to bottom, copy into your project, and run. Every implementation is tested, vectorized, and handles edge cases (nulls, empty inputs, out-of-order arrivals) out of the box.

You should use CleanDE if you:
- Want reference implementations of dimension versioning and bi-temporal patterns
- Need a drop-in function for a pipeline without pulling in a framework
- Want to understand how these patterns actually work under the hood

CleanDE is not a library you install and import. It is a collection of standalone implementations meant to be read, learned from, and adapted.

## Patterns

| Pattern | Description | pandas | PyArrow |
|---------|-------------|--------|---------|
| [SCD Type 2](cleanDE/scd_type2/) | Row-versioned slowly changing dimensions with `valid_from`/`valid_to`/`is_current` tracking | [211 lines](cleanDE/scd_type2/pandas_impl.py) | [263 lines](cleanDE/scd_type2/pyarrow_impl.py) |
| [EAV SCD Type 2](cleanDE/eav_scd_type2/) | SCD Type 2 over entity-attribute-value structures, versioning each (entity, attribute) pair independently | [216 lines](cleanDE/eav_scd_type2/pandas_impl.py) | [266 lines](cleanDE/eav_scd_type2/pyarrow_impl.py) |
| [Bi-temporal Diff](cleanDE/bitemporal_diff/) | Two-axis versioning — valid time (real-world truth) and transaction time (system knowledge) — for full audit trails | [209 lines](cleanDE/bitemporal_diff/pandas_impl.py) | [272 lines](cleanDE/bitemporal_diff/pyarrow_impl.py) |

## Quick Start

```bash
# clone and set up
git clone https://github.com/zkhorozianbc/CleanDE.git
cd CleanDE
uv sync

# run tests (90 cases across all patterns)
uv run pytest
```

## Usage

Each implementation is a single function call:

```python
import pandas as pd
from datetime import datetime
from cleanDE.scd_type2.pandas_impl import scd_type2

dimension = pd.read_parquet("dim_customer.parquet")
incoming  = pd.read_parquet("staging_customer.parquet")

result = scd_type2(dimension, incoming, keys=["customer_id"], effective_time=datetime.now())
```

## Design Principles

- **One file per implementation.** No shared base classes, no internal imports between patterns.
- **Pure functions.** Data in, data out. No side effects, no global state, no environment reads.
- **Correct first, then fast.** Edge cases (nulls, composites, empties) are handled before optimizing throughput.
- **Vectorized by default.** Set-based operations over row-by-row loops. Batch merges over iterative updates.
- **pandas + PyArrow only.** No polars, no numpy, no delta, no iceberg.

## Project Structure

```
cleanDE/
  scd_type2/
    pandas_impl.py          # standalone implementation
    pyarrow_impl.py
    test/
      conftest.py           # shared fixtures
      pandas_test.py
      pyarrow_test.py
  eav_scd_type2/
    ...same layout
  bitemporal_diff/
    ...same layout
```

Python files use the `_impl` suffix to avoid shadowing library imports.
