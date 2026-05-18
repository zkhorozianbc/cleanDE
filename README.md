# CleanDE

**Single-file, vectorized implementations of the data engineering patterns every warehouse ends up rewriting.** Pure pandas and PyArrow functions — data in, data out. No frameworks, no ORMs, no pipeline DSLs, no inheritance trees. Read the file, copy it into your project, ship it.

## Why CleanDE

- **Read it in one sitting.** Each pattern is ~100–280 lines in a single file. No jumping between five modules to follow control flow.
- **Vectorized, not row-by-row.** Set-based joins and batch merges over Python loops. Both backends are written for throughput, not pedagogy.
- **Edge cases handled.** Nulls, composite keys, empty inputs, out-of-order arrivals, duplicate detection, no-op updates — all covered by tests, not docstrings.
- **152 tests, two backends.** Every pattern ships with parallel pandas and PyArrow implementations and identical test suites.
- **Zero framework lock-in.** Pure functions. Call them from Airflow, Dagster, a notebook, a Lambda, or a cron job — the pattern doesn't care.
- **Two dependencies.** `pandas` and `pyarrow`. That's it.

## Patterns

| Pattern | What it does | pandas | PyArrow |
|---|---|---|---|
| [SCD Type 1](cleanDE/scd_type1/) | Destructive overwrite — keep only the current state | [96 lines](cleanDE/scd_type1/pandas_impl.py) | [94 lines](cleanDE/scd_type1/pyarrow_impl.py) |
| [SCD Type 2](cleanDE/scd_type2/) | Row-versioned history with `valid_from` / `valid_to` / `is_current` | [211 lines](cleanDE/scd_type2/pandas_impl.py) | [263 lines](cleanDE/scd_type2/pyarrow_impl.py) |
| [SCD Type 3](cleanDE/scd_type3/) | Previous-value columns with an effective date — limited history, flat schema | [277 lines](cleanDE/scd_type3/pandas_impl.py) | [358 lines](cleanDE/scd_type3/pyarrow_impl.py) |
| [EAV SCD Type 2](cleanDE/eav_scd_type2/) | SCD Type 2 over entity-attribute-value rows, versioning each `(entity, attribute)` pair independently | [216 lines](cleanDE/eav_scd_type2/pandas_impl.py) | [266 lines](cleanDE/eav_scd_type2/pyarrow_impl.py) |
| [Bi-temporal Diff](cleanDE/bitemporal_diff/) | Two-axis versioning — valid time (real-world truth) × transaction time (system knowledge) for full audit trails and retroactive corrections | [209 lines](cleanDE/bitemporal_diff/pandas_impl.py) | [272 lines](cleanDE/bitemporal_diff/pyarrow_impl.py) |

## Install & Test

```bash
git clone https://github.com/zkhorozianbc/CleanDE.git
cd CleanDE
uv sync
uv run pytest        # 152 tests
```

## Usage

### SCD Type 1 — overwrite current state

```python
from cleanDE.scd_type1.pandas_impl import scd_type1

result = scd_type1(dimension, incoming, keys=["customer_id"])
```

### SCD Type 2 — versioned history

```python
from datetime import datetime
from cleanDE.scd_type2.pandas_impl import scd_type2

result = scd_type2(
    dimension,
    incoming,
    keys=["customer_id"],
    effective_time=datetime.utcnow(),
)
```

### SCD Type 3 — previous-value columns

```python
from datetime import datetime
from cleanDE.scd_type3.pandas_impl import scd_type3

result = scd_type3(
    dimension,
    incoming,
    keys=["customer_id"],
    effective_time=datetime.utcnow(),
    tracked_columns=["region", "tier"],
)
```

### EAV SCD Type 2 — per-attribute versioning

```python
from datetime import datetime
from cleanDE.eav_scd_type2.pandas_impl import eav_scd_type2

result = eav_scd_type2(
    dimension,
    incoming,
    entity_key="customer_id",
    effective_time=datetime.utcnow(),
)
```

### Bi-temporal Diff — valid time × transaction time

```python
from datetime import datetime
from cleanDE.bitemporal_diff.pandas_impl import bitemporal_diff

result = bitemporal_diff(
    existing,
    incoming,
    keys=["policy_id"],
    txn_time=datetime.utcnow(),
)
```

### Same API on PyArrow

```python
from cleanDE.scd_type2.pyarrow_impl import scd_type2
# identical signature, pyarrow.Table in/out
```

## Design Principles

- **One file per implementation.** No shared base classes, no internal imports between patterns.
- **Pure functions.** Data in, data out. No side effects, no global state, no environment reads.
- **Vectorized by default.** Set-based merges over row-by-row loops. Batch operations over iterative updates.
- **Correct first, then fast.** Every edge case is covered by a test before any optimization.
- **pandas + PyArrow only.** No polars, no numpy, no delta, no iceberg.
