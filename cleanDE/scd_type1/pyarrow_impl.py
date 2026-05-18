"""
SCD Type 1: apply incoming records to a dimension table by overwriting in
place. No history is kept; the dimension always reflects the current state.

Keys identify a record. Value columns carry the payload. When a value
differs, the existing row is overwritten. Keys missing from the incoming
batch are deleted from the dimension.

Usage:
    dimension = pq.read_table("dim_customer.parquet")
    incoming  = pq.read_table("staging_customer.parquet")
    result    = scd_type1(dimension, incoming, keys=["customer_id"])
"""

import pyarrow as pa


def scd_type1(
    dimension: pa.Table,
    incoming: pa.Table,
    keys: list[str],
) -> pa.Table:
    """Return an updated SCD Type 1 dimension table after applying incoming records.

    Behavior
    --------
    - Key in incoming only        -> inserted.
    - Key in both, values differ  -> dimension row overwritten with incoming values.
    - Key in both, values match   -> row unchanged.
    - Key in dimension only       -> row deleted.

    Parameters
    ----------
    dimension : pa.Table
        Current dimension table containing key and value columns only.
    incoming : pa.Table
        New records with key and value columns. Schema must match dimension.
    keys : list[str]
        Column names that uniquely identify a record (natural key).

    Returns
    -------
    pa.Table
        Updated dimension reflecting inserts, updates, and deletes. Columns
        are ordered as ``keys + value_cols``.
    """
    value_cols = [c for c in incoming.column_names if c not in keys]
    output_cols = keys + value_cols

    # Empty incoming -> everything is deleted.
    if incoming.num_rows == 0:
        return incoming.select(output_cols).slice(0, 0)

    # Empty dimension -> everything is an insert.
    if dimension.num_rows == 0:
        return incoming.select(output_cols)

    dim_values = dimension.select(output_cols)

    # Inserts: keys present only in incoming.
    inserts = incoming.join(dim_values, keys=keys, join_type="left anti")

    # Matched rows: keys present in both. Take incoming values (overwrite).
    # Unchanged rows are also here; using incoming values is a no-op for them.
    matched = incoming.join(dim_values, keys=keys, join_type="left semi")

    # Rows in dimension only are dropped (deletes).

    result = pa.concat_tables(
        [matched.select(output_cols), inserts.select(output_cols)],
        promote_options="permissive",
    )
    return result.select(output_cols)


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    dimension = pa.table({
        "customer_id": pa.array([1, 2, 3]),
        "name":        pa.array(["Alice", "Bob", "Charlie"]),
        "tier":        pa.array(["gold", "silver", "bronze"]),
    })

    incoming = pa.table({
        "customer_id": pa.array([1, 2, 4]),
        "name":        pa.array(["Alice", "Bobby", "Diana"]),
        "tier":        pa.array(["gold", "gold", "platinum"]),
    })

    result = scd_type1(dimension, incoming, keys=["customer_id"])

    print("=== Result ===")
    print(result.sort_by([("customer_id", "ascending")]).to_pandas().to_string(index=False))
