"""
SCD Type 1: apply incoming records to a dimension table by overwriting in
place. No history is kept; the dimension always reflects the current state.

Keys identify a record. Value columns carry the payload. When a value
differs, the existing row is overwritten. Keys missing from the incoming
batch are deleted from the dimension.

Usage:
    dimension = pd.read_parquet("dim_customer.parquet")
    incoming  = pd.read_parquet("staging_customer.parquet")
    result    = scd_type1(dimension, incoming, keys=["customer_id"])
"""

import pandas as pd


def scd_type1(
    dimension: pd.DataFrame,
    incoming: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Return an updated SCD Type 1 dimension table after applying incoming records.

    Behavior
    --------
    - Key in incoming only        -> inserted.
    - Key in both, values differ  -> dimension row overwritten with incoming values.
    - Key in both, values match   -> row unchanged.
    - Key in dimension only       -> row deleted.

    Parameters
    ----------
    dimension : pd.DataFrame
        Current dimension table containing key and value columns only.
    incoming : pd.DataFrame
        New records with key and value columns. Schema must match dimension.
    keys : list[str]
        Column names that uniquely identify a record (natural key).

    Returns
    -------
    pd.DataFrame
        Updated dimension reflecting inserts, updates, and deletes. Columns
        are ordered as ``keys + value_cols``.
    """
    value_cols = [c for c in incoming.columns if c not in keys]
    output_cols = keys + value_cols

    # Empty incoming -> everything is deleted.
    if len(incoming) == 0:
        return dimension.head(0)[output_cols].reset_index(drop=True)

    # Empty dimension -> everything is an insert.
    if len(dimension) == 0:
        return incoming[output_cols].reset_index(drop=True)

    dim_values = dimension[output_cols]

    merged = incoming.merge(
        dim_values, on=keys, how="outer",
        suffixes=("", "_existing"), indicator=True,
    )

    # Inserts: keys present only in incoming.
    inserts = merged.loc[merged["_merge"] == "left_only", output_cols]

    # Both sides: take incoming values (overwrite). Unchanged rows are also
    # in this set; using incoming values for them is a no-op since values match.
    matched = merged.loc[merged["_merge"] == "both", output_cols]

    # Rows in dimension only are dropped (deletes).

    result = pd.concat([matched, inserts], ignore_index=True)
    return result[output_cols]


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    dimension = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name":        ["Alice", "Bob", "Charlie"],
        "tier":        ["gold", "silver", "bronze"],
    })

    incoming = pd.DataFrame({
        "customer_id": [1, 2, 4],
        "name":        ["Alice", "Bobby", "Diana"],
        "tier":        ["gold", "gold", "platinum"],
    })

    result = scd_type1(dimension, incoming, keys=["customer_id"])

    print("=== Result ===")
    print(result.sort_values(["customer_id"]).to_string(index=False))
