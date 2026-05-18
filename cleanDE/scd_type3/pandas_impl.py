"""
SCD Type 3: apply incoming records to a slowly-changing dimension table,
producing an updated table with limited history via parallel ``previous_<col>``
columns and a single ``effective_date`` column.

Type 3 keeps exactly one row per natural key. For each tracked value column,
the prior value (as of the previous change) is held in ``previous_<col>``.
The ``effective_date`` column records when the row was last modified.

Usage:
    dimension = pd.read_parquet("dim_customer.parquet")
    incoming  = pd.read_parquet("staging_customer.parquet")
    result    = scd_type3(dimension, incoming, keys=["customer_id"], effective_time=datetime.now())
"""

from datetime import datetime

import pandas as pd


def scd_type3(
    dimension: pd.DataFrame,
    incoming: pd.DataFrame,
    keys: list[str],
    effective_time: datetime,
    tracked_columns: list[str] | None = None,
    previous_prefix: str = "previous_",
    effective_date: str = "effective_date",
) -> pd.DataFrame:
    """Return an updated SCD Type 3 dimension table after applying incoming records.

    Exactly one row exists per natural key at all times. For each tracked value
    column, ``previous_<col>`` holds the value as of the previous
    ``effective_date``. When a row is updated, the previous-value slots of ALL
    tracked columns are shifted to the current values, even tracked columns
    that did not change in this batch — this keeps ``previous_<col>`` aligned
    with ``effective_date`` ("the value as of the previous change to this row").

    Behavior:
      * Insert (key not in dimension): new row, ``previous_<col>`` = NULL for
        each tracked column, ``effective_date`` = ``effective_time``.
      * Update (key in both, at least one value changed): overwrite values,
        shift tracked columns' old values into their ``previous_<col>`` slots,
        set ``effective_date`` = ``effective_time``.
      * Unchanged (key in both, all values identical): leave the row exactly
        as-is. ``effective_date`` and ``previous_<col>`` are not touched.
      * Delete (key in dimension, not in incoming): row is removed.

    Null change detection: ``NULL → value`` and ``value → NULL`` count as
    changes; ``NULL → NULL`` is unchanged.

    Parameters
    ----------
    dimension : pd.DataFrame
        Current dimension table including ``previous_<col>`` and
        ``effective_date`` columns.
    incoming : pd.DataFrame
        New records with key and value columns only (no history columns).
    keys : list[str]
        Column names that uniquely identify a record (natural key).
    effective_time : datetime
        Timestamp written into ``effective_date`` for inserted and updated
        rows.
    tracked_columns : list[str] | None
        Subset of value columns to retain ``previous_<col>`` history for.
        ``None`` (default) means every value column is tracked. Non-tracked
        value columns are still overwritten by incoming values; they just have
        no ``previous_<col>`` slot.
    previous_prefix : str
        Prefix for the previous-value column names. Defaults to ``"previous_"``.
    effective_date : str
        Name of the last-change-timestamp column. Defaults to
        ``"effective_date"``.

    Returns
    -------
    pd.DataFrame
        Updated dimension table with columns
        ``keys + value_cols + [previous_<col> for col in tracked_columns] + [effective_date]``.
    """
    value_cols = [c for c in incoming.columns if c not in keys]
    if tracked_columns is None:
        tracked = list(value_cols)
    else:
        tracked = list(tracked_columns)
    previous_cols = [previous_prefix + c for c in tracked]
    output_cols = keys + value_cols + previous_cols + [effective_date]

    # --- Classify incoming rows ---

    dim_values = dimension[keys + value_cols] if len(dimension) else dimension.reindex(columns=keys + value_cols)
    merged = incoming.merge(
        dim_values, on=keys, how="outer",
        suffixes=("", "_existing"), indicator=True,
    )

    insert_keys = merged.loc[merged["_merge"] == "left_only", keys]
    delete_keys = merged.loc[merged["_merge"] == "right_only", keys]
    both = merged.loc[merged["_merge"] == "both"]

    # Detect value changes in matched rows
    if value_cols and len(both):
        changed_mask = pd.Series(False, index=both.index)
        for col in value_cols:
            changed_mask |= (
                both[col].ne(both[f"{col}_existing"])
                | (both[col].isna() != both[f"{col}_existing"].isna())
            )
        update_keys = both.loc[changed_mask, keys]
        unchanged_keys = both.loc[~changed_mask, keys]
    else:
        update_keys = both.head(0)[keys]
        unchanged_keys = both[keys]

    # --- Build output ---

    # 1. Unchanged rows: pass through dimension row exactly as-is.
    if len(unchanged_keys):
        unchanged = dimension.merge(unchanged_keys, on=keys, how="inner")
    else:
        unchanged = dimension.iloc[0:0]

    # 2. Updated rows: shift tracked previous_<col> from dimension's old values;
    #    overwrite all value columns with incoming; set effective_date.
    if len(update_keys):
        updates = _build_updates(
            dimension, incoming, update_keys, keys, value_cols,
            tracked, previous_prefix, effective_date, effective_time,
        )
    else:
        updates = dimension.iloc[0:0]

    # 3. Inserted rows: previous_<col> = NULL, effective_date = effective_time.
    if len(insert_keys):
        new_inserts = _build_inserts(
            incoming, insert_keys, keys, value_cols,
            tracked, previous_prefix, effective_date, effective_time,
        )
    else:
        new_inserts = dimension.iloc[0:0]

    # 4. Deletes simply don't appear in the output.
    _ = delete_keys

    result = pd.concat([unchanged, updates, new_inserts], ignore_index=True)

    # Ensure every expected column exists (handles all-empty cases).
    for col in output_cols:
        if col not in result.columns:
            result[col] = pd.Series(dtype="object")
    return result[output_cols]


def _build_updates(
    dimension: pd.DataFrame,
    incoming: pd.DataFrame,
    update_keys: pd.DataFrame,
    keys: list[str],
    value_cols: list[str],
    tracked: list[str],
    previous_prefix: str,
    effective_date: str,
    effective_time: datetime,
) -> pd.DataFrame:
    """Build update rows: shift tracked old values to ``previous_<col>``, overwrite values.

    Parameters
    ----------
    dimension : pd.DataFrame
        Current dimension table (contains old values and existing
        ``previous_<col>`` columns).
    incoming : pd.DataFrame
        Incoming records carrying new values.
    update_keys : pd.DataFrame
        Keys of rows whose values changed.
    keys : list[str]
        Natural key columns.
    value_cols : list[str]
        All value columns (from incoming).
    tracked : list[str]
        Subset of ``value_cols`` for which ``previous_<col>`` is maintained.
    previous_prefix : str
        Prefix for previous-value columns.
    effective_date : str
        Name of the effective-date column.
    effective_time : datetime
        Timestamp written into ``effective_date``.

    Returns
    -------
    pd.DataFrame
        Rows for updated keys, in the canonical output column order.
    """
    old = dimension.merge(update_keys, on=keys, how="inner")
    new = incoming.merge(update_keys, on=keys, how="inner")

    merged = new.merge(
        old[keys + tracked], on=keys, how="left",
        suffixes=("", "__old"),
    )

    out = merged[keys + value_cols].copy()
    for col in tracked:
        out[previous_prefix + col] = merged[f"{col}__old"] if f"{col}__old" in merged.columns else merged[col]
    out[effective_date] = effective_time
    return out


def _build_inserts(
    incoming: pd.DataFrame,
    insert_keys: pd.DataFrame,
    keys: list[str],
    value_cols: list[str],
    tracked: list[str],
    previous_prefix: str,
    effective_date: str,
    effective_time: datetime,
) -> pd.DataFrame:
    """Build insert rows: ``previous_<col>`` = NULL, ``effective_date`` = ``effective_time``.

    Parameters
    ----------
    incoming : pd.DataFrame
        Incoming records.
    insert_keys : pd.DataFrame
        Keys of rows not present in the existing dimension.
    keys : list[str]
        Natural key columns.
    value_cols : list[str]
        All value columns (from incoming).
    tracked : list[str]
        Subset of ``value_cols`` for which ``previous_<col>`` is maintained.
    previous_prefix : str
        Prefix for previous-value columns.
    effective_date : str
        Name of the effective-date column.
    effective_time : datetime
        Timestamp written into ``effective_date``.

    Returns
    -------
    pd.DataFrame
        Rows for inserted keys, in the canonical output column order.
    """
    new = incoming.merge(insert_keys, on=keys, how="inner")
    out = new[keys + value_cols].copy()
    for col in tracked:
        out[previous_prefix + col] = pd.Series([pd.NA] * len(out), index=out.index)
    out[effective_date] = effective_time
    return out


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 6, 1)

    dimension = pd.DataFrame({
        "customer_id":   [1, 2, 3],
        "name":          ["Alice", "Bob", "Charlie"],
        "tier":          ["gold", "silver", "bronze"],
        "previous_name": [None, None, None],
        "previous_tier": [None, None, None],
        "effective_date": [t0, t0, t0],
    })

    incoming = pd.DataFrame({
        "customer_id": [1, 2, 4],
        "name":        ["Alice", "Bobby", "Diana"],
        "tier":        ["gold", "gold", "platinum"],
    })

    result = scd_type3(dimension, incoming, keys=["customer_id"], effective_time=t1)

    print("=== Result ===")
    print(result.sort_values(["customer_id"]).to_string(index=False))
