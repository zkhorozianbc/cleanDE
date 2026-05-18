"""
SCD Type 3: apply incoming records to a slowly-changing dimension table,
producing an updated table with limited history via parallel ``previous_<col>``
columns and a single ``effective_date`` column.

Type 3 keeps exactly one row per natural key. For each tracked value column,
the prior value (as of the previous change) is held in ``previous_<col>``.
The ``effective_date`` column records when the row was last modified.

Usage:
    dimension = pq.read_table("dim_customer.parquet")
    incoming  = pq.read_table("staging_customer.parquet")
    result    = scd_type3(dimension, incoming, keys=["customer_id"], effective_time=datetime.now())
"""

from datetime import datetime

import pyarrow as pa
import pyarrow.compute as pc


def scd_type3(
    dimension: pa.Table,
    incoming: pa.Table,
    keys: list[str],
    effective_time: datetime,
    tracked_columns: list[str] | None = None,
    previous_prefix: str = "previous_",
    effective_date: str = "effective_date",
) -> pa.Table:
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
    dimension : pa.Table
        Current dimension table including ``previous_<col>`` and
        ``effective_date`` columns.
    incoming : pa.Table
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
    pa.Table
        Updated dimension table with columns
        ``keys + value_cols + [previous_<col> for col in tracked_columns] + [effective_date]``.
    """
    value_cols = [c for c in incoming.column_names if c not in keys]
    if tracked_columns is None:
        tracked = list(value_cols)
    else:
        tracked = list(tracked_columns)
    previous_cols = [previous_prefix + c for c in tracked]
    output_cols = keys + value_cols + previous_cols + [effective_date]

    dim_values = dimension.select(keys + value_cols)

    # --- Classify incoming rows via joins ---

    insert_rows = incoming.join(dim_values, keys=keys, join_type="left anti")
    delete_rows = dim_values.join(incoming, keys=keys, join_type="left anti")
    matched = incoming.join(
        dim_values, keys=keys, join_type="inner", right_suffix="_existing",
    )

    if value_cols and matched.num_rows > 0:
        diffs: list[pa.ChunkedArray] = []
        for col in value_cols:
            left_col = matched.column(col)
            right_col = matched.column(f"{col}_existing")
            one_null = pc.not_equal(pc.is_null(left_col), pc.is_null(right_col))  # ty: ignore[unresolved-attribute]
            ne = pc.fill_null(pc.not_equal(left_col, right_col), False)  # ty: ignore[unresolved-attribute]
            diffs.append(pc.or_(ne, one_null))  # ty: ignore[unresolved-attribute]

        changed_mask = diffs[0]
        for d in diffs[1:]:
            changed_mask = pc.or_(changed_mask, d)  # ty: ignore[unresolved-attribute]

        update_keys = matched.filter(changed_mask).select(keys)
        unchanged_keys = matched.filter(pc.invert(changed_mask)).select(keys)  # ty: ignore[unresolved-attribute]
    else:
        update_keys = matched.select(keys).slice(0, 0)
        unchanged_keys = matched.select(keys)

    # --- Build output ---

    # 1. Unchanged rows: pass through dimension row exactly as-is.
    if unchanged_keys.num_rows > 0:
        unchanged = dimension.join(unchanged_keys, keys=keys, join_type="left semi")
    else:
        unchanged = dimension.slice(0, 0)

    # 2. Updated rows.
    if update_keys.num_rows > 0:
        updates = _build_updates(
            dimension, incoming, update_keys, keys, value_cols,
            tracked, previous_prefix, effective_date, effective_time,
        )
    else:
        updates = _empty_output(dimension, value_cols, tracked, previous_prefix, effective_date)

    # 3. Inserted rows.
    if insert_rows.num_rows > 0:
        new_inserts = _build_inserts(
            insert_rows, dimension, value_cols, tracked,
            previous_prefix, effective_date, effective_time,
        )
    else:
        new_inserts = _empty_output(dimension, value_cols, tracked, previous_prefix, effective_date)

    # 4. Deletes simply don't appear in the output.
    _ = delete_rows

    # Re-order each piece to canonical output order for safe concat.
    unchanged = unchanged.select(output_cols) if unchanged.num_rows > 0 else _empty_output(
        dimension, value_cols, tracked, previous_prefix, effective_date,
    )
    updates = updates.select(output_cols)
    new_inserts = new_inserts.select(output_cols)

    result = pa.concat_tables(
        [unchanged, updates, new_inserts],
        promote_options="permissive",
    )
    return result.select(output_cols)


def _build_updates(
    dimension: pa.Table,
    incoming: pa.Table,
    update_keys: pa.Table,
    keys: list[str],
    value_cols: list[str],
    tracked: list[str],
    previous_prefix: str,
    effective_date: str,
    effective_time: datetime,
) -> pa.Table:
    """Build update rows: shift tracked old values to ``previous_<col>``, overwrite values.

    Parameters
    ----------
    dimension : pa.Table
        Current dimension table (source of old values).
    incoming : pa.Table
        Incoming records carrying new values.
    update_keys : pa.Table
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
    pa.Table
        Rows for updated keys with key, value, previous-value, and effective
        date columns.
    """
    old = dimension.select(keys + tracked).join(update_keys, keys=keys, join_type="left semi")
    new = incoming.join(update_keys, keys=keys, join_type="left semi")

    joined = new.join(
        old, keys=keys, join_type="inner", right_suffix="__old",
    )

    out = joined.select(keys + value_cols)
    for col in tracked:
        prev_name = previous_prefix + col
        # If the tracked col only exists on the right, no suffix is applied.
        src_name = f"{col}__old" if f"{col}__old" in joined.column_names else col
        out = out.append_column(prev_name, joined.column(src_name))

    n = out.num_rows
    ts_type = pa.timestamp("us")
    out = out.append_column(effective_date, pa.array([effective_time] * n, type=ts_type))
    return out


def _build_inserts(
    insert_rows: pa.Table,
    dimension: pa.Table,
    value_cols: list[str],
    tracked: list[str],
    previous_prefix: str,
    effective_date: str,
    effective_time: datetime,
) -> pa.Table:
    """Build insert rows: ``previous_<col>`` = NULL (typed), ``effective_date`` = ``effective_time``.

    Parameters
    ----------
    insert_rows : pa.Table
        Incoming rows whose keys are not present in the dimension.
    dimension : pa.Table
        Existing dimension (used to look up the type for each
        ``previous_<col>``).
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
    pa.Table
        Rows for inserted keys with NULL previous-value columns.
    """
    n = insert_rows.num_rows
    out = insert_rows
    for col in tracked:
        # Match the type of the dimension's existing column for stable schema.
        if col in dimension.column_names:
            col_type = dimension.schema.field(col).type
        else:
            col_type = insert_rows.schema.field(col).type
        out = out.append_column(
            previous_prefix + col,
            pa.nulls(n, type=col_type),
        )

    ts_type = pa.timestamp("us")
    out = out.append_column(effective_date, pa.array([effective_time] * n, type=ts_type))
    return out


def _empty_output(
    dimension: pa.Table,
    value_cols: list[str],
    tracked: list[str],
    previous_prefix: str,
    effective_date: str,
) -> pa.Table:
    """Return an empty table with the canonical Type 3 output schema.

    Parameters
    ----------
    dimension : pa.Table
        Source dimension whose column types are used as the schema basis.
    value_cols : list[str]
        All value columns.
    tracked : list[str]
        Subset of ``value_cols`` for which ``previous_<col>`` is maintained.
    previous_prefix : str
        Prefix for previous-value columns.
    effective_date : str
        Name of the effective-date column.

    Returns
    -------
    pa.Table
        Empty table with the canonical Type 3 output columns and types.
    """
    ts_type = pa.timestamp("us")
    fields: list[pa.Field] = []
    for name in dimension.column_names:
        fields.append(dimension.schema.field(name))
    # Make sure previous_<col> for tracked cols and effective_date exist.
    existing_names = set(dimension.column_names)
    for col in tracked:
        prev_name = previous_prefix + col
        if prev_name not in existing_names:
            col_type = dimension.schema.field(col).type if col in existing_names else pa.null()
            fields.append(pa.field(prev_name, col_type))
            existing_names.add(prev_name)
    if effective_date not in existing_names:
        fields.append(pa.field(effective_date, ts_type))
    schema = pa.schema(fields)
    empty = pa.table({f.name: pa.array([], type=f.type) for f in schema})
    output_cols = list(dimension.column_names)
    # Ensure we cover output_cols even if dimension is missing some.
    keys_and_values = [c for c in dimension.column_names if c in (value_cols)] + [
        c for c in value_cols if c not in dimension.column_names
    ]
    _ = keys_and_values
    # Select in canonical order; missing columns won't occur because we added them above.
    # Build canonical order from dimension's key/value structure:
    keys = [c for c in dimension.column_names if c not in value_cols
            and not c.startswith(previous_prefix) and c != effective_date]
    canonical = keys + value_cols + [previous_prefix + c for c in tracked] + [effective_date]
    # Some columns may not exist (e.g., a tracked col not in dimension); fall back to whatever we have.
    canonical = [c for c in canonical if c in empty.column_names]
    return empty.select(canonical)


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 6, 1)
    ts_type = pa.timestamp("us")

    dimension = pa.table({
        "customer_id":    pa.array([1, 2, 3]),
        "name":           pa.array(["Alice", "Bob", "Charlie"]),
        "tier":           pa.array(["gold", "silver", "bronze"]),
        "previous_name":  pa.array([None, None, None], type=pa.string()),
        "previous_tier":  pa.array([None, None, None], type=pa.string()),
        "effective_date": pa.array([t0, t0, t0], type=ts_type),
    })

    incoming = pa.table({
        "customer_id": pa.array([1, 2, 4]),
        "name":        pa.array(["Alice", "Bobby", "Diana"]),
        "tier":        pa.array(["gold", "gold", "platinum"]),
    })

    result = scd_type3(dimension, incoming, keys=["customer_id"], effective_time=t1)

    print("=== Result ===")
    print(result.sort_by([("customer_id", "ascending")]).to_pandas().to_string(index=False))
