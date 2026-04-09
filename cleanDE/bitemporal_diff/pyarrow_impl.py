"""
Bi-temporal diff: compute inserts, updates, deletes between incoming and
existing data, then produce a new versioned table with proper valid_from,
valid_to, and transaction timestamps.

Keys identify a record. Value columns carry the payload. Two time axes:
  - valid_time:  when the fact is true in the real world
  - txn_time:    when the system recorded the fact

Usage:
    existing = pq.read_table("current_state.parquet")
    incoming = pq.read_table("new_batch.parquet")
    result   = bitemporal_diff(existing, incoming, keys=["customer_id"], txn_time=datetime.now())
"""

from datetime import datetime

import pyarrow as pa
import pyarrow.compute as pc


def bitemporal_diff(
    existing: pa.Table,
    incoming: pa.Table,
    keys: list[str],
    txn_time: datetime,
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
    txn_from: str = "txn_from",
    txn_to: str = "txn_to",
    high_date: datetime = datetime(9999, 12, 31),
) -> pa.Table:
    """Return a complete bi-temporal table after applying the incoming batch.

    Parameters
    ----------
    existing : pa.Table
        Current state table including temporal columns.
    incoming : pa.Table
        New batch with key and value columns only (no temporal columns).
    keys : list[str]
        Column names that uniquely identify a record.
    txn_time : datetime
        Transaction timestamp to apply to all changes in this batch.
    valid_from : str
        Name of the valid-from column. Defaults to "valid_from".
    valid_to : str
        Name of the valid-to column. Defaults to "valid_to".
    txn_from : str
        Name of the transaction-from column. Defaults to "txn_from".
    txn_to : str
        Name of the transaction-to column. Defaults to "txn_to".
    high_date : datetime
        Sentinel value representing an open-ended timestamp.
        Defaults to 9999-12-31.

    Returns
    -------
    pa.Table
        All rows (historical + new versions) with correct temporal markers.
    """
    value_cols = [c for c in incoming.column_names if c not in keys]
    output_cols = keys + value_cols + [valid_from, valid_to, txn_from, txn_to]

    # Active records: valid_to == high_date AND txn_to == high_date
    ts_type = existing.schema.field(valid_to).type
    high_scalar = pa.scalar(high_date, type=ts_type)
    active_mask = pc.and_(  # ty: ignore[unresolved-attribute]
        pc.equal(existing.column(valid_to), high_scalar),  # ty: ignore[unresolved-attribute]
        pc.equal(existing.column(txn_to), high_scalar),  # ty: ignore[unresolved-attribute]
    )
    active = existing.filter(active_mask)
    active_values = active.select(keys + value_cols)

    # --- Classify incoming rows via joins ---

    inserts = incoming.join(active_values, keys=keys, join_type="left anti")
    delete_rows = active.join(incoming, keys=keys, join_type="left anti")
    matched = incoming.join(
        active_values, keys=keys, join_type="inner", right_suffix="_existing",
    )

    # Detect value changes in matched rows
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

        updates = matched.filter(changed_mask).select(keys + value_cols)
        unchanged_keys = matched.filter(pc.invert(changed_mask)).select(keys)  # ty: ignore[unresolved-attribute]
    else:
        updates = incoming.slice(0, 0)
        unchanged_keys = matched.select(keys)

    # --- Build output ---

    # 1. Already-closed historical rows pass through
    closed_history = existing.filter(pc.invert(active_mask))  # ty: ignore[unresolved-attribute]

    # 2. Unchanged active rows stay open
    if unchanged_keys.num_rows > 0:
        unchanged = active.join(unchanged_keys, keys=keys, join_type="left semi")
    else:
        unchanged = active.slice(0, 0)

    # 3. Updated rows: close old version, open new version
    if updates.num_rows > 0:
        update_keys = updates.select(keys)
        closed_updates = _close_rows(
            active.join(update_keys, keys=keys, join_type="left semi"),
            valid_to, txn_to, txn_time,
        )
        new_updates = _open_rows(updates, valid_from, valid_to, txn_from, txn_to, txn_time, high_date)
    else:
        closed_updates = active.slice(0, 0)
        new_updates = _empty_with_temporals(incoming, valid_from, valid_to, txn_from, txn_to)

    # 4. Deleted rows: close them
    if delete_rows.num_rows > 0:
        closed_deletes = _close_rows(delete_rows, valid_to, txn_to, txn_time)
    else:
        closed_deletes = active.slice(0, 0)

    # 5. Inserted rows: brand new
    if inserts.num_rows > 0:
        new_inserts = _open_rows(inserts, valid_from, valid_to, txn_from, txn_to, txn_time, high_date)
    else:
        new_inserts = _empty_with_temporals(incoming, valid_from, valid_to, txn_from, txn_to)

    result = pa.concat_tables(
        [closed_history, unchanged, closed_updates, new_updates, closed_deletes, new_inserts],
        promote_options="permissive",
    )
    return result.select(output_cols)


def _close_rows(
    table: pa.Table, valid_to: str, txn_to: str, now: datetime,
) -> pa.Table:
    """Set valid_to and txn_to to now, marking rows as closed.

    Parameters
    ----------
    table : pa.Table
        Rows to close.
    valid_to : str
        Name of the valid-to column.
    txn_to : str
        Name of the transaction-to column.
    now : datetime
        Timestamp to set as the closing time.

    Returns
    -------
    pa.Table
        Table with valid_to and txn_to set to now.
    """
    table = table.set_column(
        table.schema.get_field_index(valid_to), valid_to,
        pa.array([now] * table.num_rows, type=pa.timestamp("us")),
    )
    table = table.set_column(
        table.schema.get_field_index(txn_to), txn_to,
        pa.array([now] * table.num_rows, type=pa.timestamp("us")),
    )
    return table


def _open_rows(
    table: pa.Table,
    valid_from: str, valid_to: str, txn_from: str, txn_to: str,
    now: datetime, high_date: datetime,
) -> pa.Table:
    """Append temporal columns marking rows as newly opened.

    Parameters
    ----------
    table : pa.Table
        Rows to open (key + value columns only).
    valid_from : str
        Name of the valid-from column.
    valid_to : str
        Name of the valid-to column.
    txn_from : str
        Name of the transaction-from column.
    txn_to : str
        Name of the transaction-to column.
    now : datetime
        Timestamp for the start of validity and transaction.
    high_date : datetime
        Sentinel value for open-ended timestamps.

    Returns
    -------
    pa.Table
        Table with temporal columns appended.
    """
    n = table.num_rows
    ts_type = pa.timestamp("us")
    table = table.append_column(valid_from, pa.array([now] * n, type=ts_type))
    table = table.append_column(valid_to, pa.array([high_date] * n, type=ts_type))
    table = table.append_column(txn_from, pa.array([now] * n, type=ts_type))
    table = table.append_column(txn_to, pa.array([high_date] * n, type=ts_type))
    return table


def _empty_with_temporals(
    table: pa.Table, valid_from: str, valid_to: str, txn_from: str, txn_to: str,
) -> pa.Table:
    """Return an empty table with the same schema plus temporal columns.

    Parameters
    ----------
    table : pa.Table
        Source table whose schema (minus temporal columns) is used.
    valid_from : str
        Name of the valid-from column.
    valid_to : str
        Name of the valid-to column.
    txn_from : str
        Name of the transaction-from column.
    txn_to : str
        Name of the transaction-to column.

    Returns
    -------
    pa.Table
        Empty table with key, value, and temporal columns.
    """
    ts_type = pa.timestamp("us")
    schema = table.schema
    for col in [valid_from, valid_to, txn_from, txn_to]:
        schema = schema.append(pa.field(col, ts_type))
    return pa.table({f.name: pa.array([], type=f.type) for f in schema})


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    high_date = datetime(9999, 12, 31)
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 6, 1)
    ts_type = pa.timestamp("us")

    existing = pa.table({
        "customer_id": pa.array([1, 2, 3]),
        "name":        pa.array(["Alice", "Bob", "Charlie"]),
        "tier":        pa.array(["gold", "silver", "bronze"]),
        "valid_from":  pa.array([t0, t0, t0], type=ts_type),
        "valid_to":    pa.array([high_date, high_date, high_date], type=ts_type),
        "txn_from":    pa.array([t0, t0, t0], type=ts_type),
        "txn_to":      pa.array([high_date, high_date, high_date], type=ts_type),
    })

    incoming = pa.table({
        "customer_id": pa.array([1, 2, 4]),
        "name":        pa.array(["Alice", "Bobby", "Diana"]),
        "tier":        pa.array(["gold", "gold", "platinum"]),
    })

    result = bitemporal_diff(existing, incoming, keys=["customer_id"], txn_time=t1)

    print("=== Result ===")
    print(result.sort_by([("customer_id", "ascending"), ("valid_from", "ascending")]).to_pandas().to_string(index=False))
