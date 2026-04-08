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
    result   = bitemporal_diff(existing, incoming, keys=["customer_id"])
"""

from datetime import datetime

import pyarrow as pa


def bitemporal_diff(
    existing: pa.Table,
    incoming: pa.Table,
    keys: list[str],
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
    txn_from: str = "txn_from",
    txn_to: str = "txn_to",
    txn_time: datetime | None = None,
    high_date: datetime = datetime(9999, 12, 31),
) -> pa.Table:
    """Return a complete bi-temporal table after applying the incoming batch.

    The incoming table should contain the key columns and value columns only
    (no temporal columns). The function figures out what changed.

    Returns all rows (historical + new versions) with correct temporal markers.
    """
    now = txn_time or datetime.now()
    temporal_cols = {valid_from, valid_to, txn_from, txn_to}
    value_cols = [c for c in incoming.column_names if c not in keys]

    # Active records: valid_to == high_date AND txn_to == high_date
    vt_col = existing.column(valid_to)
    tt_col = existing.column(txn_to)
    active_mask = pa.array([
        vt_col[i].as_py() == high_date and tt_col[i].as_py() == high_date
        for i in range(existing.num_rows)
    ])
    active = existing.filter(active_mask)
    active_values = active.select(keys + value_cols)

    # Build key index from active records for lookups
    active_key_index = _build_key_index(active_values, keys)
    incoming_key_index = _build_key_index(incoming, keys)

    # Classify each incoming row
    insert_indices = []
    update_indices = []
    unchanged_active_indices = []
    delete_active_indices = []

    for i in range(incoming.num_rows):
        k = _row_key(incoming, keys, i)
        if k not in active_key_index:
            insert_indices.append(i)
        else:
            active_idx = active_key_index[k]
            if _values_differ(incoming, active_values, value_cols, i, active_idx):
                update_indices.append(i)
            else:
                unchanged_active_indices.append(active_idx)

    for k, active_idx in active_key_index.items():
        if k not in incoming_key_index:
            delete_active_indices.append(active_idx)

    # --- Build output ---

    # 1. Already-closed historical rows pass through
    inactive_mask = pa.array([not m.as_py() for m in active_mask])
    closed_history = existing.filter(inactive_mask)

    # 2. Unchanged active rows stay open
    unchanged = active.take(unchanged_active_indices) if unchanged_active_indices else active.slice(0, 0)

    # 3. Updated rows: close old version, open new version
    if update_indices:
        update_active_indices = [active_key_index[_row_key(incoming, keys, i)] for i in update_indices]
        closed_updates = _close_rows(active.take(update_active_indices), valid_to, txn_to, now)
        new_updates = _open_rows(incoming.take(update_indices), valid_from, valid_to, txn_from, txn_to, now, high_date)
    else:
        closed_updates = active.slice(0, 0)
        new_updates = _empty_with_temporals(incoming, valid_from, valid_to, txn_from, txn_to)

    # 4. Deleted rows: close them
    if delete_active_indices:
        closed_deletes = _close_rows(active.take(delete_active_indices), valid_to, txn_to, now)
    else:
        closed_deletes = active.slice(0, 0)

    # 5. Inserted rows: brand new
    if insert_indices:
        new_inserts = _open_rows(incoming.take(insert_indices), valid_from, valid_to, txn_from, txn_to, now, high_date)
    else:
        new_inserts = _empty_with_temporals(incoming, valid_from, valid_to, txn_from, txn_to)

    return pa.concat_tables(
        [closed_history, unchanged, closed_updates, new_updates, closed_deletes, new_inserts],
        promote_options="permissive",
    )


def _build_key_index(table: pa.Table, keys: list[str]) -> dict[tuple, int]:
    index: dict[tuple, int] = {}
    for i in range(table.num_rows):
        index[_row_key(table, keys, i)] = i
    return index


def _row_key(table: pa.Table, keys: list[str], row: int) -> tuple:
    return tuple(table.column(k)[row].as_py() for k in keys)


def _values_differ(
    left: pa.Table, right: pa.Table, value_cols: list[str], left_row: int, right_row: int
) -> bool:
    for col in value_cols:
        lv = left.column(col)[left_row]
        rv = right.column(col)[right_row]
        if lv != rv:
            return True
    return False


def _close_rows(
    table: pa.Table, valid_to: str, txn_to: str, now: datetime
) -> pa.Table:
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
