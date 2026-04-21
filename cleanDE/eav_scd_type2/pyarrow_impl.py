"""
EAV SCD Type 2: apply incoming entity-attribute-value records to a
slowly-changing EAV dimension table, producing an updated table with
row versioning (valid_from, valid_to, is_current).

Each (entity, attribute) pair is versioned independently. When an
attribute's value changes, the old row is closed and a new row is
opened. Attributes missing from the incoming batch are treated as
deletes.

Usage:
    dimension = pq.read_table("dim_eav.parquet")
    incoming  = pq.read_table("staging_eav.parquet")
    result    = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=datetime.now())
"""

from datetime import datetime

import pyarrow as pa
import pyarrow.compute as pc


def eav_scd_type2(
    dimension: pa.Table,
    incoming: pa.Table,
    entity_key: str,
    effective_time: datetime,
    attribute_col: str = "attribute",
    value_col: str = "value",
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
    is_current: str = "is_current",
    high_date: datetime = datetime(9999, 12, 31),
) -> pa.Table:
    """Return an updated EAV SCD Type 2 dimension table after applying incoming records.

    Parameters
    ----------
    dimension : pa.Table
        Current EAV dimension table including versioning columns.
    incoming : pa.Table
        New EAV records with entity_key, attribute, and value columns only.
    entity_key : str
        Name of the entity key column.
    effective_time : datetime
        Timestamp marking when changes take effect.
    attribute_col : str
        Name of the attribute column. Defaults to "attribute".
    value_col : str
        Name of the value column. Defaults to "value".
    valid_from : str
        Name of the valid-from column. Defaults to "valid_from".
    valid_to : str
        Name of the valid-to column. Defaults to "valid_to".
    is_current : str
        Name of the is-current flag column. Defaults to "is_current".
    high_date : datetime
        Sentinel value representing an open-ended timestamp.
        Defaults to 9999-12-31.

    Returns
    -------
    pa.Table
        All rows (historical + new versions) with correct versioning markers.
    """
    keys = [entity_key, attribute_col]
    eav_cols = [entity_key, attribute_col, value_col]
    output_cols = eav_cols + [valid_from, valid_to, is_current]

    # Current active records
    current_mask = pc.equal(dimension.column(is_current), True)  # ty: ignore[unresolved-attribute]
    current = dimension.filter(current_mask)
    current_values = current.select(eav_cols)

    # --- Classify incoming rows via joins ---

    inserts = incoming.join(current_values, keys=keys, join_type="left anti")
    delete_rows = current.join(incoming, keys=keys, join_type="left anti")
    matched = incoming.join(
        current_values, keys=keys, join_type="inner", right_suffix="_existing",
    )

    # Detect value changes in matched rows
    if matched.num_rows > 0:
        left_col = matched.column(value_col)
        right_col = matched.column(f"{value_col}_existing")
        one_null = pc.not_equal(pc.is_null(left_col), pc.is_null(right_col))  # ty: ignore[unresolved-attribute]
        ne = pc.fill_null(pc.not_equal(left_col, right_col), False)  # ty: ignore[unresolved-attribute]
        changed_mask = pc.or_(ne, one_null)  # ty: ignore[unresolved-attribute]

        updates = matched.filter(changed_mask).select(eav_cols)
        unchanged_keys = matched.filter(pc.invert(changed_mask)).select(keys)  # ty: ignore[unresolved-attribute]
    else:
        updates = incoming.slice(0, 0)
        unchanged_keys = matched.select(keys)

    # --- Build output ---

    # 1. Already-closed historical rows pass through
    history = dimension.filter(pc.invert(current_mask))  # ty: ignore[unresolved-attribute]

    # 2. Unchanged current rows stay open
    if unchanged_keys.num_rows > 0:
        unchanged = current.join(unchanged_keys, keys=keys, join_type="left semi")
    else:
        unchanged = current.slice(0, 0)

    # 3. Updated rows: close old version, open new version
    if updates.num_rows > 0:
        update_keys = updates.select(keys)
        closed_updates = _close_rows(
            current.join(update_keys, keys=keys, join_type="left semi"),
            valid_to, is_current, effective_time,
        )
        new_updates = _open_rows(
            updates, valid_from, valid_to, is_current, effective_time, high_date,
        )
    else:
        closed_updates = current.slice(0, 0)
        new_updates = _empty_with_scd(incoming, valid_from, valid_to, is_current)

    # 4. Deleted rows: close them
    if delete_rows.num_rows > 0:
        closed_deletes = _close_rows(delete_rows, valid_to, is_current, effective_time)
    else:
        closed_deletes = current.slice(0, 0)

    # 5. Inserted rows: brand new
    if inserts.num_rows > 0:
        new_inserts = _open_rows(
            inserts, valid_from, valid_to, is_current, effective_time, high_date,
        )
    else:
        new_inserts = _empty_with_scd(incoming, valid_from, valid_to, is_current)

    result = pa.concat_tables(
        [history, unchanged, closed_updates, new_updates, closed_deletes, new_inserts],
        promote_options="permissive",
    )
    return result.select(output_cols)


def _close_rows(
    table: pa.Table, valid_to: str, is_current: str, effective_time: datetime,
) -> pa.Table:
    """Set valid_to to effective_time and is_current to False, closing the rows.

    Parameters
    ----------
    table : pa.Table
        Rows to close.
    valid_to : str
        Name of the valid-to column.
    is_current : str
        Name of the is-current flag column.
    effective_time : datetime
        Timestamp to set as the closing time.

    Returns
    -------
    pa.Table
        Table with valid_to set and is_current set to False.
    """
    n = table.num_rows
    table = table.set_column(
        table.schema.get_field_index(valid_to), valid_to,
        pa.array([effective_time] * n, type=pa.timestamp("us")),
    )
    table = table.set_column(
        table.schema.get_field_index(is_current), is_current,
        pa.array([False] * n, type=pa.bool_()),
    )
    return table


def _open_rows(
    table: pa.Table,
    valid_from: str, valid_to: str, is_current: str,
    effective_time: datetime, high_date: datetime,
) -> pa.Table:
    """Append versioning columns marking rows as newly opened.

    Parameters
    ----------
    table : pa.Table
        Rows to open (EAV columns only).
    valid_from : str
        Name of the valid-from column.
    valid_to : str
        Name of the valid-to column.
    is_current : str
        Name of the is-current flag column.
    effective_time : datetime
        Timestamp for the start of validity.
    high_date : datetime
        Sentinel value for open-ended timestamps.

    Returns
    -------
    pa.Table
        Table with versioning columns appended.
    """
    n = table.num_rows
    ts_type = pa.timestamp("us")
    table = table.append_column(valid_from, pa.array([effective_time] * n, type=ts_type))
    table = table.append_column(valid_to, pa.array([high_date] * n, type=ts_type))
    table = table.append_column(is_current, pa.array([True] * n, type=pa.bool_()))
    return table


def _empty_with_scd(
    table: pa.Table, valid_from: str, valid_to: str, is_current: str,
) -> pa.Table:
    """Return an empty table with the same schema plus versioning columns.

    Parameters
    ----------
    table : pa.Table
        Source table whose schema (minus versioning columns) is used.
    valid_from : str
        Name of the valid-from column.
    valid_to : str
        Name of the valid-to column.
    is_current : str
        Name of the is-current flag column.

    Returns
    -------
    pa.Table
        Empty table with EAV and versioning columns.
    """
    ts_type = pa.timestamp("us")
    schema = table.schema
    schema = schema.append(pa.field(valid_from, ts_type))
    schema = schema.append(pa.field(valid_to, ts_type))
    schema = schema.append(pa.field(is_current, pa.bool_()))
    return pa.table({f.name: pa.array([], type=f.type) for f in schema})


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    high_date = datetime(9999, 12, 31)
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 6, 1)
    ts_type = pa.timestamp("us")

    dimension = pa.table({
        "entity_id": pa.array([1, 1, 2, 2]),
        "attribute":  pa.array(["name", "tier", "name", "tier"]),
        "value":      pa.array(["Alice", "gold", "Bob", "silver"]),
        "valid_from": pa.array([t0, t0, t0, t0], type=ts_type),
        "valid_to":   pa.array([high_date, high_date, high_date, high_date], type=ts_type),
        "is_current": pa.array([True, True, True, True]),
    })

    incoming = pa.table({
        "entity_id": pa.array([1, 1, 2, 3]),
        "attribute":  pa.array(["name", "tier", "name", "name"]),
        "value":      pa.array(["Alice", "platinum", "Bobby", "Charlie"]),
    })

    result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

    print("=== Result ===")
    print(result.sort_by([("entity_id", "ascending"), ("attribute", "ascending"), ("valid_from", "ascending")]).to_pandas().to_string(index=False))
