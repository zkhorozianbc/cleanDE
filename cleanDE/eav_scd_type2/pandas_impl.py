"""
EAV SCD Type 2: apply incoming entity-attribute-value records to a
slowly-changing EAV dimension table, producing an updated table with
row versioning (valid_from, valid_to, is_current).

Each (entity, attribute) pair is versioned independently. When an
attribute's value changes, the old row is closed and a new row is
opened. Attributes missing from the incoming batch are treated as
deletes.

Usage:
    dimension = pd.read_parquet("dim_eav.parquet")
    incoming  = pd.read_parquet("staging_eav.parquet")
    result    = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=datetime.now())
"""

from datetime import datetime

import pandas as pd


def eav_scd_type2(
    dimension: pd.DataFrame,
    incoming: pd.DataFrame,
    entity_key: str,
    effective_time: datetime,
    attribute_col: str = "attribute",
    value_col: str = "value",
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
    is_current: str = "is_current",
    high_date: datetime = datetime(9999, 12, 31),
) -> pd.DataFrame:
    """Return an updated EAV SCD Type 2 dimension table after applying incoming records.

    Parameters
    ----------
    dimension : pd.DataFrame
        Current EAV dimension table including versioning columns.
    incoming : pd.DataFrame
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
    pd.DataFrame
        All rows (historical + new versions) with correct versioning markers.
    """
    keys = [entity_key, attribute_col]
    eav_cols = [entity_key, attribute_col, value_col]
    scd_cols = [valid_from, valid_to, is_current]

    # Current active records
    current_mask = dimension[is_current] == True  # noqa: E712
    current = dimension.loc[current_mask].copy()
    current_values = current[eav_cols]

    # --- Classify incoming rows ---

    merged = incoming.merge(
        current_values, on=keys, how="outer",
        suffixes=("", "_existing"), indicator=True,
    )

    inserts = merged.loc[merged["_merge"] == "left_only", eav_cols]
    delete_keys = merged.loc[merged["_merge"] == "right_only", keys]
    both = merged.loc[merged["_merge"] == "both"]

    # Detect value changes in matched rows
    col = value_col
    existing_col = f"{value_col}_existing"
    changed_mask = (
        both[col].ne(both[existing_col])
        | (both[col].isna() != both[existing_col].isna())
    )
    updates = both.loc[changed_mask, eav_cols]
    unchanged_keys = both.loc[~changed_mask, keys]

    # --- Build output ---

    # 1. Already-closed historical rows pass through
    history = dimension.loc[~current_mask]

    # 2. Unchanged current rows stay open
    unchanged = current.merge(unchanged_keys, on=keys, how="inner")

    # 3. Updated rows: close old version, open new version
    closed_updates = _close_rows(
        current.merge(updates[keys], on=keys, how="inner"),
        valid_to, is_current, effective_time,
    )
    new_updates = _open_rows(
        updates, valid_from, valid_to, is_current, effective_time, high_date,
    )

    # 4. Deleted rows: close them
    closed_deletes = _close_rows(
        current.merge(delete_keys, on=keys, how="inner"),
        valid_to, is_current, effective_time,
    )

    # 5. Inserted rows: brand new
    new_inserts = _open_rows(
        inserts, valid_from, valid_to, is_current, effective_time, high_date,
    )

    result = pd.concat(
        [history, unchanged, closed_updates, new_updates, closed_deletes, new_inserts],
        ignore_index=True,
    )
    return result[eav_cols + scd_cols]


def _close_rows(
    df: pd.DataFrame, valid_to: str, is_current: str, effective_time: datetime,
) -> pd.DataFrame:
    """Set valid_to to effective_time and is_current to False, closing the rows.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to close.
    valid_to : str
        Name of the valid-to column.
    is_current : str
        Name of the is-current flag column.
    effective_time : datetime
        Timestamp to set as the closing time.

    Returns
    -------
    pd.DataFrame
        Copy with valid_to set and is_current set to False.
    """
    df = df.copy()
    df[valid_to] = effective_time
    df[is_current] = False
    return df


def _open_rows(
    df: pd.DataFrame,
    valid_from: str, valid_to: str, is_current: str,
    effective_time: datetime, high_date: datetime,
) -> pd.DataFrame:
    """Add versioning columns marking rows as newly opened.

    Parameters
    ----------
    df : pd.DataFrame
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
    pd.DataFrame
        Copy with versioning columns added.
    """
    df = df.copy()
    df[valid_from] = effective_time
    df[valid_to] = high_date
    df[is_current] = True
    return df


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    high_date = datetime(9999, 12, 31)
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 6, 1)

    dimension = pd.DataFrame([
        {"entity_id": 1, "attribute": "name", "value": "Alice",  "valid_from": t0, "valid_to": high_date, "is_current": True},
        {"entity_id": 1, "attribute": "tier", "value": "gold",   "valid_from": t0, "valid_to": high_date, "is_current": True},
        {"entity_id": 2, "attribute": "name", "value": "Bob",    "valid_from": t0, "valid_to": high_date, "is_current": True},
        {"entity_id": 2, "attribute": "tier", "value": "silver", "valid_from": t0, "valid_to": high_date, "is_current": True},
    ])

    incoming = pd.DataFrame([
        {"entity_id": 1, "attribute": "name", "value": "Alice"},   # unchanged
        {"entity_id": 1, "attribute": "tier", "value": "platinum"},  # update
        {"entity_id": 2, "attribute": "name", "value": "Bobby"},   # update
        # entity_id=2, attribute=tier missing → delete
        {"entity_id": 3, "attribute": "name", "value": "Charlie"}, # insert
    ])

    result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

    print("=== Result ===")
    print(result.sort_values(["entity_id", "attribute", "valid_from"]).to_string(index=False))
