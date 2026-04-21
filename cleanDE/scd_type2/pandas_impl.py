"""
SCD Type 2: apply incoming records to a slowly-changing dimension table,
producing an updated table with row versioning (valid_from, valid_to,
is_current).

Keys identify a record. Value columns carry the payload. When a value
changes, the old row is closed and a new row is opened.

Usage:
    dimension = pd.read_parquet("dim_customer.parquet")
    incoming  = pd.read_parquet("staging_customer.parquet")
    result    = scd_type2(dimension, incoming, keys=["customer_id"], effective_time=datetime.now())
"""

from datetime import datetime

import pandas as pd


def scd_type2(
    dimension: pd.DataFrame,
    incoming: pd.DataFrame,
    keys: list[str],
    effective_time: datetime,
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
    is_current: str = "is_current",
    high_date: datetime = datetime(9999, 12, 31),
) -> pd.DataFrame:
    """Return an updated SCD Type 2 dimension table after applying incoming records.

    Parameters
    ----------
    dimension : pd.DataFrame
        Current dimension table including versioning columns.
    incoming : pd.DataFrame
        New records with key and value columns only (no versioning columns).
    keys : list[str]
        Column names that uniquely identify a record (natural key).
    effective_time : datetime
        Timestamp marking when changes take effect.
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
    value_cols = [c for c in incoming.columns if c not in keys]
    scd_cols = [valid_from, valid_to, is_current]

    # Current active records
    current_mask = dimension[is_current] == True  # noqa: E712
    current = dimension.loc[current_mask].copy()
    current_values = current[keys + value_cols]

    # --- Classify incoming rows ---

    merged = incoming.merge(
        current_values, on=keys, how="outer",
        suffixes=("", "_existing"), indicator=True,
    )

    inserts = merged.loc[merged["_merge"] == "left_only", keys + value_cols]
    delete_keys = merged.loc[merged["_merge"] == "right_only", keys]
    both = merged.loc[merged["_merge"] == "both"]

    # Detect value changes in matched rows
    if value_cols:
        changed_mask = pd.Series(False, index=both.index)
        for col in value_cols:
            changed_mask |= (
                both[col].ne(both[f"{col}_existing"])
                | (both[col].isna() != both[f"{col}_existing"].isna())
            )
        updates = both.loc[changed_mask, keys + value_cols]
        unchanged_keys = both.loc[~changed_mask, keys]
    else:
        updates = both.head(0)[keys]
        unchanged_keys = both[keys]

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
    return result[keys + value_cols + scd_cols]


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
        Rows to open (key + value columns only).
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

    dimension = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name":        ["Alice", "Bob", "Charlie"],
        "tier":        ["gold", "silver", "bronze"],
        "valid_from":  [t0, t0, t0],
        "valid_to":    [high_date, high_date, high_date],
        "is_current":  [True, True, True],
    })

    incoming = pd.DataFrame({
        "customer_id": [1, 2, 4],
        "name":        ["Alice", "Bobby", "Diana"],
        "tier":        ["gold", "gold", "platinum"],
    })

    result = scd_type2(dimension, incoming, keys=["customer_id"], effective_time=t1)

    print("=== Result ===")
    print(result.sort_values(["customer_id", "valid_from"]).to_string(index=False))
