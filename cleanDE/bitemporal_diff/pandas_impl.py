"""
Bi-temporal diff: compute inserts, updates, deletes between incoming and
existing data, then produce a new versioned table with proper valid_from,
valid_to, and transaction timestamps.

Keys identify a record. Value columns carry the payload. Two time axes:
  - valid_time:  when the fact is true in the real world
  - txn_time:    when the system recorded the fact

Usage:
    existing = pd.read_parquet("current_state.parquet")
    incoming = pd.read_parquet("new_batch.parquet")
    result   = bitemporal_diff(existing, incoming, keys=["customer_id"], txn_time=datetime.now())
"""

from datetime import datetime

import pandas as pd


def bitemporal_diff(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    keys: list[str],
    txn_time: datetime,
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
    txn_from: str = "txn_from",
    txn_to: str = "txn_to",
    high_date: datetime = datetime(9999, 12, 31),
) -> pd.DataFrame:
    """Return a complete bi-temporal table after applying the incoming batch.

    Parameters
    ----------
    existing : pd.DataFrame
        Current state table including temporal columns.
    incoming : pd.DataFrame
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
    pd.DataFrame
        All rows (historical + new versions) with correct temporal markers.
    """
    value_cols = [c for c in incoming.columns if c not in keys]
    temporal_cols = [valid_from, valid_to, txn_from, txn_to]

    # Current active records
    active_mask = (existing[valid_to] == high_date) & (existing[txn_to] == high_date)
    active = existing.loc[active_mask].copy()
    active_values = active[keys + value_cols]

    # --- Classify incoming rows ---

    merged = incoming.merge(active_values, on=keys, how="outer", suffixes=("", "_existing"), indicator=True)

    inserts = merged.loc[merged["_merge"] == "left_only", keys + value_cols]
    delete_keys = merged.loc[merged["_merge"] == "right_only", keys]
    both = merged.loc[merged["_merge"] == "both"]

    # Detect value changes in matched rows
    if value_cols:
        changed_mask = pd.Series(False, index=both.index)
        for col in value_cols:
            changed_mask |= both[col].ne(both[f"{col}_existing"]) | (both[col].isna() != both[f"{col}_existing"].isna())
        updates = both.loc[changed_mask, keys + value_cols]
        unchanged_keys = both.loc[~changed_mask, keys]
    else:
        updates = both.head(0)[keys]
        unchanged_keys = both[keys]

    # --- Build output ---

    # 1. Already-closed historical rows pass through
    closed_history = existing.loc[~active_mask]

    # 2. Unchanged active rows stay open
    unchanged = active.merge(unchanged_keys, on=keys, how="inner")

    # 3. Updated rows: close old, open new
    closed_updates = _close_rows(
        active.merge(updates[keys], on=keys, how="inner"),
        valid_to, txn_to, txn_time,
    )
    new_updates = _open_rows(updates, valid_from, valid_to, txn_from, txn_to, txn_time, high_date)

    # 4. Deleted rows: close them
    closed_deletes = _close_rows(
        active.merge(delete_keys, on=keys, how="inner"),
        valid_to, txn_to, txn_time,
    )

    # 5. Inserted rows: new
    new_inserts = _open_rows(inserts, valid_from, valid_to, txn_from, txn_to, txn_time, high_date)

    result = pd.concat(
        [closed_history, unchanged, closed_updates, new_updates, closed_deletes, new_inserts],
        ignore_index=True,
    )
    return result[keys + value_cols + temporal_cols]


def _close_rows(
    df: pd.DataFrame, valid_to: str, txn_to: str, txn_time: datetime,
) -> pd.DataFrame:
    """Set valid_to and txn_to to txn_time, marking rows as closed.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to close.
    valid_to : str
        Name of the valid-to column.
    txn_to : str
        Name of the transaction-to column.
    txn_time : datetime
        Timestamp to set as the closing time.

    Returns
    -------
    pd.DataFrame
        Copy with valid_to and txn_to set to txn_time.
    """
    df = df.copy()
    df[valid_to] = txn_time
    df[txn_to] = txn_time
    return df


def _open_rows(
    df: pd.DataFrame,
    valid_from: str, valid_to: str, txn_from: str, txn_to: str,
    txn_time: datetime, high_date: datetime,
) -> pd.DataFrame:
    """Add temporal columns marking rows as newly opened.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to open (key + value columns only).
    valid_from : str
        Name of the valid-from column.
    valid_to : str
        Name of the valid-to column.
    txn_from : str
        Name of the transaction-from column.
    txn_to : str
        Name of the transaction-to column.
    txn_time : datetime
        Timestamp for the start of validity and transaction.
    high_date : datetime
        Sentinel value for open-ended timestamps.

    Returns
    -------
    pd.DataFrame
        Copy with temporal columns added.
    """
    df = df.copy()
    df[valid_from] = txn_time
    df[valid_to] = high_date
    df[txn_from] = txn_time
    df[txn_to] = high_date
    return df


# ── Quick smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    high_date = datetime(9999, 12, 31)
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 6, 1)

    existing = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name":        ["Alice", "Bob", "Charlie"],
        "tier":        ["gold", "silver", "bronze"],
        "valid_from":  [t0, t0, t0],
        "valid_to":    [high_date, high_date, high_date],
        "txn_from":    [t0, t0, t0],
        "txn_to":      [high_date, high_date, high_date],
    })

    incoming = pd.DataFrame({
        "customer_id": [1, 2, 4],
        "name":        ["Alice", "Bobby", "Diana"],
        "tier":        ["gold", "gold", "platinum"],
    })

    result = bitemporal_diff(existing, incoming, keys=["customer_id"], txn_time=t1)

    print("=== Result ===")
    print(result.sort_values(["customer_id", "valid_from"]).to_string(index=False))
