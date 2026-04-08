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
    result   = bitemporal_diff(existing, incoming, keys=["customer_id"])
"""

from datetime import datetime

import pandas as pd


def bitemporal_diff(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    keys: list[str],
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
    txn_from: str = "txn_from",
    txn_to: str = "txn_to",
    txn_time: datetime | None = None,
    high_date: datetime = datetime(9999, 12, 31),
) -> pd.DataFrame:
    """Return a complete bi-temporal table after applying the incoming batch.

    The incoming frame should contain the key columns and value columns only
    (no temporal columns). The function figures out what changed.

    Returns all rows (historical + new versions) with correct temporal markers.
    """
    now = txn_time or datetime.now()
    value_cols = [c for c in incoming.columns if c not in keys]
    temporal_cols = [valid_from, valid_to, txn_from, txn_to]

    # Current active records
    active_mask = (existing[valid_to] == high_date) & (existing[txn_to] == high_date)
    active = existing.loc[active_mask].copy()
    active_values = active[keys + value_cols]

    # --- Classify incoming rows ---

    merged = incoming.merge(active_values, on=keys, how="outer", suffixes=("", "_existing"), indicator=True)

    inserts = merged.loc[merged["_merge"] == "left_only", keys + value_cols].copy()
    delete_keys = merged.loc[merged["_merge"] == "right_only", keys].copy()
    both = merged.loc[merged["_merge"] == "both"].copy()

    # Detect value changes in matched rows
    if value_cols:
        changed_mask = pd.Series(False, index=both.index)
        for col in value_cols:
            changed_mask |= both[col].ne(both[f"{col}_existing"]) | (both[col].isna() != both[f"{col}_existing"].isna())
        updates = both.loc[changed_mask, keys + value_cols].copy()
        unchanged_keys = both.loc[~changed_mask, keys].copy()
    else:
        updates = both.head(0)[keys].copy()
        unchanged_keys = both[keys].copy()

    # --- Build output ---

    # 1. Already-closed historical rows pass through
    closed_history = existing.loc[~active_mask].copy()

    # 2. Unchanged active rows stay open
    unchanged = active.merge(unchanged_keys, on=keys, how="inner")

    # 3. Updated rows: close old, open new
    closed_updates = active.merge(updates[keys], on=keys, how="inner").copy()
    closed_updates[valid_to] = now
    closed_updates[txn_to] = now

    new_updates = updates.copy()
    new_updates[valid_from] = now
    new_updates[valid_to] = high_date
    new_updates[txn_from] = now
    new_updates[txn_to] = high_date

    # 4. Deleted rows: close them
    closed_deletes = active.merge(delete_keys, on=keys, how="inner").copy()
    closed_deletes[valid_to] = now
    closed_deletes[txn_to] = now

    # 5. Inserted rows: new
    new_inserts = inserts.copy()
    new_inserts[valid_from] = now
    new_inserts[valid_to] = high_date
    new_inserts[txn_from] = now
    new_inserts[txn_to] = high_date

    result = pd.concat(
        [closed_history, unchanged, closed_updates, new_updates, closed_deletes, new_inserts],
        ignore_index=True,
    )
    return result[keys + value_cols + temporal_cols]


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
