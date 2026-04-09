from datetime import datetime

import pandas as pd

from cleanDE.bitemporal_diff.pandas_impl import bitemporal_diff


def _active(df: pd.DataFrame, high_date: datetime) -> pd.DataFrame:
    """Filter to rows where valid_to and txn_to equal high_date."""
    return df[(df["valid_to"] == high_date) & (df["txn_to"] == high_date)]


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame(columns=["id", "val", "valid_from", "valid_to", "txn_from", "txn_to"])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert len(result) == 1
        assert active.iloc[0]["val"] == "a"
        assert active.iloc[0]["valid_from"] == t1

    def test_insert_new_key(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}, {"id": 2, "val": "b"}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert len(active) == 2
        new_row = active[active["id"] == 2].iloc[0]
        assert new_row["val"] == "b"
        assert new_row["valid_from"] == t1


class TestUpdates:
    def test_update_closes_old_opens_new(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        closed = result[(result["valid_to"] != high_date) | (result["txn_to"] != high_date)]
        active = _active(result, high_date)

        assert len(closed) == 1
        assert closed.iloc[0]["val"] == "a"
        assert closed.iloc[0]["valid_to"] == t1

        assert len(active) == 1
        assert active.iloc[0]["val"] == "b"
        assert active.iloc[0]["valid_from"] == t1


class TestDeletes:
    def test_delete_closes_row(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert len(active) == 0
        assert len(result) == 1
        assert result.iloc[0]["valid_to"] == t1

    def test_delete_preserves_others(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
            {"id": 2, "val": "b", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert len(active) == 1
        assert active.iloc[0]["id"] == 1


class TestUnchanged:
    def test_unchanged_stays_open(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["valid_from"] == t0
        assert result.iloc[0]["valid_to"] == high_date


# ── Edge cases ───────────────────────────────────────────────────────


class TestHistory:
    def test_closed_history_passes_through(self, t1: datetime, high_date: datetime) -> None:
        t0 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        existing = pd.DataFrame([
            {"id": 1, "val": "old", "valid_from": t0, "valid_to": t1, "txn_from": t0, "txn_to": t1},
            {"id": 1, "val": "cur", "valid_from": t1, "valid_to": high_date, "txn_from": t1, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "cur"}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t2)

        assert len(result) == 2
        hist = result[result["valid_to"] == t1]
        assert len(hist) == 1
        assert hist.iloc[0]["val"] == "old"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert len(active) == 1
        assert pd.isna(active.iloc[0]["val"])

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": None, "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert len(active) == 1
        assert active.iloc[0]["val"] == "a"

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": None, "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["valid_from"] == t0


class TestCompositeKeys:
    def test_multi_column_key(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"k1": 1, "k2": "x", "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([{"k1": 1, "k2": "x", "val": "b"}, {"k1": 1, "k2": "y", "val": "c"}])

        result = bitemporal_diff(existing, incoming, keys=["k1", "k2"], txn_time=t1)
        active = _active(result, high_date)

        assert len(active) == 2
        updated = active[(active["k1"] == 1) & (active["k2"] == "x")].iloc[0]
        assert updated["val"] == "b"
        inserted = active[(active["k1"] == 1) & (active["k2"] == "y")].iloc[0]
        assert inserted["val"] == "c"


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime) -> None:
        existing = pd.DataFrame(columns=["id", "val", "valid_from", "valid_to", "txn_from", "txn_to"])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)

        assert len(result) == 0

    def test_empty_incoming_closes_all(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
            {"id": 2, "val": "b", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert len(active) == 0
        assert len(result) == 2


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        """One of each operation in a single batch."""
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
            {"id": 2, "val": "b", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
            {"id": 3, "val": "c", "valid_from": t0, "valid_to": high_date, "txn_from": t0, "txn_to": high_date},
        ])
        incoming = pd.DataFrame([
            {"id": 1, "val": "a"},   # unchanged
            {"id": 2, "val": "B"},   # update
            {"id": 4, "val": "d"},   # insert
            # id=3 missing → delete
        ])

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert set(active["id"].tolist()) == {1, 2, 4}
        assert active[active["id"] == 1].iloc[0]["valid_from"] == t0
        assert active[active["id"] == 2].iloc[0]["val"] == "B"
        assert active[active["id"] == 4].iloc[0]["val"] == "d"

        closed_3 = result[(result["id"] == 3) & (result["valid_to"] == t1)]
        assert len(closed_3) == 1


class TestCustomColumnNames:
    def test_custom_temporal_columns(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        existing = pd.DataFrame([
            {"id": 1, "val": "a", "vf": t0, "vt": high_date, "tf": t0, "tt": high_date},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = bitemporal_diff(
            existing, incoming, keys=["id"], txn_time=t1,
            valid_from="vf", valid_to="vt", txn_from="tf", txn_to="tt",
        )
        active = result[(result["vt"] == high_date) & (result["tt"] == high_date)]

        assert len(active) == 1
        assert active.iloc[0]["val"] == "b"
