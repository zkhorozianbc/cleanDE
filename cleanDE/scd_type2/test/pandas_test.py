from datetime import datetime

import pandas as pd

from cleanDE.scd_type2.pandas_impl import scd_type2


def _current(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows where is_current is True."""
    return df[df["is_current"] == True]  # noqa: E712


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame(columns=["id", "val", "valid_from", "valid_to", "is_current"])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert len(result) == 1
        assert current.iloc[0]["val"] == "a"
        assert current.iloc[0]["valid_from"] == t1
        assert current.iloc[0]["valid_to"] == high_date

    def test_insert_new_key(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}, {"id": 2, "val": "b"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert len(current) == 2
        new_row = current[current["id"] == 2].iloc[0]
        assert new_row["val"] == "b"
        assert new_row["valid_from"] == t1


class TestUpdates:
    def test_update_closes_old_opens_new(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        closed = result[result["is_current"] == False]  # noqa: E712
        current = _current(result)

        assert len(closed) == 1
        assert closed.iloc[0]["val"] == "a"
        assert closed.iloc[0]["valid_to"] == t1

        assert len(current) == 1
        assert current.iloc[0]["val"] == "b"
        assert current.iloc[0]["valid_from"] == t1

    def test_update_preserves_original_valid_from(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        closed = result[result["is_current"] == False]  # noqa: E712

        assert closed.iloc[0]["valid_from"] == t0


class TestDeletes:
    def test_delete_closes_row(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert len(current) == 0
        assert len(result) == 1
        assert result.iloc[0]["valid_to"] == t1
        assert result.iloc[0]["is_current"] == False  # noqa: E712

    def test_delete_preserves_others(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"id": 2, "val": "b", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert len(current) == 1
        assert current.iloc[0]["id"] == 1


class TestUnchanged:
    def test_unchanged_stays_current(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["valid_from"] == t0
        assert result.iloc[0]["valid_to"] == high_date
        assert result.iloc[0]["is_current"] == True  # noqa: E712


# ── Edge cases ───────────────────────────────────────────────────────


class TestHistory:
    def test_closed_history_passes_through(self, t1: datetime, high_date: datetime) -> None:
        t0 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        dimension = pd.DataFrame([
            {"id": 1, "val": "old", "valid_from": t0, "valid_to": t1, "is_current": False},
            {"id": 1, "val": "cur", "valid_from": t1, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "cur"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t2)

        assert len(result) == 2
        hist = result[result["valid_to"] == t1]
        assert len(hist) == 1
        assert hist.iloc[0]["val"] == "old"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert len(current) == 1
        assert pd.isna(current.iloc[0]["val"])

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": None, "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert len(current) == 1
        assert current.iloc[0]["val"] == "a"

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": None, "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["valid_from"] == t0


class TestCompositeKeys:
    def test_multi_column_key(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"k1": 1, "k2": "x", "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"k1": 1, "k2": "x", "val": "b"}, {"k1": 1, "k2": "y", "val": "c"}])

        result = scd_type2(dimension, incoming, keys=["k1", "k2"], effective_time=t1)
        current = _current(result)

        assert len(current) == 2
        updated = current[(current["k1"] == 1) & (current["k2"] == "x")].iloc[0]
        assert updated["val"] == "b"
        inserted = current[(current["k1"] == 1) & (current["k2"] == "y")].iloc[0]
        assert inserted["val"] == "c"


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime) -> None:
        dimension = pd.DataFrame(columns=["id", "val", "valid_from", "valid_to", "is_current"])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 0

    def test_empty_incoming_closes_all(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"id": 2, "val": "b", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert len(current) == 0
        assert len(result) == 2


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        """One of each operation in a single batch."""
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"id": 2, "val": "b", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"id": 3, "val": "c", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([
            {"id": 1, "val": "a"},   # unchanged
            {"id": 2, "val": "B"},   # update
            {"id": 4, "val": "d"},   # insert
            # id=3 missing → delete
        ])

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert set(current["id"].tolist()) == {1, 2, 4}
        assert current[current["id"] == 1].iloc[0]["valid_from"] == t0
        assert current[current["id"] == 2].iloc[0]["val"] == "B"
        assert current[current["id"] == 4].iloc[0]["val"] == "d"

        closed_3 = result[(result["id"] == 3) & (result["is_current"] == False)]  # noqa: E712
        assert len(closed_3) == 1


class TestCustomColumnNames:
    def test_custom_scd_columns(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "eff_start": t0, "eff_end": high_date, "active": True},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type2(
            dimension, incoming, keys=["id"], effective_time=t1,
            valid_from="eff_start", valid_to="eff_end", is_current="active",
        )
        current = result[result["active"] == True]  # noqa: E712

        assert len(current) == 1
        assert current.iloc[0]["val"] == "b"
