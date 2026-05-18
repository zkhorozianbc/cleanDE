from datetime import datetime

import pandas as pd

from cleanDE.scd_type3.pandas_impl import scd_type3


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self, t1: datetime) -> None:
        dimension = pd.DataFrame(
            columns=["id", "val", "previous_val", "effective_date"],
        )
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["val"] == "a"
        assert pd.isna(row["previous_val"])
        assert row["effective_date"] == t1

    def test_insert_new_key(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}, {"id": 2, "val": "b"}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 2
        new_row = result[result["id"] == 2].iloc[0]
        assert new_row["val"] == "b"
        assert pd.isna(new_row["previous_val"])
        assert new_row["effective_date"] == t1


class TestUpdates:
    def test_update_shifts_to_previous(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["val"] == "b"
        assert row["previous_val"] == "a"
        assert row["effective_date"] == t1

    def test_update_row_count_unchanged(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1

    def test_two_updates_only_last_previous_retained(self, t0: datetime, t1: datetime) -> None:
        t2 = datetime(2024, 12, 1)
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
        ])

        first = scd_type3(
            dimension, pd.DataFrame([{"id": 1, "val": "b"}]),
            keys=["id"], effective_time=t1,
        )
        # After first: val=b, previous_val=a, effective_date=t1
        second = scd_type3(
            first, pd.DataFrame([{"id": 1, "val": "c"}]),
            keys=["id"], effective_time=t2,
        )

        assert len(second) == 1
        row = second.iloc[0]
        assert row["val"] == "c"
        assert row["previous_val"] == "b"  # NOT "a"
        assert row["effective_date"] == t2


class TestDeletes:
    def test_delete_removes_row(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 0


class TestUnchanged:
    def test_unchanged_does_not_update_effective_date(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["effective_date"] == t0  # NOT t1

    def test_unchanged_does_not_shift_previous(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": "old", "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["previous_val"] == "old"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        row = result.iloc[0]
        assert pd.isna(row["val"])
        assert row["previous_val"] == "a"
        assert row["effective_date"] == t1

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": None, "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["val"] == "a"
        assert pd.isna(row["previous_val"])
        assert row["effective_date"] == t1

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": None, "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["effective_date"] == t0


class TestCompositeKeys:
    def test_multi_column_key(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"k1": 1, "k2": "x", "val": "a", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([
            {"k1": 1, "k2": "x", "val": "b"},  # update
            {"k1": 1, "k2": "y", "val": "c"},  # insert
        ])

        result = scd_type3(dimension, incoming, keys=["k1", "k2"], effective_time=t1)

        assert len(result) == 2
        updated = result[(result["k1"] == 1) & (result["k2"] == "x")].iloc[0]
        assert updated["val"] == "b"
        assert updated["previous_val"] == "a"
        assert updated["effective_date"] == t1

        inserted = result[(result["k1"] == 1) & (result["k2"] == "y")].iloc[0]
        assert inserted["val"] == "c"
        assert pd.isna(inserted["previous_val"])
        assert inserted["effective_date"] == t1


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime) -> None:
        dimension = pd.DataFrame(columns=["id", "val", "previous_val", "effective_date"])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 0

    def test_empty_incoming_deletes_all(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
            {"id": 2, "val": "b", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert len(result) == 0


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime) -> None:
        """One of each operation in a single batch."""
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "previous_val": None, "effective_date": t0},
            {"id": 2, "val": "b", "previous_val": None, "effective_date": t0},
            {"id": 3, "val": "c", "previous_val": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([
            {"id": 1, "val": "a"},   # unchanged
            {"id": 2, "val": "B"},   # update
            {"id": 4, "val": "d"},   # insert
            # id=3 missing → delete
        ])

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert set(result["id"].tolist()) == {1, 2, 4}

        unchanged = result[result["id"] == 1].iloc[0]
        assert unchanged["val"] == "a"
        assert pd.isna(unchanged["previous_val"])
        assert unchanged["effective_date"] == t0

        updated = result[result["id"] == 2].iloc[0]
        assert updated["val"] == "B"
        assert updated["previous_val"] == "b"
        assert updated["effective_date"] == t1

        inserted = result[result["id"] == 4].iloc[0]
        assert inserted["val"] == "d"
        assert pd.isna(inserted["previous_val"])
        assert inserted["effective_date"] == t1


class TestTrackedColumnsSubset:
    def test_only_some_columns_tracked(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val_a": "x", "val_b": "y", "previous_val_a": None, "effective_date": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val_a": "X", "val_b": "Y"}])

        result = scd_type3(
            dimension, incoming, keys=["id"], effective_time=t1,
            tracked_columns=["val_a"],
        )

        assert "previous_val_a" in result.columns
        assert "previous_val_b" not in result.columns
        row = result.iloc[0]
        assert row["val_a"] == "X"
        assert row["val_b"] == "Y"
        assert row["previous_val_a"] == "x"
        assert row["effective_date"] == t1


class TestCustomColumnNames:
    def test_custom_previous_prefix_and_effective_date(self, t0: datetime, t1: datetime) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a", "prev_val": None, "last_modified": t0},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type3(
            dimension, incoming, keys=["id"], effective_time=t1,
            previous_prefix="prev_", effective_date="last_modified",
        )

        assert "prev_val" in result.columns
        assert "last_modified" in result.columns
        row = result.iloc[0]
        assert row["val"] == "b"
        assert row["prev_val"] == "a"
        assert row["last_modified"] == t1
