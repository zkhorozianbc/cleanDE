from datetime import datetime

import pandas as pd
import pyarrow as pa
import pytest

from cleanDE.scd_type3.pyarrow_impl import scd_type3


@pytest.fixture()
def ts_type() -> pa.DataType:
    return pa.timestamp("us")


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
            "previous_val": pa.array([], type=pa.string()),
            "effective_date": pa.array([], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() == "a"
        assert result.column("previous_val")[0].as_py() is None
        assert result.column("effective_date")[0].as_py() == t1

    def test_insert_new_key(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1, 2]), "val": pa.array(["a", "b"])})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)
        pdf = result.to_pandas()

        assert result.num_rows == 2
        new_row = pdf[pdf["id"] == 2].iloc[0]
        assert new_row["val"] == "b"
        assert pd.isna(new_row["previous_val"])
        assert new_row["effective_date"] == t1


class TestUpdates:
    def test_update_shifts_to_previous(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() == "b"
        assert result.column("previous_val")[0].as_py() == "a"
        assert result.column("effective_date")[0].as_py() == t1

    def test_update_row_count_unchanged(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1

    def test_two_updates_only_last_previous_retained(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        t2 = datetime(2024, 12, 1)
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })

        first = scd_type3(
            dimension, pa.table({"id": pa.array([1]), "val": pa.array(["b"])}),
            keys=["id"], effective_time=t1,
        )
        second = scd_type3(
            first, pa.table({"id": pa.array([1]), "val": pa.array(["c"])}),
            keys=["id"], effective_time=t2,
        )

        assert second.num_rows == 1
        assert second.column("val")[0].as_py() == "c"
        assert second.column("previous_val")[0].as_py() == "b"
        assert second.column("effective_date")[0].as_py() == t2


class TestDeletes:
    def test_delete_removes_row(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 0


class TestUnchanged:
    def test_unchanged_does_not_update_effective_date(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("effective_date")[0].as_py() == t0

    def test_unchanged_does_not_shift_previous(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array(["old"], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("previous_val")[0].as_py() == "old"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array([None], type=pa.string())})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() is None
        assert result.column("previous_val")[0].as_py() == "a"
        assert result.column("effective_date")[0].as_py() == t1

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() == "a"
        assert result.column("previous_val")[0].as_py() is None
        assert result.column("effective_date")[0].as_py() == t1

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array([None], type=pa.string())})

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("effective_date")[0].as_py() == t0


class TestCompositeKeys:
    def test_multi_column_key(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "k1": pa.array([1]),
            "k2": pa.array(["x"]),
            "val": pa.array(["a"]),
            "previous_val": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({
            "k1": pa.array([1, 1]),
            "k2": pa.array(["x", "y"]),
            "val": pa.array(["b", "c"]),
        })

        result = scd_type3(dimension, incoming, keys=["k1", "k2"], effective_time=t1)
        pdf = result.to_pandas()

        assert result.num_rows == 2
        updated = pdf[(pdf["k1"] == 1) & (pdf["k2"] == "x")].iloc[0]
        assert updated["val"] == "b"
        assert updated["previous_val"] == "a"
        assert updated["effective_date"] == t1

        inserted = pdf[(pdf["k1"] == 1) & (pdf["k2"] == "y")].iloc[0]
        assert inserted["val"] == "c"
        assert pd.isna(inserted["previous_val"])
        assert inserted["effective_date"] == t1


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
            "previous_val": pa.array([], type=pa.string()),
            "effective_date": pa.array([], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 0

    def test_empty_incoming_deletes_all(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1, 2]),
            "val": pa.array(["a", "b"]),
            "previous_val": pa.array([None, None], type=pa.string()),
            "effective_date": pa.array([t0, t0], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 0


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        """One of each operation in a single batch."""
        dimension = pa.table({
            "id": pa.array([1, 2, 3]),
            "val": pa.array(["a", "b", "c"]),
            "previous_val": pa.array([None, None, None], type=pa.string()),
            "effective_date": pa.array([t0, t0, t0], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([1, 2, 4]),
            "val": pa.array(["a", "B", "d"]),
        })

        result = scd_type3(dimension, incoming, keys=["id"], effective_time=t1)
        pdf = result.to_pandas()

        assert set(pdf["id"].tolist()) == {1, 2, 4}

        unchanged = pdf[pdf["id"] == 1].iloc[0]
        assert unchanged["val"] == "a"
        assert pd.isna(unchanged["previous_val"])
        assert unchanged["effective_date"] == t0

        updated = pdf[pdf["id"] == 2].iloc[0]
        assert updated["val"] == "B"
        assert updated["previous_val"] == "b"
        assert updated["effective_date"] == t1

        inserted = pdf[pdf["id"] == 4].iloc[0]
        assert inserted["val"] == "d"
        assert pd.isna(inserted["previous_val"])
        assert inserted["effective_date"] == t1


class TestTrackedColumnsSubset:
    def test_only_some_columns_tracked(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val_a": pa.array(["x"]),
            "val_b": pa.array(["y"]),
            "previous_val_a": pa.array([None], type=pa.string()),
            "effective_date": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([1]),
            "val_a": pa.array(["X"]),
            "val_b": pa.array(["Y"]),
        })

        result = scd_type3(
            dimension, incoming, keys=["id"], effective_time=t1,
            tracked_columns=["val_a"],
        )

        assert "previous_val_a" in result.column_names
        assert "previous_val_b" not in result.column_names
        assert result.column("val_a")[0].as_py() == "X"
        assert result.column("val_b")[0].as_py() == "Y"
        assert result.column("previous_val_a")[0].as_py() == "x"
        assert result.column("effective_date")[0].as_py() == t1


class TestCustomColumnNames:
    def test_custom_previous_prefix_and_effective_date(self, t0: datetime, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "prev_val": pa.array([None], type=pa.string()),
            "last_modified": pa.array([t0], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type3(
            dimension, incoming, keys=["id"], effective_time=t1,
            previous_prefix="prev_", effective_date="last_modified",
        )

        assert "prev_val" in result.column_names
        assert "last_modified" in result.column_names
        assert result.column("val")[0].as_py() == "b"
        assert result.column("prev_val")[0].as_py() == "a"
        assert result.column("last_modified")[0].as_py() == t1
