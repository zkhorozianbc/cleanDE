from datetime import datetime

import pyarrow as pa
import pytest

from cleanDE.scd_type2.pyarrow_impl import scd_type2


@pytest.fixture()
def ts_type() -> pa.DataType:
    return pa.timestamp("us")


def _current(table: pa.Table) -> pa.Table:
    """Filter to rows where is_current is True."""
    pdf = table.to_pandas()
    mask = pdf["is_current"] == True  # noqa: E712
    return pa.Table.from_pandas(pdf[mask])


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
            "valid_from": pa.array([], type=ts_type),
            "valid_to": pa.array([], type=ts_type),
            "is_current": pa.array([], type=pa.bool_()),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert result.num_rows == 1
        assert current.column("val")[0].as_py() == "a"
        assert current.column("valid_from")[0].as_py() == t1

    def test_insert_new_key(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1, 2]), "val": pa.array(["a", "b"])})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert current.num_rows == 2


class TestUpdates:
    def test_update_closes_old_opens_new(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)
        pdf = result.to_pandas()
        closed = pdf[pdf["is_current"] == False]  # noqa: E712

        assert len(closed) == 1
        assert closed.iloc[0]["val"] == "a"
        assert closed.iloc[0]["valid_to"] == t1

        assert current.num_rows == 1
        assert current.column("val")[0].as_py() == "b"

    def test_update_preserves_original_valid_from(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        pdf = result.to_pandas()
        closed = pdf[pdf["is_current"] == False]  # noqa: E712

        assert closed.iloc[0]["valid_from"] == t0


class TestDeletes:
    def test_delete_closes_row(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert current.num_rows == 0
        assert result.num_rows == 1
        assert result.column("valid_to")[0].as_py() == t1


class TestUnchanged:
    def test_unchanged_stays_current(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("valid_from")[0].as_py() == t0
        assert result.column("valid_to")[0].as_py() == high_date


# ── Edge cases ───────────────────────────────────────────────────────


class TestHistory:
    def test_closed_history_passes_through(self, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        t0 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        dimension = pa.table({
            "id": pa.array([1, 1]),
            "val": pa.array(["old", "cur"]),
            "valid_from": pa.array([t0, t1], type=ts_type),
            "valid_to": pa.array([t1, high_date], type=ts_type),
            "is_current": pa.array([False, True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["cur"])})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t2)
        pdf = result.to_pandas()
        hist = pdf[pdf["valid_to"] == t1]

        assert result.num_rows == 2
        assert len(hist) == 1
        assert hist.iloc[0]["val"] == "old"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array([None], type=pa.string())})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert current.num_rows == 1
        assert current.column("val")[0].as_py() is None

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert current.num_rows == 1
        assert current.column("val")[0].as_py() == "a"

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array([None], type=pa.string())})

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 1
        assert result.column("valid_from")[0].as_py() == t0


class TestCompositeKeys:
    def test_multi_column_key(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "k1": pa.array([1]),
            "k2": pa.array(["x"]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "k1": pa.array([1, 1]),
            "k2": pa.array(["x", "y"]),
            "val": pa.array(["b", "c"]),
        })

        result = scd_type2(dimension, incoming, keys=["k1", "k2"], effective_time=t1)
        current = _current(result)

        assert current.num_rows == 2


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
            "valid_from": pa.array([], type=ts_type),
            "valid_to": pa.array([], type=ts_type),
            "is_current": pa.array([], type=pa.bool_()),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)

        assert result.num_rows == 0

    def test_empty_incoming_closes_all(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1, 2]),
            "val": pa.array(["a", "b"]),
            "valid_from": pa.array([t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date], type=ts_type),
            "is_current": pa.array([True, True]),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)

        assert current.num_rows == 0
        assert result.num_rows == 2


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        """One of each operation in a single batch."""
        dimension = pa.table({
            "id": pa.array([1, 2, 3]),
            "val": pa.array(["a", "b", "c"]),
            "valid_from": pa.array([t0, t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date, high_date], type=ts_type),
            "is_current": pa.array([True, True, True]),
        })
        incoming = pa.table({
            "id": pa.array([1, 2, 4]),
            "val": pa.array(["a", "B", "d"]),
        })

        result = scd_type2(dimension, incoming, keys=["id"], effective_time=t1)
        current = _current(result)
        current_ids = set(current.column("id").to_pylist())

        assert current_ids == {1, 2, 4}

        pdf = result.to_pandas()
        closed_3 = pdf[(pdf["id"] == 3) & (pdf["is_current"] == False)]  # noqa: E712
        assert len(closed_3) == 1


class TestCustomColumnNames:
    def test_custom_scd_columns(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "eff_start": pa.array([t0], type=ts_type),
            "eff_end": pa.array([high_date], type=ts_type),
            "active": pa.array([True]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type2(
            dimension, incoming, keys=["id"], effective_time=t1,
            valid_from="eff_start", valid_to="eff_end", is_current="active",
        )
        pdf = result.to_pandas()
        current = pdf[pdf["active"] == True]  # noqa: E712

        assert len(current) == 1
        assert current.iloc[0]["val"] == "b"
