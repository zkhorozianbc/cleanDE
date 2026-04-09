from datetime import datetime

import pyarrow as pa
import pytest

from cleanDE.bitemporal_diff.pyarrow_impl import bitemporal_diff


@pytest.fixture()
def ts_type() -> pa.DataType:
    return pa.timestamp("us")


def _active(table: pa.Table, high_date: datetime) -> pa.Table:
    """Filter to rows where valid_to and txn_to equal high_date."""
    pdf = table.to_pandas()
    mask = (pdf["valid_to"] == high_date) & (pdf["txn_to"] == high_date)
    return pa.Table.from_pandas(pdf[mask])


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
            "valid_from": pa.array([], type=ts_type),
            "valid_to": pa.array([], type=ts_type),
            "txn_from": pa.array([], type=ts_type),
            "txn_to": pa.array([], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert result.num_rows == 1
        assert active.column("val")[0].as_py() == "a"
        assert active.column("valid_from")[0].as_py() == t1

    def test_insert_new_key(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1, 2]), "val": pa.array(["a", "b"])})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert active.num_rows == 2


class TestUpdates:
    def test_update_closes_old_opens_new(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)
        pdf = result.to_pandas()
        closed = pdf[(pdf["valid_to"] != high_date) | (pdf["txn_to"] != high_date)]

        assert len(closed) == 1
        assert closed.iloc[0]["val"] == "a"
        assert closed.iloc[0]["valid_to"] == t1

        assert active.num_rows == 1
        assert active.column("val")[0].as_py() == "b"


class TestDeletes:
    def test_delete_closes_row(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert active.num_rows == 0
        assert result.num_rows == 1
        assert result.column("valid_to")[0].as_py() == t1


class TestUnchanged:
    def test_unchanged_stays_open(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)

        assert result.num_rows == 1
        assert result.column("valid_from")[0].as_py() == t0
        assert result.column("valid_to")[0].as_py() == high_date


# ── Edge cases ───────────────────────────────────────────────────────


class TestHistory:
    def test_closed_history_passes_through(self, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        t0 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        existing = pa.table({
            "id": pa.array([1, 1]),
            "val": pa.array(["old", "cur"]),
            "valid_from": pa.array([t0, t1], type=ts_type),
            "valid_to": pa.array([t1, high_date], type=ts_type),
            "txn_from": pa.array([t0, t1], type=ts_type),
            "txn_to": pa.array([t1, high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["cur"])})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t2)
        pdf = result.to_pandas()
        hist = pdf[pdf["valid_to"] == t1]

        assert result.num_rows == 2
        assert len(hist) == 1
        assert hist.iloc[0]["val"] == "old"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array([None], type=pa.string())})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert active.num_rows == 1
        assert active.column("val")[0].as_py() is None

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert active.num_rows == 1
        assert active.column("val")[0].as_py() == "a"

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array([None], type=pa.string())})

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)

        assert result.num_rows == 1
        assert result.column("valid_from")[0].as_py() == t0


class TestCompositeKeys:
    def test_multi_column_key(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "k1": pa.array([1]),
            "k2": pa.array(["x"]),
            "val": pa.array(["a"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "txn_from": pa.array([t0], type=ts_type),
            "txn_to": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({
            "k1": pa.array([1, 1]),
            "k2": pa.array(["x", "y"]),
            "val": pa.array(["b", "c"]),
        })

        result = bitemporal_diff(existing, incoming, keys=["k1", "k2"], txn_time=t1)
        active = _active(result, high_date)

        assert active.num_rows == 2


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
            "valid_from": pa.array([], type=ts_type),
            "valid_to": pa.array([], type=ts_type),
            "txn_from": pa.array([], type=ts_type),
            "txn_to": pa.array([], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)

        assert result.num_rows == 0

    def test_empty_incoming_closes_all(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1, 2]),
            "val": pa.array(["a", "b"]),
            "valid_from": pa.array([t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date], type=ts_type),
            "txn_from": pa.array([t0, t0], type=ts_type),
            "txn_to": pa.array([high_date, high_date], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)

        assert active.num_rows == 0
        assert result.num_rows == 2


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        """One of each operation in a single batch."""
        existing = pa.table({
            "id": pa.array([1, 2, 3]),
            "val": pa.array(["a", "b", "c"]),
            "valid_from": pa.array([t0, t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date, high_date], type=ts_type),
            "txn_from": pa.array([t0, t0, t0], type=ts_type),
            "txn_to": pa.array([high_date, high_date, high_date], type=ts_type),
        })
        incoming = pa.table({
            "id": pa.array([1, 2, 4]),
            "val": pa.array(["a", "B", "d"]),
        })

        result = bitemporal_diff(existing, incoming, keys=["id"], txn_time=t1)
        active = _active(result, high_date)
        active_ids = set(active.column("id").to_pylist())

        assert active_ids == {1, 2, 4}

        pdf = result.to_pandas()
        closed_3 = pdf[(pdf["id"] == 3) & (pdf["valid_to"] == t1)]
        assert len(closed_3) == 1


class TestCustomColumnNames:
    def test_custom_temporal_columns(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        existing = pa.table({
            "id": pa.array([1]),
            "val": pa.array(["a"]),
            "vf": pa.array([t0], type=ts_type),
            "vt": pa.array([high_date], type=ts_type),
            "tf": pa.array([t0], type=ts_type),
            "tt": pa.array([high_date], type=ts_type),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = bitemporal_diff(
            existing, incoming, keys=["id"], txn_time=t1,
            valid_from="vf", valid_to="vt", txn_from="tf", txn_to="tt",
        )
        pdf = result.to_pandas()
        active = pdf[(pdf["vt"] == high_date) & (pdf["tt"] == high_date)]

        assert len(active) == 1
        assert active.iloc[0]["val"] == "b"
