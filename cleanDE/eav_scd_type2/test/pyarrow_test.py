from datetime import datetime

import pyarrow as pa
import pytest

from cleanDE.eav_scd_type2.pyarrow_impl import eav_scd_type2


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
            "entity_id": pa.array([], type=pa.int64()),
            "attribute": pa.array([], type=pa.string()),
            "value": pa.array([], type=pa.string()),
            "valid_from": pa.array([], type=ts_type),
            "valid_to": pa.array([], type=ts_type),
            "is_current": pa.array([], type=pa.bool_()),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert result.num_rows == 1
        assert current.column("value")[0].as_py() == "Alice"
        assert current.column("valid_from")[0].as_py() == t1

    def test_insert_new_entity_attribute(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1, 1]),
            "attribute": pa.array(["name", "tier"]),
            "value": pa.array(["Alice", "gold"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert current.num_rows == 2


class TestUpdates:
    def test_update_closes_old_opens_new(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alicia"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)
        pdf = result.to_pandas()
        closed = pdf[pdf["is_current"] == False]  # noqa: E712

        assert len(closed) == 1
        assert closed.iloc[0]["value"] == "Alice"
        assert closed.iloc[0]["valid_to"] == t1

        assert current.num_rows == 1
        assert current.column("value")[0].as_py() == "Alicia"

    def test_update_one_attribute_leaves_others_unchanged(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        """Only the changed attribute gets a new version."""
        dimension = pa.table({
            "entity_id": pa.array([1, 1]),
            "attribute": pa.array(["name", "tier"]),
            "value": pa.array(["Alice", "gold"]),
            "valid_from": pa.array([t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date], type=ts_type),
            "is_current": pa.array([True, True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1, 1]),
            "attribute": pa.array(["name", "tier"]),
            "value": pa.array(["Alice", "platinum"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)
        pdf_current = current.to_pandas()

        name_row = pdf_current[pdf_current["attribute"] == "name"].iloc[0]
        assert name_row["valid_from"] == t0  # unchanged

        tier_row = pdf_current[pdf_current["attribute"] == "tier"].iloc[0]
        assert tier_row["value"] == "platinum"
        assert tier_row["valid_from"] == t1


class TestDeletes:
    def test_delete_closes_attribute(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1, 1]),
            "attribute": pa.array(["name", "tier"]),
            "value": pa.array(["Alice", "gold"]),
            "valid_from": pa.array([t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date], type=ts_type),
            "is_current": pa.array([True, True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert current.num_rows == 1
        assert current.column("attribute")[0].as_py() == "name"


class TestUnchanged:
    def test_unchanged_stays_current(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

        assert result.num_rows == 1
        assert result.column("valid_from")[0].as_py() == t0


# ── Edge cases ───────────────────────────────────────────────────────


class TestHistory:
    def test_closed_history_passes_through(self, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        t0 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        dimension = pa.table({
            "entity_id": pa.array([1, 1]),
            "attribute": pa.array(["name", "name"]),
            "value": pa.array(["Al", "Alice"]),
            "valid_from": pa.array([t0, t1], type=ts_type),
            "valid_to": pa.array([t1, high_date], type=ts_type),
            "is_current": pa.array([False, True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t2)
        pdf = result.to_pandas()
        hist = pdf[pdf["valid_to"] == t1]

        assert result.num_rows == 2
        assert len(hist) == 1
        assert hist.iloc[0]["value"] == "Al"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array([None], type=pa.string()),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert current.num_rows == 1
        assert current.column("value")[0].as_py() is None

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array([None], type=pa.string()),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array(["Alice"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert current.num_rows == 1
        assert current.column("value")[0].as_py() == "Alice"

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array([None], type=pa.string()),
            "valid_from": pa.array([t0], type=ts_type),
            "valid_to": pa.array([high_date], type=ts_type),
            "is_current": pa.array([True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1]),
            "attribute": pa.array(["name"]),
            "value": pa.array([None], type=pa.string()),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

        assert result.num_rows == 1
        assert result.column("valid_from")[0].as_py() == t0


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([], type=pa.int64()),
            "attribute": pa.array([], type=pa.string()),
            "value": pa.array([], type=pa.string()),
            "valid_from": pa.array([], type=ts_type),
            "valid_to": pa.array([], type=ts_type),
            "is_current": pa.array([], type=pa.bool_()),
        })
        incoming = pa.table({
            "entity_id": pa.array([], type=pa.int64()),
            "attribute": pa.array([], type=pa.string()),
            "value": pa.array([], type=pa.string()),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

        assert result.num_rows == 0

    def test_empty_incoming_closes_all(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "entity_id": pa.array([1, 1]),
            "attribute": pa.array(["name", "tier"]),
            "value": pa.array(["Alice", "gold"]),
            "valid_from": pa.array([t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date], type=ts_type),
            "is_current": pa.array([True, True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([], type=pa.int64()),
            "attribute": pa.array([], type=pa.string()),
            "value": pa.array([], type=pa.string()),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert current.num_rows == 0
        assert result.num_rows == 2


class TestMultipleEntities:
    def test_independent_entity_versioning(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        """Each entity's attributes version independently."""
        dimension = pa.table({
            "entity_id": pa.array([1, 2]),
            "attribute": pa.array(["name", "name"]),
            "value": pa.array(["Alice", "Bob"]),
            "valid_from": pa.array([t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date], type=ts_type),
            "is_current": pa.array([True, True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1, 2]),
            "attribute": pa.array(["name", "name"]),
            "value": pa.array(["Alicia", "Bob"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)
        pdf_current = current.to_pandas()

        assert current.num_rows == 2
        e1 = pdf_current[pdf_current["entity_id"] == 1].iloc[0]
        assert e1["value"] == "Alicia"
        assert e1["valid_from"] == t1

        e2 = pdf_current[pdf_current["entity_id"] == 2].iloc[0]
        assert e2["value"] == "Bob"
        assert e2["valid_from"] == t0


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        """One of each operation in a single batch."""
        dimension = pa.table({
            "entity_id": pa.array([1, 1, 1]),
            "attribute": pa.array(["name", "tier", "email"]),
            "value": pa.array(["Alice", "gold", "a@b"]),
            "valid_from": pa.array([t0, t0, t0], type=ts_type),
            "valid_to": pa.array([high_date, high_date, high_date], type=ts_type),
            "is_current": pa.array([True, True, True]),
        })
        incoming = pa.table({
            "entity_id": pa.array([1, 1, 1]),
            "attribute": pa.array(["name", "tier", "phone"]),
            "value": pa.array(["Alice", "platinum", "555"]),
        })

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)
        current_attrs = set(current.column("attribute").to_pylist())

        assert current_attrs == {"name", "tier", "phone"}

        pdf = result.to_pandas()
        closed_email = pdf[(pdf["attribute"] == "email") & (pdf["is_current"] == False)]  # noqa: E712
        assert len(closed_email) == 1


class TestCustomColumnNames:
    def test_custom_eav_and_scd_columns(self, t0: datetime, t1: datetime, high_date: datetime, ts_type: pa.DataType) -> None:
        dimension = pa.table({
            "eid": pa.array([1]),
            "attr": pa.array(["name"]),
            "val": pa.array(["Alice"]),
            "eff_start": pa.array([t0], type=ts_type),
            "eff_end": pa.array([high_date], type=ts_type),
            "active": pa.array([True]),
        })
        incoming = pa.table({
            "eid": pa.array([1]),
            "attr": pa.array(["name"]),
            "val": pa.array(["Alicia"]),
        })

        result = eav_scd_type2(
            dimension, incoming, entity_key="eid", effective_time=t1,
            attribute_col="attr", value_col="val",
            valid_from="eff_start", valid_to="eff_end", is_current="active",
        )
        pdf = result.to_pandas()
        current = pdf[pdf["active"] == True]  # noqa: E712

        assert len(current) == 1
        assert current.iloc[0]["val"] == "Alicia"
