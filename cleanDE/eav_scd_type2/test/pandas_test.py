from datetime import datetime

import pandas as pd

from cleanDE.eav_scd_type2.pandas_impl import eav_scd_type2


def _current(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows where is_current is True."""
    return df[df["is_current"] == True]  # noqa: E712


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame(columns=["entity_id", "attribute", "value", "valid_from", "valid_to", "is_current"])
        incoming = pd.DataFrame([{"entity_id": 1, "attribute": "name", "value": "Alice"}])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert len(result) == 1
        assert current.iloc[0]["value"] == "Alice"
        assert current.iloc[0]["valid_from"] == t1

    def test_insert_new_entity_attribute(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice"},
            {"entity_id": 1, "attribute": "tier", "value": "gold"},
        ])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert len(current) == 2
        new_row = current[current["attribute"] == "tier"].iloc[0]
        assert new_row["value"] == "gold"
        assert new_row["valid_from"] == t1


class TestUpdates:
    def test_update_closes_old_opens_new(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"entity_id": 1, "attribute": "name", "value": "Alicia"}])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        closed = result[result["is_current"] == False]  # noqa: E712
        current = _current(result)

        assert len(closed) == 1
        assert closed.iloc[0]["value"] == "Alice"
        assert closed.iloc[0]["valid_to"] == t1

        assert len(current) == 1
        assert current.iloc[0]["value"] == "Alicia"
        assert current.iloc[0]["valid_from"] == t1

    def test_update_one_attribute_leaves_others_unchanged(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        """Only the changed attribute gets a new version."""
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"entity_id": 1, "attribute": "tier", "value": "gold",  "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice"},  # unchanged
            {"entity_id": 1, "attribute": "tier", "value": "platinum"},  # update
        ])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        name_row = current[current["attribute"] == "name"].iloc[0]
        assert name_row["valid_from"] == t0  # unchanged, keeps original valid_from

        tier_row = current[current["attribute"] == "tier"].iloc[0]
        assert tier_row["value"] == "platinum"
        assert tier_row["valid_from"] == t1


class TestDeletes:
    def test_delete_closes_attribute(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"entity_id": 1, "attribute": "tier", "value": "gold",  "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice"},
            # tier missing → delete
        ])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert len(current) == 1
        assert current.iloc[0]["attribute"] == "name"

        closed_tier = result[(result["attribute"] == "tier") & (result["is_current"] == False)]  # noqa: E712
        assert len(closed_tier) == 1
        assert closed_tier.iloc[0]["valid_to"] == t1


class TestUnchanged:
    def test_unchanged_stays_current(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"entity_id": 1, "attribute": "name", "value": "Alice"}])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["valid_from"] == t0
        assert result.iloc[0]["is_current"] == True  # noqa: E712


# ── Edge cases ───────────────────────────────────────────────────────


class TestHistory:
    def test_closed_history_passes_through(self, t1: datetime, high_date: datetime) -> None:
        t0 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Al",    "valid_from": t0, "valid_to": t1, "is_current": False},
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t1, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"entity_id": 1, "attribute": "name", "value": "Alice"}])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t2)

        assert len(result) == 2
        hist = result[result["valid_to"] == t1]
        assert len(hist) == 1
        assert hist.iloc[0]["value"] == "Al"


class TestNulls:
    def test_null_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"entity_id": 1, "attribute": "name", "value": None}])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert len(current) == 1
        assert pd.isna(current.iloc[0]["value"])

    def test_null_to_value_detected_as_change(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": None, "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"entity_id": 1, "attribute": "name", "value": "Alice"}])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert len(current) == 1
        assert current.iloc[0]["value"] == "Alice"

    def test_null_to_null_is_unchanged(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": None, "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([{"entity_id": 1, "attribute": "name", "value": None}])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

        assert len(result) == 1
        assert result.iloc[0]["valid_from"] == t0


class TestEmptyInputs:
    def test_both_empty(self, t1: datetime) -> None:
        dimension = pd.DataFrame(columns=["entity_id", "attribute", "value", "valid_from", "valid_to", "is_current"])
        incoming = pd.DataFrame(columns=["entity_id", "attribute", "value"])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)

        assert len(result) == 0

    def test_empty_incoming_closes_all(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"entity_id": 1, "attribute": "tier", "value": "gold",  "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame(columns=["entity_id", "attribute", "value"])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert len(current) == 0
        assert len(result) == 2


class TestMultipleEntities:
    def test_independent_entity_versioning(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        """Each entity's attributes version independently."""
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"entity_id": 2, "attribute": "name", "value": "Bob",   "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alicia"},  # update
            {"entity_id": 2, "attribute": "name", "value": "Bob"},     # unchanged
        ])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        assert len(current) == 2
        e1 = current[current["entity_id"] == 1].iloc[0]
        assert e1["value"] == "Alicia"
        assert e1["valid_from"] == t1

        e2 = current[current["entity_id"] == 2].iloc[0]
        assert e2["value"] == "Bob"
        assert e2["valid_from"] == t0


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        """One of each operation in a single batch."""
        dimension = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice", "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"entity_id": 1, "attribute": "tier", "value": "gold",  "valid_from": t0, "valid_to": high_date, "is_current": True},
            {"entity_id": 1, "attribute": "email", "value": "a@b",  "valid_from": t0, "valid_to": high_date, "is_current": True},
        ])
        incoming = pd.DataFrame([
            {"entity_id": 1, "attribute": "name", "value": "Alice"},     # unchanged
            {"entity_id": 1, "attribute": "tier", "value": "platinum"},  # update
            {"entity_id": 1, "attribute": "phone", "value": "555"},      # insert
            # email missing → delete
        ])

        result = eav_scd_type2(dimension, incoming, entity_key="entity_id", effective_time=t1)
        current = _current(result)

        current_attrs = set(current["attribute"].tolist())
        assert current_attrs == {"name", "tier", "phone"}

        assert current[current["attribute"] == "name"].iloc[0]["valid_from"] == t0
        assert current[current["attribute"] == "tier"].iloc[0]["value"] == "platinum"
        assert current[current["attribute"] == "phone"].iloc[0]["value"] == "555"

        closed_email = result[
            (result["attribute"] == "email") & (result["is_current"] == False)  # noqa: E712
        ]
        assert len(closed_email) == 1


class TestCustomColumnNames:
    def test_custom_eav_and_scd_columns(self, t0: datetime, t1: datetime, high_date: datetime) -> None:
        dimension = pd.DataFrame([
            {"eid": 1, "attr": "name", "val": "Alice", "eff_start": t0, "eff_end": high_date, "active": True},
        ])
        incoming = pd.DataFrame([{"eid": 1, "attr": "name", "val": "Alicia"}])

        result = eav_scd_type2(
            dimension, incoming, entity_key="eid", effective_time=t1,
            attribute_col="attr", value_col="val",
            valid_from="eff_start", valid_to="eff_end", is_current="active",
        )
        current = result[result["active"] == True]  # noqa: E712

        assert len(current) == 1
        assert current.iloc[0]["val"] == "Alicia"
