import pandas as pd

from cleanDE.scd_type1.pandas_impl import scd_type1


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self) -> None:
        dimension = pd.DataFrame(columns=["id", "val"])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1
        assert result.iloc[0]["val"] == "a"

    def test_insert_new_key(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": "a"}])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}, {"id": 2, "val": "b"}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 2
        new_row = result[result["id"] == 2].iloc[0]
        assert new_row["val"] == "b"


class TestUpdates:
    def test_update_overwrites_value(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": "a"}])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1
        assert result.iloc[0]["val"] == "b"

    def test_update_does_not_add_row(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": "a"}])
        incoming = pd.DataFrame([{"id": 1, "val": "b"}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1


class TestDeletes:
    def test_delete_removes_row(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": "a"}])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 0

    def test_delete_preserves_others(self) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a"},
            {"id": 2, "val": "b"},
        ])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1
        assert result.iloc[0]["id"] == 1


class TestUnchanged:
    def test_unchanged_stays(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": "a"}])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1
        assert result.iloc[0]["val"] == "a"


# ── Edge cases ───────────────────────────────────────────────────────


class TestNulls:
    def test_null_value_overwrites(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": "a"}])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["val"])

    def test_null_to_value_overwrites(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": None}])
        incoming = pd.DataFrame([{"id": 1, "val": "a"}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1
        assert result.iloc[0]["val"] == "a"

    def test_null_to_null_unchanged(self) -> None:
        dimension = pd.DataFrame([{"id": 1, "val": None}])
        incoming = pd.DataFrame([{"id": 1, "val": None}])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 1


class TestCompositeKeys:
    def test_multi_column_key(self) -> None:
        dimension = pd.DataFrame([{"k1": 1, "k2": "x", "val": "a"}])
        incoming = pd.DataFrame([
            {"k1": 1, "k2": "x", "val": "b"},
            {"k1": 1, "k2": "y", "val": "c"},
        ])

        result = scd_type1(dimension, incoming, keys=["k1", "k2"])

        assert len(result) == 2
        updated = result[(result["k1"] == 1) & (result["k2"] == "x")].iloc[0]
        assert updated["val"] == "b"
        inserted = result[(result["k1"] == 1) & (result["k2"] == "y")].iloc[0]
        assert inserted["val"] == "c"


class TestEmptyInputs:
    def test_both_empty(self) -> None:
        dimension = pd.DataFrame(columns=["id", "val"])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 0

    def test_empty_incoming_deletes_all(self) -> None:
        dimension = pd.DataFrame([
            {"id": 1, "val": "a"},
            {"id": 2, "val": "b"},
        ])
        incoming = pd.DataFrame(columns=["id", "val"])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert len(result) == 0


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self) -> None:
        """One of each operation in a single batch."""
        dimension = pd.DataFrame([
            {"id": 1, "val": "a"},
            {"id": 2, "val": "b"},
            {"id": 3, "val": "c"},
        ])
        incoming = pd.DataFrame([
            {"id": 1, "val": "a"},   # unchanged
            {"id": 2, "val": "B"},   # update
            {"id": 4, "val": "d"},   # insert
            # id=3 missing → delete
        ])

        result = scd_type1(dimension, incoming, keys=["id"])

        assert set(result["id"].tolist()) == {1, 2, 4}
        assert result[result["id"] == 1].iloc[0]["val"] == "a"
        assert result[result["id"] == 2].iloc[0]["val"] == "B"
        assert result[result["id"] == 4].iloc[0]["val"] == "d"
