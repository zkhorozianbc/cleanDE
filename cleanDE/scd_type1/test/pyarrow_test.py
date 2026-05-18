import pyarrow as pa

from cleanDE.scd_type1.pyarrow_impl import scd_type1


# ── Basic operations ─────────────────────────────────────────────────


class TestInserts:
    def test_insert_into_empty(self) -> None:
        dimension = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() == "a"

    def test_insert_new_key(self) -> None:
        dimension = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})
        incoming = pa.table({"id": pa.array([1, 2]), "val": pa.array(["a", "b"])})

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 2
        pdf = result.to_pandas()
        new_row = pdf[pdf["id"] == 2].iloc[0]
        assert new_row["val"] == "b"


class TestUpdates:
    def test_update_overwrites_value(self) -> None:
        dimension = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() == "b"

    def test_update_does_not_add_row(self) -> None:
        dimension = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["b"])})

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1


class TestDeletes:
    def test_delete_removes_row(self) -> None:
        dimension = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 0

    def test_delete_preserves_others(self) -> None:
        dimension = pa.table({
            "id": pa.array([1, 2]),
            "val": pa.array(["a", "b"]),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1
        assert result.column("id")[0].as_py() == 1


class TestUnchanged:
    def test_unchanged_stays(self) -> None:
        dimension = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() == "a"


# ── Edge cases ───────────────────────────────────────────────────────


class TestNulls:
    def test_null_value_overwrites(self) -> None:
        dimension = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})
        incoming = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
        })

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() is None

    def test_null_to_value_overwrites(self) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
        })
        incoming = pa.table({"id": pa.array([1]), "val": pa.array(["a"])})

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1
        assert result.column("val")[0].as_py() == "a"

    def test_null_to_null_unchanged(self) -> None:
        dimension = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
        })
        incoming = pa.table({
            "id": pa.array([1]),
            "val": pa.array([None], type=pa.string()),
        })

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 1


class TestCompositeKeys:
    def test_multi_column_key(self) -> None:
        dimension = pa.table({
            "k1": pa.array([1]),
            "k2": pa.array(["x"]),
            "val": pa.array(["a"]),
        })
        incoming = pa.table({
            "k1": pa.array([1, 1]),
            "k2": pa.array(["x", "y"]),
            "val": pa.array(["b", "c"]),
        })

        result = scd_type1(dimension, incoming, keys=["k1", "k2"])

        assert result.num_rows == 2
        pdf = result.to_pandas()
        updated = pdf[(pdf["k1"] == 1) & (pdf["k2"] == "x")].iloc[0]
        assert updated["val"] == "b"
        inserted = pdf[(pdf["k1"] == 1) & (pdf["k2"] == "y")].iloc[0]
        assert inserted["val"] == "c"


class TestEmptyInputs:
    def test_both_empty(self) -> None:
        dimension = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 0

    def test_empty_incoming_deletes_all(self) -> None:
        dimension = pa.table({
            "id": pa.array([1, 2]),
            "val": pa.array(["a", "b"]),
        })
        incoming = pa.table({
            "id": pa.array([], type=pa.int64()),
            "val": pa.array([], type=pa.string()),
        })

        result = scd_type1(dimension, incoming, keys=["id"])

        assert result.num_rows == 0


class TestMixedOperations:
    def test_insert_update_delete_unchanged(self) -> None:
        """One of each operation in a single batch."""
        dimension = pa.table({
            "id": pa.array([1, 2, 3]),
            "val": pa.array(["a", "b", "c"]),
        })
        incoming = pa.table({
            "id": pa.array([1, 2, 4]),
            "val": pa.array(["a", "B", "d"]),
        })

        result = scd_type1(dimension, incoming, keys=["id"])
        pdf = result.to_pandas()

        assert set(pdf["id"].tolist()) == {1, 2, 4}
        assert pdf[pdf["id"] == 1].iloc[0]["val"] == "a"
        assert pdf[pdf["id"] == 2].iloc[0]["val"] == "B"
        assert pdf[pdf["id"] == 4].iloc[0]["val"] == "d"
