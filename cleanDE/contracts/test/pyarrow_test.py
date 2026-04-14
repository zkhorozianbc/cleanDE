from pathlib import Path

import pyarrow as pa
import pytest

from cleanDE.contracts.pyarrow_impl import (
    ContractError,
    ContractSpec,
    Verified,
    VersionError,
    _parse_version,
    check_version_constraint,
    columns_present,
    detect_drift,
    no_nulls_in,
    non_empty,
    parse_lock_versions,
    row_count_between,
    schema_field_is,
    unique_on,
    verified,
    version_switch,
)

SAMPLE_LOCK = """\
version = 1
requires-python = ">=3.12"

[[package]]
name = "pandas"
version = "3.0.2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pyarrow"
version = "23.0.1"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "cleande"
version = "0.1.0"
source = { virtual = "." }
"""


@pytest.fixture()
def lock_path(tmp_path: Path) -> Path:
    p = tmp_path / "uv.lock"
    p.write_text(SAMPLE_LOCK)
    return p


@pytest.fixture()
def lock_versions(lock_path: Path) -> dict[str, str]:
    return parse_lock_versions(lock_path)


# ── Lock file parsing ────────────────────────────────────────────────


class TestParseLockVersions:
    def test_extracts_all_packages(self, lock_versions: dict[str, str]) -> None:
        assert lock_versions["pandas"] == "3.0.2"
        assert lock_versions["pyarrow"] == "23.0.1"
        assert lock_versions["cleande"] == "0.1.0"

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_lock_versions(Path("/nonexistent/uv.lock"))

    def test_empty_lock_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.lock"
        p.write_text('version = 1\n')
        assert parse_lock_versions(p) == {}


# ── Version parsing ──────────────────────────────────────────────────


class TestParseVersion:
    def test_standard_version(self) -> None:
        assert _parse_version("23.0.1") == (23, 0, 1)

    def test_prerelease_stripped(self) -> None:
        assert _parse_version("14.0.0beta2") == (14, 0, 0)


# ── Version constraint checking ──────────────────────────────────────


class TestCheckVersionConstraint:
    @pytest.mark.parametrize(
        "version,constraint,expected",
        [
            ("23.0.1", ">=14.0.0", True),
            ("13.0.0", ">=14.0.0", False),
            ("23.0.1", ">=14.0.0,<24.0.0", True),
            ("24.0.0", ">=14.0.0,<24.0.0", False),
            ("23.0.1", "==23.0.1", True),
            ("23.0.2", "==23.0.1", False),
            ("23.0.1", "!=22.0.0", True),
            ("22.0.0", "!=22.0.0", False),
        ],
    )
    def test_constraint(self, version: str, constraint: str, expected: bool) -> None:
        assert check_version_constraint(version, constraint) == expected

    def test_unsupported_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            check_version_constraint("23.0.1", "~=23.0")


# ── Drift detection ──────────────────────────────────────────────────


class TestDetectDrift:
    def test_no_drift_when_versions_match(self) -> None:
        import importlib.metadata

        pkg = "pytest"
        installed = importlib.metadata.version(pkg)
        lock = {pkg: installed}
        assert detect_drift(lock, [pkg]) == {}

    def test_drift_when_versions_differ(self) -> None:
        import importlib.metadata

        pkg = "pytest"
        installed = importlib.metadata.version(pkg)
        lock = {pkg: "0.0.1"}
        drift = detect_drift(lock, [pkg])
        assert pkg in drift
        assert drift[pkg] == ("0.0.1", installed)

    def test_missing_installed_package(self) -> None:
        lock = {"nonexistent_pkg_xyz": "1.0.0"}
        drift = detect_drift(lock, ["nonexistent_pkg_xyz"])
        assert drift["nonexistent_pkg_xyz"] == ("1.0.0", "<not installed>")


# ── verified decorator ───────────────────────────────────────────────


class TestVerified:
    def test_passes_when_requirements_met(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        @verified(require={"pyarrow": ">=14.0.0"}, lock_versions=versions)
        def my_fn(x: int) -> int:
            return x + 1

        assert my_fn(1) == 2

    def test_fails_when_version_too_low(self) -> None:
        versions = {"pyarrow": "10.0.0"}

        with pytest.raises(VersionError, match="locked at 10.0.0"):

            @verified(require={"pyarrow": ">=14.0.0"}, lock_versions=versions)
            def my_fn(x: int) -> int:
                return x + 1

    def test_fails_when_package_missing(self) -> None:
        versions = {"pandas": "3.0.2"}

        with pytest.raises(VersionError, match="not found"):

            @verified(require={"pyarrow": ">=14.0.0"}, lock_versions=versions)
            def my_fn(x: int) -> int:
                return x + 1

    def test_precondition_passes(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        @verified(
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            pre=lambda table: non_empty(table),
        )
        def my_fn(table: pa.Table) -> pa.Table:
            return table

        table = pa.table({"a": [1, 2, 3]})
        result = my_fn(table)
        assert result.num_rows == 3

    def test_precondition_fails(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        @verified(
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            pre=lambda table: non_empty(table),
        )
        def my_fn(table: pa.Table) -> pa.Table:
            return table

        empty = pa.table({"a": pa.array([], type=pa.int64())})
        with pytest.raises(ContractError, match="Precondition"):
            my_fn(empty)

    def test_postcondition_passes(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        @verified(
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            post=lambda result: columns_present(result, ["a", "b"]),
        )
        def my_fn() -> pa.Table:
            return pa.table({"a": [1], "b": [2]})

        result = my_fn()
        assert result.column_names == ["a", "b"]

    def test_postcondition_fails(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        @verified(
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            post=lambda result: columns_present(result, ["a", "b", "c"]),
        )
        def my_fn() -> pa.Table:
            return pa.table({"a": [1], "b": [2]})

        with pytest.raises(ContractError, match="Postcondition"):
            my_fn()

    def test_attaches_version_metadata(self) -> None:
        versions = {"pyarrow": "23.0.1", "pandas": "3.0.2"}

        @verified(
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
        )
        def my_fn() -> None:
            pass

        assert my_fn.__version_requirements__ == {"pyarrow": ">=14.0.0"}
        assert my_fn.__locked_versions__ == {"pyarrow": "23.0.1"}

    def test_preserves_function_name(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        @verified(require={"pyarrow": ">=14.0.0"}, lock_versions=versions)
        def important_function() -> None:
            pass

        assert important_function.__name__ == "important_function"


# ── version_switch ───────────────────────────────────────────────────


class TestVersionSwitch:
    def test_selects_matching_branch(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        result = version_switch(
            lock_versions=versions,
            package="pyarrow",
            branches={
                ">=14.0.0": "modern_impl",
                "<14.0.0": "legacy_impl",
            },
        )
        assert result == "modern_impl"

    def test_first_match_wins(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        result = version_switch(
            lock_versions=versions,
            package="pyarrow",
            branches={
                ">=10.0.0": "broad",
                ">=23.0.0": "narrow",
            },
        )
        assert result == "broad"

    def test_no_match_raises(self) -> None:
        versions = {"pyarrow": "5.0.0"}
        with pytest.raises(VersionError, match="No branch matches"):
            version_switch(
                lock_versions=versions,
                package="pyarrow",
                branches={">=14.0.0": "modern"},
            )

    def test_no_match_uses_default(self) -> None:
        versions = {"pyarrow": "5.0.0"}
        result = version_switch(
            lock_versions=versions,
            package="pyarrow",
            branches={">=14.0.0": "modern"},
            default="fallback",
        )
        assert result == "fallback"

    def test_missing_package_raises(self) -> None:
        versions = {"pandas": "3.0.2"}
        with pytest.raises(VersionError, match="not found"):
            version_switch(
                lock_versions=versions,
                package="pyarrow",
                branches={">=14.0.0": "modern"},
            )

    def test_with_callable_branches(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        def modern(t: pa.Table) -> int:
            return t.num_rows

        def legacy(t: pa.Table) -> int:
            return len(t)

        fn = version_switch(
            lock_versions=versions,
            package="pyarrow",
            branches={">=14.0.0": modern, "<14.0.0": legacy},
        )
        assert fn(pa.table({"a": [1, 2, 3]})) == 3


# ── Table contract predicates ────────────────────────────────────────


class TestColumnsPresent:
    def test_all_present(self) -> None:
        table = pa.table({"a": [1], "b": [2], "c": [3]})
        assert columns_present(table, ["a", "b"]) is True

    def test_missing_column(self) -> None:
        table = pa.table({"a": [1], "b": [2]})
        assert columns_present(table, ["a", "c"]) is False

    def test_empty_columns_list(self) -> None:
        table = pa.table({"a": [1]})
        assert columns_present(table, []) is True


class TestNoNullsIn:
    def test_clean_data(self) -> None:
        table = pa.table({"a": [1, 2], "b": [3, 4]})
        assert no_nulls_in(table, ["a", "b"]) is True

    def test_null_present(self) -> None:
        table = pa.table({"a": [1, None], "b": [3, 4]})
        assert no_nulls_in(table, ["a"]) is False

    def test_null_in_unchecked_column(self) -> None:
        table = pa.table({"a": [1, 2], "b": [None, 4]})
        assert no_nulls_in(table, ["a"]) is True


class TestUniqueOn:
    def test_unique(self) -> None:
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert unique_on(table, ["a"]) is True

    def test_duplicates(self) -> None:
        table = pa.table({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        assert unique_on(table, ["a"]) is False

    def test_composite_key_unique(self) -> None:
        table = pa.table({"a": [1, 1, 2], "b": ["x", "y", "x"]})
        assert unique_on(table, ["a", "b"]) is True

    def test_composite_key_duplicate(self) -> None:
        table = pa.table({"a": [1, 1], "b": ["x", "x"]})
        assert unique_on(table, ["a", "b"]) is False


class TestNonEmpty:
    def test_non_empty(self) -> None:
        assert non_empty(pa.table({"a": [1]})) is True

    def test_empty(self) -> None:
        empty = pa.table({"a": pa.array([], type=pa.int64())})
        assert non_empty(empty) is False


class TestRowCountBetween:
    def test_within_range(self) -> None:
        table = pa.table({"a": [1, 2, 3]})
        assert row_count_between(table, 1, 5) is True

    def test_at_boundaries(self) -> None:
        table = pa.table({"a": [1, 2, 3]})
        assert row_count_between(table, 3, 3) is True

    def test_below_range(self) -> None:
        table = pa.table({"a": [1]})
        assert row_count_between(table, 2, 5) is False

    def test_above_range(self) -> None:
        table = pa.table({"a": [1, 2, 3, 4, 5, 6]})
        assert row_count_between(table, 1, 5) is False


class TestSchemaFieldIs:
    def test_matching_type(self) -> None:
        table = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
        assert schema_field_is(table, "a", pa.int64()) is True

    def test_mismatched_type(self) -> None:
        table = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
        assert schema_field_is(table, "a", pa.string()) is False

    def test_string_column(self) -> None:
        table = pa.table({"name": pa.array(["a", "b"])})
        actual_type = table.schema.field("name").type
        assert schema_field_is(table, "name", actual_type) is True
        assert schema_field_is(table, "name", pa.int64()) is False


# ── Integration: verified with real lock file ────────────────────────


class TestVerifiedWithLockFile:
    def test_end_to_end(self, lock_path: Path) -> None:
        versions = parse_lock_versions(lock_path)

        @verified(
            require={"pyarrow": ">=23.0.0,<24.0.0"},
            lock_versions=versions,
            pre=lambda table: non_empty(table) and columns_present(table, ["id", "val"]),
            post=lambda result: columns_present(result, ["id", "val", "doubled"]),
        )
        def double_val(table: pa.Table) -> pa.Table:
            import pyarrow.compute as pc

            doubled = pc.multiply(table.column("val"), 2)
            return table.append_column("doubled", doubled)

        table = pa.table({"id": [1, 2], "val": [10, 20]})
        result = double_val(table)
        assert result.column("doubled").to_pylist() == [20, 40]


# ── Verified wrapper ─────────────────────────────────────────────────


class TestVerifiedWrapper:
    def test_wraps_value(self) -> None:
        table = pa.table({"a": [1, 2]})
        v = Verified(table, frozenset({"check_a"}), {"pyarrow": "23.0.1"})
        assert v.inner is table

    def test_exposes_contracts(self) -> None:
        v = Verified(42, frozenset({"check_a", "check_b"}), {})
        assert v.contracts == frozenset({"check_a", "check_b"})

    def test_exposes_versions(self) -> None:
        v = Verified(42, frozenset(), {"pyarrow": "23.0.1"})
        assert v.versions == {"pyarrow": "23.0.1"}

    def test_versions_returns_copy(self) -> None:
        v = Verified(42, frozenset(), {"pyarrow": "23.0.1"})
        v.versions["hacked"] = "1.0.0"
        assert "hacked" not in v.versions

    def test_unsafe_bypasses_checks(self) -> None:
        table = pa.table({"a": [1]})
        v = Verified.unsafe(table)
        assert v.inner is table
        assert "__unsafe__" in v.contracts
        assert v.versions == {}

    def test_repr_contains_value_and_contracts(self) -> None:
        v = Verified(42, frozenset({"check"}), {})
        assert "Verified(42" in repr(v)
        assert "check" in repr(v)


# ── ContractSpec ─────────────────────────────────────────────────────


class TestContractSpec:
    def test_construction_validates_versions(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
        )
        assert spec.name == "test"
        assert spec.checked_versions == {"pyarrow": "23.0.1"}

    def test_construction_fails_on_bad_version(self) -> None:
        versions = {"pyarrow": "10.0.0"}
        with pytest.raises(VersionError):
            ContractSpec(
                name="test",
                require={"pyarrow": ">=14.0.0"},
                lock_versions=versions,
            )

    def test_construction_fails_on_missing_package(self) -> None:
        versions = {"pandas": "3.0.2"}
        with pytest.raises(VersionError):
            ContractSpec(
                name="test",
                require={"pyarrow": ">=14.0.0"},
                lock_versions=versions,
            )

    def test_verify_returns_verified(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            preconditions={
                "non_empty": lambda table, **kw: non_empty(table),
            },
        )
        table = pa.table({"a": [1, 2, 3]})
        result = spec.verify(table)
        assert isinstance(result, Verified)
        assert result.inner is table
        assert "non_empty" in result.contracts
        assert result.versions == {"pyarrow": "23.0.1"}

    def test_verify_fails_on_precondition(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            preconditions={
                "non_empty": lambda table, **kw: non_empty(table),
            },
        )
        empty = pa.table({"a": pa.array([], type=pa.int64())})
        with pytest.raises(ContractError, match="non_empty"):
            spec.verify(empty)

    def test_verify_with_context(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            preconditions={
                "has_key_cols": lambda table, **kw: columns_present(table, kw["keys"]),
            },
        )
        table = pa.table({"id": [1], "val": [2]})
        result = spec.verify(table, keys=["id"])
        assert result.inner is table

    def test_check_output_returns_verified(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            postconditions={
                "has_cols": lambda table: columns_present(table, ["a", "b"]),
            },
        )
        table = pa.table({"a": [1], "b": [2]})
        result = spec.check_output(table)
        assert isinstance(result, Verified)
        assert "has_cols" in result.contracts

    def test_check_output_fails(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            postconditions={
                "has_cols": lambda table: columns_present(table, ["a", "b", "c"]),
            },
        )
        with pytest.raises(ContractError, match="has_cols"):
            spec.check_output(pa.table({"a": [1], "b": [2]}))

    def test_wrap_returns_verified(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            preconditions={
                "non_empty": lambda table: non_empty(table),
            },
            postconditions={
                "has_out": lambda result: columns_present(result, ["a", "doubled"]),
            },
        )

        def double_col(table: pa.Table) -> pa.Table:
            import pyarrow.compute as pc

            return table.append_column("doubled", pc.multiply(table.column("a"), 2))

        safe_fn = spec.wrap(double_col)
        result = safe_fn(pa.table({"a": [1, 2]}))
        assert isinstance(result, Verified)
        assert result.inner.column("doubled").to_pylist() == [2, 4]
        assert "non_empty" in result.contracts
        assert "has_out" in result.contracts

    def test_wrap_precondition_fails(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            preconditions={
                "non_empty": lambda table: non_empty(table),
            },
        )

        safe_fn = spec.wrap(lambda table: table)
        empty = pa.table({"a": pa.array([], type=pa.int64())})
        with pytest.raises(ContractError, match="Precondition.*non_empty"):
            safe_fn(empty)

    def test_wrap_postcondition_fails(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            postconditions={
                "has_c": lambda result: columns_present(result, ["c"]),
            },
        )

        safe_fn = spec.wrap(lambda: pa.table({"a": [1]}))
        with pytest.raises(ContractError, match="Postcondition.*has_c"):
            safe_fn()

    def test_wrap_preserves_function_name(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
        )

        def my_function() -> pa.Table:
            return pa.table({"a": [1]})

        safe_fn = spec.wrap(my_function)
        assert safe_fn.__name__ == "my_function"

    def test_wrap_attaches_spec(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
        )

        safe_fn = spec.wrap(lambda: None)
        assert safe_fn.__contract_spec__ is spec

    def test_no_preconditions_or_postconditions(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="test",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
        )

        safe_fn = spec.wrap(lambda x: x + 1)
        result = safe_fn(5)
        assert isinstance(result, Verified)
        assert result.inner == 6
        assert result.contracts == frozenset()


# ── Safe/unsafe distinction ──────────────────────────────────────────


class TestSafeUnsafeDistinction:
    """Demonstrate the Rust-like safe/unsafe boundary."""

    def test_safe_path_through_spec(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="transform",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            preconditions={
                "non_empty": lambda table: non_empty(table),
            },
        )

        safe_fn = spec.wrap(lambda table: table)
        result = safe_fn(pa.table({"a": [1]}))

        assert isinstance(result, Verified)
        assert "__unsafe__" not in result.contracts
        assert "non_empty" in result.contracts

    def test_unsafe_path_is_explicit(self) -> None:
        table = pa.table({"a": [1]})
        result = Verified.unsafe(table)

        assert isinstance(result, Verified)
        assert "__unsafe__" in result.contracts
        assert result.versions == {}

    def test_safe_and_unsafe_are_distinguishable(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        spec = ContractSpec(
            name="check",
            require={"pyarrow": ">=14.0.0"},
            lock_versions=versions,
            preconditions={"exists": lambda table: non_empty(table)},
        )

        safe = spec.verify(pa.table({"a": [1]}))
        unsafe = Verified.unsafe(pa.table({"a": [1]}))

        assert "__unsafe__" not in safe.contracts
        assert "__unsafe__" in unsafe.contracts
        assert safe.versions != unsafe.versions

    def test_spec_wrap_end_to_end_with_lock_file(self, lock_path: Path) -> None:
        import pyarrow.compute as pc

        versions = parse_lock_versions(lock_path)

        spec = ContractSpec(
            name="bitemporal",
            require={"pyarrow": ">=23.0.0,<24.0.0"},
            lock_versions=versions,
            preconditions={
                "non_empty": lambda table: non_empty(table),
                "has_inputs": lambda table: columns_present(table, ["id", "val"]),
            },
            postconditions={
                "has_output": lambda r: columns_present(r, ["id", "val", "doubled"]),
            },
        )

        def double_val(table: pa.Table) -> pa.Table:
            doubled = pc.multiply(table.column("val"), 2)
            return table.append_column("doubled", doubled)

        safe_fn = spec.wrap(double_val)
        result = safe_fn(pa.table({"id": [1, 2], "val": [10, 20]}))

        assert isinstance(result, Verified)
        assert result.inner.column("doubled").to_pylist() == [20, 40]
        assert result.contracts == frozenset({"non_empty", "has_inputs", "has_output"})
        assert result.versions == {"pyarrow": "23.0.1"}


# ── Version-optional usage (the 90% path) ────────────────────────────


class TestPureDataContracts:
    """Contracts without any version requirements — the primary use case."""

    def test_verified_decorator_no_versions(self) -> None:
        @verified(
            pre=lambda table: non_empty(table),
            post=lambda result: columns_present(result, ["a", "out"]),
        )
        def transform(table: pa.Table) -> pa.Table:
            import pyarrow.compute as pc

            return table.append_column("out", pc.multiply(table.column("a"), 2))

        result = transform(pa.table({"a": [1, 2]}))
        assert result.column("out").to_pylist() == [2, 4]

    def test_verified_decorator_pre_fails_no_versions(self) -> None:
        @verified(pre=lambda table: non_empty(table))
        def transform(table: pa.Table) -> pa.Table:
            return table

        empty = pa.table({"a": pa.array([], type=pa.int64())})
        with pytest.raises(ContractError, match="Precondition"):
            transform(empty)

    def test_verified_decorator_post_fails_no_versions(self) -> None:
        @verified(post=lambda result: columns_present(result, ["missing"]))
        def transform() -> pa.Table:
            return pa.table({"a": [1]})

        with pytest.raises(ContractError, match="Postcondition"):
            transform()

    def test_verified_decorator_metadata_empty_when_no_versions(self) -> None:
        @verified(pre=lambda x: True)
        def my_fn(x: int) -> int:
            return x

        assert my_fn.__version_requirements__ == {}
        assert my_fn.__locked_versions__ == {}

    def test_verified_require_without_lock_raises(self) -> None:
        with pytest.raises(ValueError, match="lock_versions must be provided"):
            @verified(require={"pyarrow": ">=14.0.0"})
            def my_fn() -> None:
                pass

    def test_contract_spec_no_versions(self) -> None:
        spec = ContractSpec(
            name="pure_contract",
            preconditions={"non_empty": lambda table: non_empty(table)},
            postconditions={"has_cols": lambda r: columns_present(r, ["a", "out"])},
        )

        def transform(table: pa.Table) -> pa.Table:
            import pyarrow.compute as pc

            return table.append_column("out", pc.multiply(table.column("a"), 2))

        safe_fn = spec.wrap(transform)
        result = safe_fn(pa.table({"a": [1, 2, 3]}))

        assert isinstance(result, Verified)
        assert result.inner.column("out").to_pylist() == [2, 4, 6]
        assert "non_empty" in result.contracts
        assert "has_cols" in result.contracts
        assert result.versions == {}

    def test_contract_spec_verify_no_versions(self) -> None:
        spec = ContractSpec(
            name="input_check",
            preconditions={
                "non_empty": lambda table, **kw: non_empty(table),
                "has_id": lambda table, **kw: columns_present(table, ["id"]),
            },
        )

        table = pa.table({"id": [1, 2], "val": [10, 20]})
        result = spec.verify(table)

        assert isinstance(result, Verified)
        assert result.inner is table
        assert result.versions == {}
        assert "non_empty" in result.contracts
        assert "has_id" in result.contracts

    def test_contract_spec_require_without_lock_raises(self) -> None:
        with pytest.raises(ValueError, match="lock_versions must be provided"):
            ContractSpec(
                name="test",
                require={"pyarrow": ">=14.0.0"},
            )

    def test_contract_spec_checked_versions_empty(self) -> None:
        spec = ContractSpec(
            name="test",
            preconditions={"check": lambda x: True},
        )
        assert spec.checked_versions == {}
