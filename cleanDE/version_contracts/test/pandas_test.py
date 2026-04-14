from pathlib import Path

import pandas as pd
import pytest

from cleanDE.version_contracts.pandas_impl import (
    ContractError,
    VersionError,
    _parse_version,
    check_version_constraint,
    columns_present,
    detect_drift,
    dtype_is,
    no_nulls_in,
    non_empty,
    parse_lock_versions,
    row_count_between,
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

    def test_skips_packages_without_version(self, tmp_path: Path) -> None:
        p = tmp_path / "partial.lock"
        p.write_text(
            'version = 1\n'
            '[[package]]\n'
            'name = "noversion"\n'
            'source = { virtual = "." }\n'
        )
        assert "noversion" not in parse_lock_versions(p)


# ── Version parsing ──────────────────────────────────────────────────


class TestParseVersion:
    def test_standard_version(self) -> None:
        assert _parse_version("3.0.2") == (3, 0, 2)

    def test_large_components(self) -> None:
        assert _parse_version("23.0.1") == (23, 0, 1)

    def test_prerelease_stripped(self) -> None:
        assert _parse_version("3.0.0rc1") == (3, 0, 0)

    def test_single_component(self) -> None:
        assert _parse_version("5") == (5,)

    def test_many_components(self) -> None:
        assert _parse_version("1.2.3.4") == (1, 2, 3, 4)


# ── Version constraint checking ──────────────────────────────────────


class TestCheckVersionConstraint:
    @pytest.mark.parametrize(
        "version,constraint,expected",
        [
            ("3.0.2", ">=3.0.0", True),
            ("2.9.0", ">=3.0.0", False),
            ("3.0.0", ">=3.0.0", True),
            ("3.5.0", ">=3.0.0,<4.0.0", True),
            ("4.0.0", ">=3.0.0,<4.0.0", False),
            ("3.0.0", ">=3.0.0,<4.0.0", True),
            ("3.0.2", "==3.0.2", True),
            ("3.0.3", "==3.0.2", False),
            ("3.0.0", "!=2.0.0", True),
            ("2.0.0", "!=2.0.0", False),
            ("3.0.0", ">2.0.0", True),
            ("2.0.0", ">2.0.0", False),
            ("1.9.9", ">2.0.0", False),
            ("2.0.0", "<=2.0.0", True),
            ("2.0.1", "<=2.0.0", False),
            ("1.0.0", "<=2.0.0", True),
            ("2.0.0", "<3.0.0", True),
            ("3.0.0", "<3.0.0", False),
        ],
    )
    def test_constraint(self, version: str, constraint: str, expected: bool) -> None:
        assert check_version_constraint(version, constraint) == expected

    def test_unsupported_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            check_version_constraint("3.0.0", "~=3.0.0")


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

    def test_skips_packages_not_in_lock(self) -> None:
        lock = {"pandas": "3.0.2"}
        assert detect_drift(lock, ["pyarrow"]) == {}


# ── verified decorator ───────────────────────────────────────────────


class TestVerified:
    def test_passes_when_requirements_met(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(require={"pandas": ">=3.0.0"}, lock_versions=versions)
        def my_fn(x: int) -> int:
            return x + 1

        assert my_fn(1) == 2

    def test_fails_when_version_too_low(self) -> None:
        versions = {"pandas": "2.0.0"}

        with pytest.raises(VersionError, match="locked at 2.0.0"):

            @verified(require={"pandas": ">=3.0.0"}, lock_versions=versions)
            def my_fn(x: int) -> int:
                return x + 1

    def test_fails_when_package_missing(self) -> None:
        versions = {"pyarrow": "23.0.1"}

        with pytest.raises(VersionError, match="not found"):

            @verified(require={"pandas": ">=3.0.0"}, lock_versions=versions)
            def my_fn(x: int) -> int:
                return x + 1

    def test_multiple_requirements(self) -> None:
        versions = {"pandas": "3.0.2", "pyarrow": "23.0.1"}

        @verified(
            require={"pandas": ">=3.0.0", "pyarrow": ">=14.0.0"},
            lock_versions=versions,
        )
        def my_fn() -> str:
            return "ok"

        assert my_fn() == "ok"

    def test_multiple_requirements_one_fails(self) -> None:
        versions = {"pandas": "3.0.2", "pyarrow": "10.0.0"}

        with pytest.raises(VersionError, match="pyarrow"):

            @verified(
                require={"pandas": ">=3.0.0", "pyarrow": ">=14.0.0"},
                lock_versions=versions,
            )
            def my_fn() -> str:
                return "ok"

    def test_precondition_passes(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(
            require={"pandas": ">=3.0.0"},
            lock_versions=versions,
            pre=lambda df: non_empty(df),
        )
        def my_fn(df: pd.DataFrame) -> pd.DataFrame:
            return df

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = my_fn(df)
        assert len(result) == 3

    def test_precondition_fails(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(
            require={"pandas": ">=3.0.0"},
            lock_versions=versions,
            pre=lambda df: non_empty(df),
        )
        def my_fn(df: pd.DataFrame) -> pd.DataFrame:
            return df

        with pytest.raises(ContractError, match="Precondition"):
            my_fn(pd.DataFrame())

    def test_postcondition_passes(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(
            require={"pandas": ">=3.0.0"},
            lock_versions=versions,
            post=lambda result: columns_present(result, ["a", "b"]),
        )
        def my_fn() -> pd.DataFrame:
            return pd.DataFrame({"a": [1], "b": [2]})

        result = my_fn()
        assert list(result.columns) == ["a", "b"]

    def test_postcondition_fails(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(
            require={"pandas": ">=3.0.0"},
            lock_versions=versions,
            post=lambda result: columns_present(result, ["a", "b", "c"]),
        )
        def my_fn() -> pd.DataFrame:
            return pd.DataFrame({"a": [1], "b": [2]})

        with pytest.raises(ContractError, match="Postcondition"):
            my_fn()

    def test_both_pre_and_post(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(
            require={"pandas": ">=3.0.0"},
            lock_versions=versions,
            pre=lambda df: non_empty(df),
            post=lambda result: columns_present(result, ["a", "out"]),
        )
        def my_fn(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["out"] = df["a"] * 2
            return df

        result = my_fn(pd.DataFrame({"a": [1, 2]}))
        assert list(result["out"]) == [2, 4]

    def test_preserves_function_name(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(require={"pandas": ">=3.0.0"}, lock_versions=versions)
        def important_function() -> None:
            pass

        assert important_function.__name__ == "important_function"

    def test_attaches_version_metadata(self) -> None:
        versions = {"pandas": "3.0.2", "pyarrow": "23.0.1"}

        @verified(
            require={"pandas": ">=3.0.0"},
            lock_versions=versions,
        )
        def my_fn() -> None:
            pass

        assert my_fn.__version_requirements__ == {"pandas": ">=3.0.0"}
        assert my_fn.__locked_versions__ == {"pandas": "3.0.2"}

    def test_no_contracts_just_version_guard(self) -> None:
        versions = {"pandas": "3.0.2"}

        @verified(require={"pandas": ">=3.0.0"}, lock_versions=versions)
        def my_fn(x: int, y: int) -> int:
            return x + y

        assert my_fn(2, 3) == 5


# ── version_switch ───────────────────────────────────────────────────


class TestVersionSwitch:
    def test_selects_matching_branch(self) -> None:
        versions = {"pandas": "3.0.2"}
        result = version_switch(
            lock_versions=versions,
            package="pandas",
            branches={
                ">=3.0.0": "v3_impl",
                ">=2.0.0,<3.0.0": "v2_impl",
            },
        )
        assert result == "v3_impl"

    def test_first_match_wins(self) -> None:
        versions = {"pandas": "3.0.2"}
        result = version_switch(
            lock_versions=versions,
            package="pandas",
            branches={
                ">=2.0.0": "broad",
                ">=3.0.0": "narrow",
            },
        )
        assert result == "broad"

    def test_selects_second_branch(self) -> None:
        versions = {"pandas": "2.5.0"}
        result = version_switch(
            lock_versions=versions,
            package="pandas",
            branches={
                ">=3.0.0": "v3_impl",
                ">=2.0.0,<3.0.0": "v2_impl",
            },
        )
        assert result == "v2_impl"

    def test_no_match_raises(self) -> None:
        versions = {"pandas": "1.0.0"}
        with pytest.raises(VersionError, match="No branch matches"):
            version_switch(
                lock_versions=versions,
                package="pandas",
                branches={">=2.0.0": "v2", ">=3.0.0": "v3"},
            )

    def test_no_match_uses_default(self) -> None:
        versions = {"pandas": "1.0.0"}
        result = version_switch(
            lock_versions=versions,
            package="pandas",
            branches={">=2.0.0": "v2"},
            default="fallback",
        )
        assert result == "fallback"

    def test_missing_package_raises(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        with pytest.raises(VersionError, match="not found"):
            version_switch(
                lock_versions=versions,
                package="pandas",
                branches={">=3.0.0": "v3"},
            )

    def test_missing_package_uses_default(self) -> None:
        versions = {"pyarrow": "23.0.1"}
        result = version_switch(
            lock_versions=versions,
            package="pandas",
            branches={">=3.0.0": "v3"},
            default="fallback",
        )
        assert result == "fallback"

    def test_with_callable_branches(self) -> None:
        versions = {"pandas": "3.0.2"}

        def impl_v3(x: int) -> int:
            return x * 3

        def impl_v2(x: int) -> int:
            return x * 2

        fn = version_switch(
            lock_versions=versions,
            package="pandas",
            branches={">=3.0.0": impl_v3, ">=2.0.0,<3.0.0": impl_v2},
        )
        assert fn(10) == 30


# ── Contract predicates ──────────────────────────────────────────────


class TestColumnsPresent:
    def test_all_present(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert columns_present(df, ["a", "b"]) is True

    def test_missing_column(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert columns_present(df, ["a", "c"]) is False

    def test_empty_columns_list(self) -> None:
        df = pd.DataFrame({"a": [1]})
        assert columns_present(df, []) is True


class TestNoNullsIn:
    def test_clean_data(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert no_nulls_in(df, ["a", "b"]) is True

    def test_null_present(self) -> None:
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        assert no_nulls_in(df, ["a"]) is False

    def test_null_in_unchecked_column(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [None, 4]})
        assert no_nulls_in(df, ["a"]) is True


class TestUniqueOn:
    def test_unique(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert unique_on(df, ["a"]) is True

    def test_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        assert unique_on(df, ["a"]) is False

    def test_composite_key_unique(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
        assert unique_on(df, ["a", "b"]) is True

    def test_composite_key_duplicate(self) -> None:
        df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
        assert unique_on(df, ["a", "b"]) is False


class TestNonEmpty:
    def test_non_empty(self) -> None:
        assert non_empty(pd.DataFrame({"a": [1]})) is True

    def test_empty(self) -> None:
        assert non_empty(pd.DataFrame()) is False

    def test_empty_with_columns(self) -> None:
        assert non_empty(pd.DataFrame(columns=["a", "b"])) is False


class TestRowCountBetween:
    def test_within_range(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert row_count_between(df, 1, 5) is True

    def test_at_boundaries(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert row_count_between(df, 3, 3) is True

    def test_below_range(self) -> None:
        df = pd.DataFrame({"a": [1]})
        assert row_count_between(df, 2, 5) is False

    def test_above_range(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        assert row_count_between(df, 1, 5) is False


class TestDtypeIs:
    def test_matching_dtype(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert dtype_is(df, "a", str(df["a"].dtype)) is True

    def test_mismatched_dtype(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert dtype_is(df, "a", "object") is False

    def test_string_column(self) -> None:
        df = pd.DataFrame({"a": ["x", "y"]})
        actual_dtype = str(df["a"].dtype)
        assert dtype_is(df, "a", actual_dtype) is True


# ── Integration: verified with real lock file ────────────────────────


class TestVerifiedWithLockFile:
    def test_end_to_end(self, lock_path: Path) -> None:
        versions = parse_lock_versions(lock_path)

        @verified(
            require={"pandas": ">=3.0.0,<4.0.0"},
            lock_versions=versions,
            pre=lambda df: non_empty(df) and columns_present(df, ["id", "val"]),
            post=lambda result: columns_present(result, ["id", "val", "doubled"]),
        )
        def double_val(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["doubled"] = df["val"] * 2
            return df

        result = double_val(pd.DataFrame({"id": [1, 2], "val": [10, 20]}))
        assert list(result["doubled"]) == [20, 40]
