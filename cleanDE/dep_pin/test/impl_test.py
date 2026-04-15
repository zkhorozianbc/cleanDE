from pathlib import Path

import pytest

from cleanDE.dep_pin.impl import (
    DependencyError,
    _parse_version,
    _satisfies,
    check,
    parse_lock,
    parse_requirements,
    pin,
    requires,
    resolve,
    select,
)

SAMPLE_LOCK = """\
version = 1
requires-python = ">=3.12"

[[package]]
name = "cleande"
version = "0.1.0"
source = { virtual = "." }
dependencies = [
    { name = "pandas" },
    { name = "pyarrow" },
]

[package.metadata]
requires-dist = [
    { name = "pandas", specifier = ">=3.0.2" },
    { name = "pyarrow", specifier = ">=23.0.1" },
]

[[package]]
name = "pandas"
version = "3.0.2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pyarrow"
version = "23.0.1"
source = { registry = "https://pypi.org/simple" }
"""


@pytest.fixture()
def lock_path(tmp_path: Path) -> Path:
    p = tmp_path / "uv.lock"
    p.write_text(SAMPLE_LOCK)
    return p


# ── parse_lock ───────────────────────────────────────────────────────


class TestParseLock:
    def test_extracts_packages(self) -> None:
        versions = parse_lock(SAMPLE_LOCK)
        assert versions["pandas"] == (3, 0, 2)
        assert versions["pyarrow"] == (23, 0, 1)
        assert versions["cleande"] == (0, 1, 0)

    def test_skips_packages_without_version(self) -> None:
        text = (
            'version = 1\n'
            '[[package]]\n'
            'name = "noversion"\n'
            'source = { virtual = "." }\n'
        )
        assert "noversion" not in parse_lock(text)

    def test_empty_lock(self) -> None:
        assert parse_lock('version = 1\n') == {}

    def test_prerelease_stripped(self) -> None:
        text = (
            'version = 1\n'
            '[[package]]\n'
            'name = "beta"\n'
            'version = "3.0.0rc1"\n'
        )
        assert parse_lock(text)["beta"] == (3, 0, 0)


# ── parse_requirements ───────────────────────────────────────────────


class TestParseRequirements:
    def test_extracts_from_root_package(self) -> None:
        reqs = parse_requirements(SAMPLE_LOCK)
        assert reqs["pandas"] == ">=3.0.2"
        assert reqs["pyarrow"] == ">=23.0.1"

    def test_only_from_virtual_source(self) -> None:
        # Non-virtual packages' requires-dist should be ignored
        reqs = parse_requirements(SAMPLE_LOCK)
        # Only the root package's direct deps, not transitive ones
        assert "numpy" not in reqs

    def test_no_root_package(self) -> None:
        text = (
            'version = 1\n'
            '[[package]]\n'
            'name = "pandas"\n'
            'version = "3.0.2"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )
        assert parse_requirements(text) == {}

    def test_root_without_metadata(self) -> None:
        text = (
            'version = 1\n'
            '[[package]]\n'
            'name = "myproject"\n'
            'version = "0.1.0"\n'
            'source = { virtual = "." }\n'
        )
        assert parse_requirements(text) == {}


# ── _parse_version ───────────────────────────────────────────────────


class TestParseVersion:
    def test_standard(self) -> None:
        assert _parse_version("3.0.2") == (3, 0, 2)

    def test_two_components(self) -> None:
        assert _parse_version("23.0") == (23, 0)

    def test_four_components(self) -> None:
        assert _parse_version("1.2.3.4") == (1, 2, 3, 4)

    def test_prerelease(self) -> None:
        assert _parse_version("3.0.0rc1") == (3, 0, 0)


# ── _satisfies ───────────────────────────────────────────────────────


class TestSatisfies:
    @pytest.mark.parametrize(
        "version,constraint,expected",
        [
            ((3, 0, 2), ">=3.0.0", True),
            ((2, 9, 0), ">=3.0.0", False),
            ((3, 0, 0), ">=3.0.0", True),
            ((3, 5, 0), ">=3.0.0,<4.0.0", True),
            ((4, 0, 0), ">=3.0.0,<4.0.0", False),
            ((3, 0, 2), "==3.0.2", True),
            ((3, 0, 3), "==3.0.2", False),
            ((3, 0, 0), "!=2.0.0", True),
            ((2, 0, 0), "!=2.0.0", False),
            ((3, 0, 0), ">2.0.0", True),
            ((2, 0, 0), ">2.0.0", False),
            ((2, 0, 0), "<=2.0.0", True),
            ((2, 0, 1), "<=2.0.0", False),
        ],
    )
    def test_constraint(
        self,
        version: tuple[int, ...],
        constraint: str,
        expected: bool,
    ) -> None:
        assert _satisfies(version, constraint) == expected

    def test_unsupported_operator(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            _satisfies((3, 0, 0), "~=3.0")


# ── check ────────────────────────────────────────────────────────────


class TestCheck:
    def test_passes_when_satisfied(self) -> None:
        versions = {"pandas": (3, 0, 2), "pyarrow": (23, 0, 1)}
        check(versions, {"pandas": ">=3.0.0", "pyarrow": ">=14.0"})

    def test_raises_on_violation(self) -> None:
        versions = {"pandas": (2, 0, 0)}
        with pytest.raises(DependencyError, match="pandas.*2.0.0.*>=3.0.0"):
            check(versions, {"pandas": ">=3.0.0"})

    def test_raises_on_missing_package(self) -> None:
        with pytest.raises(DependencyError, match="pandas.*<missing>"):
            check({}, {"pandas": ">=3.0.0"})

    def test_compound_constraint(self) -> None:
        versions = {"pandas": (3, 5, 0)}
        check(versions, {"pandas": ">=3.0.0,<4.0.0"})

    def test_empty_requirements(self) -> None:
        check({}, {})

    def test_error_has_structured_data(self) -> None:
        versions = {"pandas": (2, 0, 0)}
        with pytest.raises(DependencyError) as exc_info:
            check(versions, {"pandas": ">=3.0.0"})
        err = exc_info.value
        assert err.package == "pandas"
        assert err.locked == (2, 0, 0)
        assert err.constraint == ">=3.0.0"


# ── pin ──────────────────────────────────────────────────────────────


class TestPin:
    def test_returns_versions(self, lock_path: Path) -> None:
        versions = pin(lock_path, {"pandas": ">=3.0.0"})
        assert versions["pandas"] == (3, 0, 2)
        assert versions["pyarrow"] == (23, 0, 1)

    def test_raises_on_violation(self, lock_path: Path) -> None:
        with pytest.raises(DependencyError, match="pandas"):
            pin(lock_path, {"pandas": ">=4.0.0"})

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            pin(Path("/nonexistent/uv.lock"), {"pandas": ">=3.0"})

    def test_empty_requirements(self, lock_path: Path) -> None:
        versions = pin(lock_path, {})
        assert "pandas" in versions

    def test_native_reads_from_lock_metadata(self, lock_path: Path) -> None:
        # No require arg — reads requirements from the lock file itself
        versions = pin(lock_path)
        assert versions["pandas"] == (3, 0, 2)
        assert versions["pyarrow"] == (23, 0, 1)

    def test_native_raises_when_lock_violated(self, tmp_path: Path) -> None:
        # Lock file says requires pandas>=99.0, but resolves to 3.0.2
        lock = tmp_path / "uv.lock"
        lock.write_text(
            'version = 1\n'
            '[[package]]\n'
            'name = "myproject"\n'
            'version = "0.1.0"\n'
            'source = { virtual = "." }\n'
            '[package.metadata]\n'
            'requires-dist = [\n'
            '    { name = "pandas", specifier = ">=99.0" },\n'
            ']\n'
            '[[package]]\n'
            'name = "pandas"\n'
            'version = "3.0.2"\n'
        )
        with pytest.raises(DependencyError, match="pandas"):
            pin(lock)


# ── select ───────────────────────────────────────────────────────────


class TestSelect:
    def test_picks_matching_branch(self) -> None:
        versions = {"pandas": (3, 0, 2)}
        result = select(versions, "pandas", {
            ">=3.0": "v3",
            ">=2.0,<3.0": "v2",
        })
        assert result == "v3"

    def test_first_match_wins(self) -> None:
        versions = {"pandas": (3, 0, 2)}
        result = select(versions, "pandas", {
            ">=2.0": "broad",
            ">=3.0": "narrow",
        })
        assert result == "broad"

    def test_second_branch(self) -> None:
        versions = {"pandas": (2, 5, 0)}
        result = select(versions, "pandas", {
            ">=3.0": "v3",
            ">=2.0,<3.0": "v2",
        })
        assert result == "v2"

    def test_no_match_raises(self) -> None:
        versions = {"pandas": (1, 0, 0)}
        with pytest.raises(ValueError, match="No branch matches"):
            select(versions, "pandas", {">=2.0": "v2"})

    def test_no_match_uses_default(self) -> None:
        versions = {"pandas": (1, 0, 0)}
        result = select(versions, "pandas", {">=2.0": "v2"}, default="fallback")
        assert result == "fallback"

    def test_missing_package_raises(self) -> None:
        with pytest.raises(DependencyError, match="pandas"):
            select({}, "pandas", {">=3.0": "v3"})

    def test_missing_package_uses_default(self) -> None:
        result = select({}, "pandas", {">=3.0": "v3"}, default="fallback")
        assert result == "fallback"

    def test_with_callable_branches(self) -> None:
        versions = {"pandas": (3, 0, 2)}

        def v3(x: int) -> int:
            return x * 3

        def v2(x: int) -> int:
            return x * 2

        fn = select(versions, "pandas", {">=3.0": v3, ">=2.0,<3.0": v2})
        assert fn(10) == 30


# ── Integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_pin_then_select(self, lock_path: Path) -> None:
        versions = pin(lock_path, {"pandas": ">=3.0", "pyarrow": ">=14.0"})

        label = select(versions, "pandas", {
            ">=3.0": "arrow-backed",
            ">=2.0,<3.0": "numpy-backed",
        })
        assert label == "arrow-backed"

        label = select(versions, "pyarrow", {
            ">=23.0": "modern",
            "<23.0": "legacy",
        })
        assert label == "modern"


# ── requires + resolve ───────────────────────────────────────────────


class TestRequires:
    def test_attaches_requirements(self) -> None:
        @requires(pandas=">=3.0")
        def fn() -> None:
            pass

        assert fn.__requires__ == {"pandas": ">=3.0"}

    def test_multiple_packages(self) -> None:
        @requires(pandas=">=3.0", pyarrow=">=14.0")
        def fn() -> None:
            pass

        assert fn.__requires__ == {"pandas": ">=3.0", "pyarrow": ">=14.0"}

    def test_does_not_change_behavior(self) -> None:
        @requires(pandas=">=3.0")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_name(self) -> None:
        @requires(pandas=">=3.0")
        def important_function() -> None:
            pass

        assert important_function.__name__ == "important_function"


class TestResolve:
    def test_picks_matching_candidate(self) -> None:
        versions = {"pandas": (3, 0, 2)}

        @requires(pandas=">=3.0")
        def v3(x: int) -> int:
            return x * 3

        @requires(pandas=">=2.0,<3.0")
        def v2(x: int) -> int:
            return x * 2

        fn = resolve(versions, v3, v2)
        assert fn(10) == 30

    def test_first_match_wins(self) -> None:
        versions = {"pandas": (3, 0, 2)}

        @requires(pandas=">=2.0")
        def broad() -> str:
            return "broad"

        @requires(pandas=">=3.0")
        def narrow() -> str:
            return "narrow"

        assert resolve(versions, broad, narrow)() == "broad"

    def test_second_candidate(self) -> None:
        versions = {"pandas": (2, 5, 0)}

        @requires(pandas=">=3.0")
        def v3() -> str:
            return "v3"

        @requires(pandas=">=2.0,<3.0")
        def v2() -> str:
            return "v2"

        assert resolve(versions, v3, v2)() == "v2"

    def test_multi_package_constraint(self) -> None:
        versions = {"pandas": (3, 0, 2), "pyarrow": (23, 0, 1)}

        @requires(pandas=">=3.0", pyarrow=">=23.0")
        def modern() -> str:
            return "modern"

        @requires(pandas=">=2.0")
        def fallback() -> str:
            return "fallback"

        assert resolve(versions, modern, fallback)() == "modern"

    def test_multi_package_partial_fail(self) -> None:
        versions = {"pandas": (3, 0, 2), "pyarrow": (10, 0, 0)}

        @requires(pandas=">=3.0", pyarrow=">=23.0")
        def modern() -> str:
            return "modern"

        @requires(pandas=">=2.0")
        def fallback() -> str:
            return "fallback"

        # modern fails because pyarrow is too old, falls through to fallback
        assert resolve(versions, modern, fallback)() == "fallback"

    def test_no_match_raises(self) -> None:
        versions = {"pandas": (1, 0, 0)}

        @requires(pandas=">=2.0")
        def v2() -> None:
            pass

        with pytest.raises(ValueError, match="No candidate matches"):
            resolve(versions, v2)

    def test_untagged_function_always_matches(self) -> None:
        versions = {"pandas": (1, 0, 0)}

        @requires(pandas=">=3.0")
        def strict() -> str:
            return "strict"

        def fallback() -> str:
            return "fallback"

        # fallback has no __requires__, so it matches anything
        assert resolve(versions, strict, fallback)() == "fallback"

    def test_pin_then_resolve(self, lock_path: Path) -> None:
        versions = pin(lock_path, {"pandas": ">=3.0"})

        @requires(pandas=">=3.0")
        def v3(x: int) -> int:
            return x * 3

        @requires(pandas=">=2.0,<3.0")
        def v2(x: int) -> int:
            return x * 2

        fn = resolve(versions, v3, v2)
        assert fn(10) == 30
