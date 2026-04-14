"""
Version-aware contracts for pandas functions.

Bridges the gap between your uv.lock dependency versions and your code's
runtime assumptions. Enforces preconditions and postconditions that can
vary based on which dependency versions are locked.

Three core capabilities:
  1. Lock file parsing — extract pinned versions from uv.lock
  2. Version-guarded contracts — a decorator that enforces pre/post
     conditions and fails fast when dependency versions fall outside
     declared ranges
  3. Version-conditional dispatch — select values or implementations
     based on the locked version of a dependency

Usage:
    from pathlib import Path
    from cleanDE.version_contracts.pandas_impl import (
        parse_lock_versions,
        verified,
        version_switch,
        non_empty,
        columns_present,
    )

    versions = parse_lock_versions(Path("uv.lock"))

    @verified(
        require={"pandas": ">=3.0.0"},
        lock_versions=versions,
        pre=lambda df, **kw: non_empty(df),
        post=lambda result: columns_present(result, ["id", "value"]),
    )
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        ...
"""

import functools
import importlib.metadata
import tomllib
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

import pandas as pd

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# ── Exceptions ───────────────────────────────────────────────────────


class VersionError(Exception):
    """Raised when a locked dependency version does not satisfy a requirement."""


class ContractError(Exception):
    """Raised when a pre- or postcondition is violated at call time."""


# ── Lock file parsing ────────────────────────────────────────────────


def parse_lock_versions(lock_path: Path) -> dict[str, str]:
    """Parse a uv.lock file and return a mapping of package names to locked versions.

    Parameters
    ----------
    lock_path : Path
        Path to the uv.lock file.

    Returns
    -------
    dict[str, str]
        Mapping of package names to their pinned version strings.

    Raises
    ------
    FileNotFoundError
        If the lock file does not exist.
    """
    with open(lock_path, "rb") as f:
        data = tomllib.load(f)
    return {
        pkg["name"]: pkg["version"]
        for pkg in data.get("package", [])
        if "version" in pkg
    }


# ── Version comparison ───────────────────────────────────────────────


def _parse_version(v: str) -> tuple[int, ...]:
    """Split a version string into a comparable tuple of integers.

    Handles standard versions (3.0.2) and strips pre-release suffixes
    (3.0.0rc1 becomes 3.0.0).

    Parameters
    ----------
    v : str
        Version string.

    Returns
    -------
    tuple[int, ...]
        Tuple of integer version components.
    """
    parts: list[int] = []
    for segment in v.strip().split("."):
        digits = ""
        for ch in segment:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_version_constraint(version: str, constraint: str) -> bool:
    """Check whether a version string satisfies a PEP 440-style constraint.

    Supports comma-separated constraints: ">=3.0.0,<4.0.0".
    Supported operators: >=, <=, >, <, ==, !=.

    Parameters
    ----------
    version : str
        The version to check (e.g. "3.0.2").
    constraint : str
        One or more comma-separated version constraints (e.g. ">=3.0.0,<4.0").

    Returns
    -------
    bool
        True if the version satisfies all constraints.

    Raises
    ------
    ValueError
        If a constraint uses an unsupported operator.
    """
    v = _parse_version(version)
    for part in constraint.split(","):
        part = part.strip()
        if part.startswith(">="):
            if v < _parse_version(part[2:]):
                return False
        elif part.startswith("<="):
            if v > _parse_version(part[2:]):
                return False
        elif part.startswith("!="):
            if v == _parse_version(part[2:]):
                return False
        elif part.startswith(">"):
            if v <= _parse_version(part[1:]):
                return False
        elif part.startswith("<"):
            if v >= _parse_version(part[1:]):
                return False
        elif part.startswith("=="):
            if v != _parse_version(part[2:]):
                return False
        else:
            raise ValueError(f"Unsupported version constraint: {part!r}")
    return True


# ── Drift detection ──────────────────────────────────────────────────


def detect_drift(
    lock_versions: dict[str, str],
    packages: list[str],
) -> dict[str, tuple[str, str]]:
    """Compare locked versions against installed versions to find mismatches.

    Parameters
    ----------
    lock_versions : dict[str, str]
        Mapping of package names to their locked versions
        (from parse_lock_versions).
    packages : list[str]
        Package names to check.

    Returns
    -------
    dict[str, tuple[str, str]]
        Mapping of drifted package names to (locked_version, installed_version).
        Only includes packages where versions differ.  Missing packages appear
        with installed_version set to "<not installed>".
    """
    drift: dict[str, tuple[str, str]] = {}
    for pkg in packages:
        locked_ver = lock_versions.get(pkg)
        if locked_ver is None:
            continue
        try:
            installed_ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            drift[pkg] = (locked_ver, "<not installed>")
            continue
        if _parse_version(installed_ver) != _parse_version(locked_ver):
            drift[pkg] = (locked_ver, installed_ver)
    return drift


# ── Version-guarded contracts ────────────────────────────────────────


def verified(
    require: dict[str, str],
    lock_versions: dict[str, str],
    pre: Callable[..., bool] | None = None,
    post: Callable[..., bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that enforces version requirements and data contracts.

    At decoration time, validates that every package in *require* exists in
    *lock_versions* and satisfies its constraint.  At call time, runs optional
    *pre* and *post* checks around the wrapped function.

    Parameters
    ----------
    require : dict[str, str]
        Mapping of package names to version constraints
        (e.g. {"pandas": ">=3.0.0"}).
    lock_versions : dict[str, str]
        Locked versions from parse_lock_versions().
    pre : Callable[..., bool] | None
        Precondition checked before every call.  Receives the same positional
        and keyword arguments as the decorated function.  Must return True
        when satisfied.
    post : Callable[..., bool] | None
        Postcondition checked after every call.  Receives the return value as
        its sole argument.  Must return True when satisfied.

    Returns
    -------
    Callable
        Decorator that wraps the target function with contract enforcement.

    Raises
    ------
    VersionError
        At decoration time, if a version requirement is not met or a required
        package is missing from the lock versions.
    ContractError
        At call time, if a pre- or postcondition fails.
    """
    for pkg, constraint in require.items():
        version = lock_versions.get(pkg)
        if version is None:
            raise VersionError(
                f"Package {pkg!r} not found in lock versions"
            )
        if not check_version_constraint(version, constraint):
            raise VersionError(
                f"Package {pkg!r} locked at {version}, "
                f"but {constraint!r} is required"
            )

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if pre is not None and not pre(*args, **kwargs):
                raise ContractError(
                    f"Precondition failed for {fn.__qualname__}"
                )
            result = fn(*args, **kwargs)
            if post is not None and not post(result):
                raise ContractError(
                    f"Postcondition failed for {fn.__qualname__}"
                )
            return result

        wrapper.__version_requirements__ = require  # type: ignore[attr-defined]
        wrapper.__locked_versions__ = {  # type: ignore[attr-defined]
            pkg: lock_versions.get(pkg) for pkg in require
        }
        return wrapper

    return decorator


# ── Version-conditional dispatch ─────────────────────────────────────


def version_switch(
    lock_versions: dict[str, str],
    package: str,
    branches: dict[str, T],
    default: T | None = None,
) -> T:
    """Select a value based on the locked version of a package.

    Iterates through *branches* in insertion order and returns the value
    associated with the first constraint that matches the locked version.
    Typically used to pick between implementation functions at module load
    time.

    Parameters
    ----------
    lock_versions : dict[str, str]
        Locked versions from parse_lock_versions().
    package : str
        Package name to look up.
    branches : dict[str, T]
        Mapping of version constraints to values (typically callables).
        Checked in insertion order; first match wins.
    default : T | None
        Fallback value if no branch matches or the package is absent.
        If None and no branch matches, raises VersionError.

    Returns
    -------
    T
        The value from the first matching branch, or *default*.

    Raises
    ------
    VersionError
        If the package is not in lock_versions and no default is provided,
        or if no branch matches and no default is provided.
    """
    version = lock_versions.get(package)
    if version is None:
        if default is not None:
            return default
        raise VersionError(f"Package {package!r} not found in lock versions")

    for constraint, value in branches.items():
        if check_version_constraint(version, constraint):
            return value

    if default is not None:
        return default
    raise VersionError(
        f"No branch matches {package}=={version}. "
        f"Defined branches: {list(branches.keys())}"
    )


# ── DataFrame contract predicates ────────────────────────────────────


def columns_present(df: pd.DataFrame, columns: list[str]) -> bool:
    """Check that every column name in *columns* exists in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : list[str]
        Expected column names.

    Returns
    -------
    bool
        True if all columns are present.
    """
    return all(c in df.columns for c in columns)


def no_nulls_in(df: pd.DataFrame, columns: list[str]) -> bool:
    """Check that the specified columns contain no null values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : list[str]
        Columns to inspect for nulls.

    Returns
    -------
    bool
        True if none of the specified columns contain nulls.
    """
    return not df[columns].isna().any().any()


def unique_on(df: pd.DataFrame, columns: list[str]) -> bool:
    """Check that the specified columns form a unique key.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : list[str]
        Columns that should jointly be unique.

    Returns
    -------
    bool
        True if no duplicate rows exist on the given columns.
    """
    return not df.duplicated(subset=columns).any()


def non_empty(df: pd.DataFrame) -> bool:
    """Check that the DataFrame has at least one row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.

    Returns
    -------
    bool
        True if the DataFrame is non-empty.
    """
    return len(df) > 0


def row_count_between(df: pd.DataFrame, low: int, high: int) -> bool:
    """Check that the row count falls within [low, high] inclusive.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    low : int
        Minimum acceptable row count.
    high : int
        Maximum acceptable row count.

    Returns
    -------
    bool
        True if low <= len(df) <= high.
    """
    return low <= len(df) <= high


def dtype_is(df: pd.DataFrame, column: str, expected: str) -> bool:
    """Check that a column's dtype matches the expected string representation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    column : str
        Column name.
    expected : str
        Expected dtype string (e.g. "int64", "object", "datetime64[ns]").

    Returns
    -------
    bool
        True if the column dtype matches.
    """
    return str(df[column].dtype) == expected
