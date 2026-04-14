"""
Version-aware contracts for pyarrow functions.

Bridges the gap between your uv.lock dependency versions and your code's
runtime assumptions.  Uses a Rust-inspired safe/unsafe model where
verified data is wrapped in a distinct type.

Core capabilities:
  1. Lock file parsing — extract pinned versions from uv.lock
  2. Verified[T] — a Rust-style newtype wrapper proving data passed checks
  3. ContractSpec — a metaprogram that generates verification pipelines
  4. @verified — a lightweight decorator for simple version + contract guards
  5. version_switch — select implementations based on locked versions

Usage (ContractSpec — the metaprogram):
    from pathlib import Path
    from cleanDE.version_contracts.pyarrow_impl import (
        parse_lock_versions, ContractSpec, Verified, non_empty, columns_present,
    )

    versions = parse_lock_versions(Path("uv.lock"))

    spec = ContractSpec(
        name="transform",
        require={"pyarrow": ">=23.0.0"},
        lock_versions=versions,
        preconditions={"non_empty": lambda t: non_empty(t)},
        postconditions={"has_output": lambda r: columns_present(r, ["out"])},
    )

    safe_transform = spec.wrap(transform)    # returns Verified[Table]
    result = safe_transform(my_table)        # result.inner is the Table
    raw = Verified.unsafe(my_table)          # explicit unsafe bypass
"""

import functools
import importlib.metadata
import tomllib
from pathlib import Path
from typing import Any, Callable, Generic, ParamSpec, TypeVar

import pyarrow as pa

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# ── Exceptions ───────────────────────────────────────────────────────


class VersionError(Exception):
    """Raised when a locked dependency version does not satisfy a requirement."""


class ContractError(Exception):
    """Raised when a pre- or postcondition is violated at call time."""


# ── Verified wrapper (Rust-inspired newtype) ─────────────────────────


class Verified(Generic[T]):
    """A value that has passed contract verification.

    Inspired by Rust's newtype pattern and safe/unsafe distinction.
    Obtaining a Verified[T] requires either:
      - Passing through ContractSpec.verify() or a wrapped function (safe)
      - Calling Verified.unsafe() to explicitly bypass checks (unsafe)

    Functions that accept Verified[T] encode at the type level that they
    require pre-checked input.  Passing unverified data is a type error,
    not a runtime surprise.

    Parameters
    ----------
    inner : T
        The value being wrapped.
    contracts : frozenset[str]
        Names of the contracts that were verified.
    versions : dict[str, str]
        Locked dependency versions at verification time.
    """

    __slots__ = ("_inner", "_contracts", "_versions")

    def __init__(
        self,
        inner: T,
        contracts: frozenset[str],
        versions: dict[str, str],
    ) -> None:
        self._inner = inner
        self._contracts = contracts
        self._versions = versions

    @property
    def inner(self) -> T:
        """The underlying verified value."""
        return self._inner

    @property
    def contracts(self) -> frozenset[str]:
        """Names of the contracts that were checked."""
        return self._contracts

    @property
    def versions(self) -> dict[str, str]:
        """Locked dependency versions at verification time (copy)."""
        return dict(self._versions)

    @staticmethod
    def unsafe(value: T) -> "Verified[T]":  # type: ignore[misc]
        """Bypass verification.  Caller assumes responsibility.

        Analogous to Rust's ``unsafe`` block: explicitly marks a value as
        trusted without running any checks.  The ``__unsafe__`` marker in
        ``.contracts`` makes these bypass points visible during audits.

        Parameters
        ----------
        value : T
            The value to mark as verified without checking.

        Returns
        -------
        Verified[T]
            Wrapper with contracts={"__unsafe__"} and empty versions.
        """
        return Verified(value, frozenset({"__unsafe__"}), {})

    def __repr__(self) -> str:
        return f"Verified({self._inner!r}, contracts={self._contracts})"


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
        The version to check (e.g. "23.0.1").
    constraint : str
        One or more comma-separated version constraints (e.g. ">=14.0.0,<24.0").

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
        (e.g. {"pyarrow": ">=23.0.0"}).
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


# ── ContractSpec (the metaprogram) ───────────────────────────────────


class ContractSpec:
    """A specification that generates verification pipelines.

    This is the metaprogram: instead of writing imperative checks, you
    declare a specification — name, version requirements, named pre- and
    postconditions — and the spec *generates* functions that enforce it.

    Mirrors Rust's trait system: define what guarantees you need once,
    then ``.wrap()`` generates the enforcement code for each use site.
    Version requirements are validated at construction time (fail fast),
    not at call time.

    Parameters
    ----------
    name : str
        Human-readable name for this contract (appears in error messages).
    require : dict[str, str]
        Package version constraints checked at construction time.
    lock_versions : dict[str, str]
        Locked versions from parse_lock_versions().
    preconditions : dict[str, Callable[..., bool]] | None
        Named checks run on inputs.  Each callable receives the same
        arguments as the target function.  Keys are contract names that
        appear in Verified.contracts.
    postconditions : dict[str, Callable[..., bool]] | None
        Named checks run on the return value.  Each callable receives the
        return value as its sole argument.  Keys are contract names.

    Raises
    ------
    VersionError
        At construction time, if any version requirement is unmet.
    """

    __slots__ = (
        "_name", "_require", "_lock_versions",
        "_preconditions", "_postconditions",
    )

    def __init__(
        self,
        name: str,
        require: dict[str, str],
        lock_versions: dict[str, str],
        preconditions: dict[str, Callable[..., bool]] | None = None,
        postconditions: dict[str, Callable[..., bool]] | None = None,
    ) -> None:
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
        self._name = name
        self._require = require
        self._lock_versions = lock_versions
        self._preconditions = preconditions or {}
        self._postconditions = postconditions or {}

    @property
    def name(self) -> str:
        """The contract spec's human-readable name."""
        return self._name

    @property
    def checked_versions(self) -> dict[str, str]:
        """Locked versions for all required packages."""
        return {pkg: self._lock_versions[pkg] for pkg in self._require}

    def verify(self, value: T, **context: Any) -> Verified[T]:
        """Run preconditions on a value and wrap it as Verified.

        Like a Rust borrow-check boundary: if this returns without
        raising, the data satisfies all declared preconditions.

        Parameters
        ----------
        value : T
            The value to verify.  Passed as the first positional argument
            to each precondition callable.
        **context
            Additional keyword arguments forwarded to each precondition.

        Returns
        -------
        Verified[T]
            Wrapper whose ``.contracts`` lists the preconditions checked.

        Raises
        ------
        ContractError
            If any precondition returns False.
        """
        for check_name, check_fn in self._preconditions.items():
            if not check_fn(value, **context):
                raise ContractError(
                    f"[{self._name}] Precondition {check_name!r} failed"
                )
        return Verified(
            value,
            frozenset(self._preconditions.keys()),
            self.checked_versions,
        )

    def check_output(self, value: T) -> Verified[T]:
        """Run postconditions on a value and wrap it as Verified.

        Parameters
        ----------
        value : T
            The output value to check.

        Returns
        -------
        Verified[T]
            Wrapper whose ``.contracts`` lists the postconditions checked.

        Raises
        ------
        ContractError
            If any postcondition returns False.
        """
        for check_name, check_fn in self._postconditions.items():
            if not check_fn(value):
                raise ContractError(
                    f"[{self._name}] Postcondition {check_name!r} failed"
                )
        return Verified(
            value,
            frozenset(self._postconditions.keys()),
            self.checked_versions,
        )

    def wrap(self, fn: Callable[P, R]) -> Callable[P, "Verified[R]"]:
        """Generate a verified wrapper around a function.

        The returned function:
          1. Runs all preconditions on the input arguments
          2. Calls the original function
          3. Runs all postconditions on the return value
          4. Returns Verified[R] with the full set of checked contracts

        Parameters
        ----------
        fn : Callable[P, R]
            The function to wrap.

        Returns
        -------
        Callable[P, Verified[R]]
            A wrapper that returns verified results.
        """
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Verified[R]:
            for check_name, check_fn in self._preconditions.items():
                if not check_fn(*args, **kwargs):
                    raise ContractError(
                        f"[{self._name}] Precondition {check_name!r} "
                        f"failed for {fn.__qualname__}"
                    )
            result = fn(*args, **kwargs)
            for check_name, check_fn in self._postconditions.items():
                if not check_fn(result):
                    raise ContractError(
                        f"[{self._name}] Postcondition {check_name!r} "
                        f"failed for {fn.__qualname__}"
                    )
            all_checks = frozenset(
                list(self._preconditions.keys())
                + list(self._postconditions.keys())
            )
            return Verified(result, all_checks, self.checked_versions)

        wrapper.__contract_spec__ = self  # type: ignore[attr-defined]
        return wrapper


# ── Table contract predicates ────────────────────────────────────────


def columns_present(table: pa.Table, columns: list[str]) -> bool:
    """Check that every column name in *columns* exists in the Table.

    Parameters
    ----------
    table : pa.Table
        Table to check.
    columns : list[str]
        Expected column names.

    Returns
    -------
    bool
        True if all columns are present.
    """
    return all(c in table.column_names for c in columns)


def no_nulls_in(table: pa.Table, columns: list[str]) -> bool:
    """Check that the specified columns contain no null values.

    Parameters
    ----------
    table : pa.Table
        Table to check.
    columns : list[str]
        Columns to inspect for nulls.

    Returns
    -------
    bool
        True if none of the specified columns contain nulls.
    """
    return all(table.column(c).null_count == 0 for c in columns)


def unique_on(table: pa.Table, columns: list[str]) -> bool:
    """Check that the specified columns form a unique key.

    Parameters
    ----------
    table : pa.Table
        Table to check.
    columns : list[str]
        Columns that should jointly be unique.

    Returns
    -------
    bool
        True if no duplicate rows exist on the given columns.
    """
    sub = table.select(columns)
    grouped = sub.group_by(columns).aggregate([])
    return grouped.num_rows == sub.num_rows


def non_empty(table: pa.Table) -> bool:
    """Check that the Table has at least one row.

    Parameters
    ----------
    table : pa.Table
        Table to check.

    Returns
    -------
    bool
        True if the Table is non-empty.
    """
    return table.num_rows > 0


def row_count_between(table: pa.Table, low: int, high: int) -> bool:
    """Check that the row count falls within [low, high] inclusive.

    Parameters
    ----------
    table : pa.Table
        Table to check.
    low : int
        Minimum acceptable row count.
    high : int
        Maximum acceptable row count.

    Returns
    -------
    bool
        True if low <= num_rows <= high.
    """
    return low <= table.num_rows <= high


def schema_field_is(table: pa.Table, column: str, expected_type: pa.DataType) -> bool:
    """Check that a column's type matches the expected pyarrow DataType.

    Parameters
    ----------
    table : pa.Table
        Table to check.
    column : str
        Column name.
    expected_type : pa.DataType
        Expected pyarrow type (e.g. pa.int64(), pa.string()).

    Returns
    -------
    bool
        True if the column type matches.
    """
    return table.schema.field(column).type == expected_type
