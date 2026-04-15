"""
Dependency pinning: read a uv.lock file and bind your code to locked versions.

The lock file is a source of truth about your project's dependency versions.
This pattern makes that truth available to your code so version upgrades that
violate your assumptions fail at import time, not at 3am in production.

    from pathlib import Path
    from cleanDE.dep_pin.impl import pin, requires, resolve

    VERSIONS = pin(Path("uv.lock"), {"pandas": ">=3.0", "pyarrow": ">=14.0"})

    @requires(pandas=">=3.0")
    def normalize_arrow(df): ...

    @requires(pandas=">=2.0,<3.0")
    def normalize_numpy(df): ...

    normalize = resolve(VERSIONS, normalize_arrow, normalize_numpy)

Six functions:
  parse_lock  — pure: lock text in, {package: version_tuple} out
  check       — pure: versions + requirements in, raise or nothing
  pin         — convenience: file path + requirements in, versions out
  select      — pure: versions + branches in, first matching value out
  requires    — decorator: tag a function with version requirements
  resolve     — pure: versions + tagged candidates in, winner out
"""

import tomllib
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class DependencyError(Exception):
    """Raised when a locked version does not satisfy a declared requirement.

    Attributes
    ----------
    package : str
        The package that failed.
    locked : tuple[int, ...]
        The version found in the lock file (empty tuple if missing).
    constraint : str
        The constraint that was violated.
    """

    def __init__(self, package: str, locked: tuple[int, ...], constraint: str) -> None:
        self.package = package
        self.locked = locked
        self.constraint = constraint
        locked_str = ".".join(map(str, locked)) if locked else "<missing>"
        super().__init__(f"{package} locked at {locked_str}, requires {constraint}")


def parse_lock(lock_text: str) -> dict[str, tuple[int, ...]]:
    """Parse uv.lock text into a mapping of package names to version tuples.

    Pure function — no I/O.  The caller reads the file.

    Parameters
    ----------
    lock_text : str
        Contents of a uv.lock file.

    Returns
    -------
    dict[str, tuple[int, ...]]
        {package_name: (major, minor, patch, ...)} for every package
        that declares a version.
    """
    data = tomllib.loads(lock_text)
    return {
        pkg["name"]: _parse_version(pkg["version"])
        for pkg in data.get("package", [])
        if "version" in pkg
    }


def check(
    versions: dict[str, tuple[int, ...]],
    require: dict[str, str],
) -> None:
    """Raise DependencyError if any locked version violates a requirement.

    Pure function — no I/O, no side effects beyond raising.

    Parameters
    ----------
    versions : dict[str, tuple[int, ...]]
        Locked versions from parse_lock().
    require : dict[str, str]
        {package: constraint}, e.g. {"pandas": ">=3.0.0,<4.0"}.

    Raises
    ------
    DependencyError
        On the first package whose locked version violates its constraint,
        or that is missing from the versions dict.
    """
    for pkg, constraint in require.items():
        locked = versions.get(pkg)
        if locked is None:
            raise DependencyError(pkg, (), constraint)
        if not _satisfies(locked, constraint):
            raise DependencyError(pkg, locked, constraint)


def pin(
    lock_path: Path,
    require: dict[str, str],
) -> dict[str, tuple[int, ...]]:
    """Read a uv.lock file, verify requirements, return locked versions.

    Convenience that combines file I/O + parse_lock() + check().
    Intended for module-level use so violations surface at import time:

        VERSIONS = pin(Path("uv.lock"), {"pandas": ">=3.0"})

    Parameters
    ----------
    lock_path : Path
        Path to the uv.lock file.
    require : dict[str, str]
        {package: constraint}.

    Returns
    -------
    dict[str, tuple[int, ...]]
        All locked versions (not just the required ones).

    Raises
    ------
    FileNotFoundError
        If the lock file does not exist.
    DependencyError
        If any requirement is unmet.
    """
    versions = parse_lock(lock_path.read_text())
    check(versions, require)
    return versions


def select(
    versions: dict[str, tuple[int, ...]],
    package: str,
    branches: dict[str, T],
    default: T | None = None,
) -> T:
    """Pick a value based on the locked version of a package.

    Iterates *branches* in insertion order and returns the value for the
    first constraint that the locked version satisfies.  Use this to select
    implementations at module load time:

        normalize = select(VERSIONS, "pandas", {
            ">=3.0": normalize_arrow_dtypes,
            ">=2.0,<3.0": normalize_numpy_dtypes,
        })

    Parameters
    ----------
    versions : dict[str, tuple[int, ...]]
        Locked versions from parse_lock() or pin().
    package : str
        Package name to branch on.
    branches : dict[str, T]
        {constraint: value}.  First matching constraint wins.
    default : T | None
        Returned when no branch matches.  If None and no match, raises.

    Returns
    -------
    T
        Value from the first matching branch, or *default*.

    Raises
    ------
    DependencyError
        If the package is not in *versions* and no *default* is given.
    ValueError
        If no branch matches and no *default* is given.
    """
    locked = versions.get(package)
    if locked is None:
        if default is not None:
            return default
        raise DependencyError(package, (), f"one of {list(branches.keys())}")

    for constraint, value in branches.items():
        if _satisfies(locked, constraint):
            return value

    if default is not None:
        return default
    raise ValueError(
        f"No branch matches {package}=="
        f"{'.'.join(map(str, locked))}; "
        f"branches: {list(branches.keys())}"
    )


# ── Version-tagged functions ─────────────────────────────────────────


def requires(**deps: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Tag a function with the dependency versions it needs.

    Pure annotation — does not change the function's behavior.  Attaches
    a ``__requires__`` dict for use with resolve().

        @requires(pandas=">=3.0", pyarrow=">=14.0")
        def transform(df): ...

        transform.__requires__  # {"pandas": ">=3.0", "pyarrow": ">=14.0"}

    Parameters
    ----------
    **deps : str
        Package names to version constraints.

    Returns
    -------
    Callable
        The original function, unchanged, with ``__requires__`` attached.
    """
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        fn.__requires__ = deps  # type: ignore[attr-defined]
        return fn
    return decorator


def resolve(
    versions: dict[str, tuple[int, ...]],
    *candidates: Callable[P, R],
) -> Callable[P, R]:
    """Pick the first candidate whose @requires are all satisfied.

    Unlike select(), which branches on a single package, resolve() checks
    every requirement on each candidate — so multi-package constraints
    work naturally.

        @requires(pandas=">=3.0")
        def impl_v3(df): ...

        @requires(pandas=">=2.0,<3.0")
        def impl_v2(df): ...

        impl = resolve(VERSIONS, impl_v3, impl_v2)

    Parameters
    ----------
    versions : dict[str, tuple[int, ...]]
        Locked versions from parse_lock() or pin().
    *candidates : Callable
        Functions tagged with @requires.  First match wins.

    Returns
    -------
    Callable
        The first candidate whose requirements are all met.

    Raises
    ------
    ValueError
        If no candidate matches.
    """
    for fn in candidates:
        reqs: dict[str, str] = getattr(fn, "__requires__", {})
        if all(
            pkg in versions and _satisfies(versions[pkg], constraint)
            for pkg, constraint in reqs.items()
        ):
            return fn
    names = [getattr(fn, "__name__", repr(fn)) for fn in candidates]
    raise ValueError(f"No candidate matches locked versions: {names}")


# ── Internal ─────────────────────────────────────────────────────────


def _parse_version(v: str) -> tuple[int, ...]:
    """Split "3.0.2" or "3.0.0rc1" into (3, 0, 2) or (3, 0, 0)."""
    parts: list[int] = []
    for segment in v.split("."):
        digits = ""
        for ch in segment:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _satisfies(version: tuple[int, ...], constraint: str) -> bool:
    """Check a version tuple against a comma-separated constraint string."""
    for part in constraint.split(","):
        part = part.strip()
        if part.startswith(">="):
            if version < _parse_version(part[2:]):
                return False
        elif part.startswith("<="):
            if version > _parse_version(part[2:]):
                return False
        elif part.startswith("!="):
            if version == _parse_version(part[2:]):
                return False
        elif part.startswith(">"):
            if version <= _parse_version(part[1:]):
                return False
        elif part.startswith("<"):
            if version >= _parse_version(part[1:]):
                return False
        elif part.startswith("=="):
            if version != _parse_version(part[2:]):
                return False
        else:
            raise ValueError(f"Unsupported constraint operator: {part!r}")
    return True
