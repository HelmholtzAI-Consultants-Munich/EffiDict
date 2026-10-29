"""Scaffolding shared by every Phase 0 spec: the value catalogue, the
comparison helper, and the conventions two or more spec files have to agree on.

The per-backend tables below were derived by round-tripping each value through
each backend, not by reading the source. Anything asserted here is observed
behaviour on ``main``; ``test_harness.py`` re-derives the tables on every run so
they cannot drift.
"""

from __future__ import annotations

import inspect
import math
import os

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a dev dependency
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a dev dependency
    pd = None

from effidict import (
    Hdf5Backend,
    JSONBackend,
    PickleBackend,
    SqliteBackend,
)

#: Policies whose ``get`` writes a disk value back into the cache. Only these
#: can mutate the cache during a read, so only these violate I6 -- several
#: specs need to scope their expectations to this set rather than assuming the
#: defect is universal.
#:
#: Empty since issue 1.2. Promotion used to be per-strategy: four of the seven
#: wrote the loaded value back, three read straight through. ``Store.get`` now
#: answers from one path for every policy and does not promote, because
#: promoting without the clean/dirty tag means evicting, and evicting means
#: writing a value the disk already holds. Issue 1.3 may reintroduce promotion
#: once that tag exists, and this set is where it would be recorded.
PROMOTING_POLICIES = set()


def _ndarray():
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for the 'ndarray' value kind")
    return np.ones(4)


def _dataframe():
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for the 'dataframe' value kind")
    return pd.DataFrame({"a": [1, 2]})


#: Every value shape the specs exercise. Includes the shapes that broke in
#: production: ``None``, ``{}``, nested dicts and DataFrames.
VALUE_KINDS = {
    "int": lambda: 7,
    "float": lambda: 1.5,
    "bool": lambda: True,
    "str": lambda: "hello",
    "none": lambda: None,
    "empty_dict": lambda: {},
    "dict": lambda: {"a": 1},
    "nested_dict": lambda: {"a": {"b": [1, 2]}},
    "list": lambda: [1, 2, 3],
    "tuple": lambda: (1, 2),
    "ndarray": _ndarray,
    "dataframe": _dataframe,
}

#: Kinds a backend refuses outright. Writing one of these raises.
UNSUPPORTED = {
    SqliteBackend: {"ndarray", "dataframe"},
    JSONBackend: {"ndarray", "dataframe"},
    PickleBackend: set(),
    Hdf5Backend: set(),
}

#: Kinds a backend accepts but does not round-trip faithfully. These are
#: DEFECTS, deliberately kept separate from UNSUPPORTED: folding them together
#: would let the specs skip them, and a silently-wrong round-trip is exactly
#: what issue 0.8 (`test_value_roundtrip_fidelity`) has to assert on.
#:
#:   * SQLite/JSON encode via JSON, so a tuple returns as a list.
#:   * HDF5 widens Python scalars to numpy scalars (int -> int64, bool -> bool_).
#:   * HDF5 converts a numeric DataFrame to an array, so the write succeeds and
#:     the value comes back as an ndarray.
LOSSY = {
    SqliteBackend: {"tuple"},
    JSONBackend: {"tuple"},
    PickleBackend: set(),
    Hdf5Backend: {"int", "float", "bool", "dataframe"},
}


def supported_kinds(backend_cls):
    """Kinds that round-trip faithfully through ``backend_cls``."""
    return set(VALUE_KINDS) - UNSUPPORTED[backend_cls] - LOSSY[backend_cls]


def is_unsupported(backend_cls, kind):
    return kind in UNSUPPORTED[backend_cls]


def is_lossy(backend_cls, kind):
    return kind in LOSSY[backend_cls]


#: Iterator types that prove the sequence behind them was already built. An
#: iterator over a container is a perfectly good iterator and calls no backend
#: method, so neither "is it an iterator" nor a call spy can see it -- but it
#: holds every key in memory, which is what the streaming contract exists to
#: avoid. Shared by the backend-level and facade-level streaming specs so one
#: definition covers both.
EAGER_ITERATOR_TYPES = frozenset(
    {
        "list_iterator",
        "tuple_iterator",
        "set_iterator",
        "dict_keyiterator",
        "dict_itemiterator",
        "dict_valueiterator",
        "range_iterator",
        # Both spellings: CPython returns str_ascii_iterator for an ASCII
        # string and str_iterator only for a non-ASCII one, so listing just
        # the latter leaves the common case unmatched -- and this set is
        # used in a `not in` assertion, which passes silently when it misses.
        "str_ascii_iterator",
        "str_iterator",
    }
)


def in_cache(effidict, key):
    """Whether ``key`` currently sits in the in-memory tier.

    ``Store.in_cache`` deliberately takes no lock, which is what lets the I5
    specs observe tier residency while an eviction is held open.
    """
    return effidict._store.in_cache(key)


def cached_keys(effidict):
    """The set of keys currently resident in the cache.

    Companion to :func:`in_cache` for the specs that need the whole set rather
    than one membership test -- "did this operation change what is cached?".
    Issue 1.2 replaces the body with ``set(effidict._store.cache)``.

    Behind a helper for the same reason as :func:`in_cache`, and the reason is
    sharper than tidiness: a spec that read ``replacement_strategy.memory``
    directly raised ``AttributeError`` once 1.2 removed that alias, and a strict
    xfail cannot tell "still broken" from "broken differently". Those guards
    would have stayed red on a correct implementation instead of flipping.
    """
    return set(effidict._store.cache)


def cached_bytes(effidict):
    """Approximate number of bytes the cache is holding.

    Issue 1.4 gives the cache a real byte budget and ``Cache.nbytes()`` becomes
    the contract; this becomes ``effidict._store.cache.nbytes()`` then. Until
    that lands, summing ``len`` over the resident values is the only measure
    available, which is why the budget specs assert with generous headroom
    rather than an exact figure.

    Read through ``peek``, not ``get``: measuring the cache must not reorder it.
    """
    cache = effidict._store.cache
    total = 0
    for key in cache:
        total += len(cache.peek(key))
    return total


def on_disk(effidict, key):
    """Whether ``key`` currently sits in the persistent tier.

    ``has()`` rather than ``keys()``: issue 1.2 gave it a naive implementation,
    and issue 2.2 makes it cheap without changing the answer.
    """
    return effidict._store.backend.has(key)


def assert_equal_value(actual, expected):
    """Assert two stored values are equal, handling arrays and DataFrames.

    Type-strict by design: ``1`` is not ``True`` and ``numpy.int64(7)`` is not
    ``7``. Plain ``==`` conflates all three, which would hide the HDF5 scalar
    widening and the SQLite tuple-to-list conversion recorded in ``LOSSY``.
    """
    if not _equal(actual, expected):
        raise AssertionError(
            f"values differ:\n  actual   {type(actual).__name__}: {actual!r}"
            f"\n  expected {type(expected).__name__}: {expected!r}"
        )


def _equal(actual, expected):
    if _is_nan(actual) and _is_nan(expected):
        # Type first, exactly as the non-NaN path does further down. ``NaN !=
        # NaN`` is why this shortcut exists at all, but returning True outright
        # skipped the type check that every other scalar gets: ``np.float64`` is
        # a *subclass* of ``float``, so ``_is_nan`` says yes to both and a
        # backend returning ``np.float64('nan')`` for a stored ``float('nan')``
        # read as faithful. The same round-trip with 1.5 was already rejected,
        # so only NaN slipped through.
        return type(actual) is type(expected)

    if pd is not None:
        frame_types = (pd.DataFrame, pd.Series)
        if isinstance(actual, frame_types) or isinstance(expected, frame_types):
            if type(actual) is not type(expected):
                return False
            return actual.equals(expected)

    if np is not None:
        if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
            if not (isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)):
                return False
            # dtype before values. ``np.array_equal`` compares element values
            # only, so an array stored as float64 and returned as float32 --
            # or as int32, or widened to float64 -- compares equal and the
            # round-trip reads as faithful. That is the same silent widening
            # the scalar path already rejects below via ``type(actual) is not
            # type(expected)``; arrays were the gap.
            if actual.dtype != expected.dtype:
                return False
            # equal_nan calls isnan(), which raises TypeError on string and
            # object dtypes, so only ask for it where NaN can exist.
            if np.issubdtype(actual.dtype, np.inexact) and np.issubdtype(
                expected.dtype, np.inexact
            ):
                return np.array_equal(actual, expected, equal_nan=True)
            return np.array_equal(actual, expected)

    if type(actual) is not type(expected):
        return False

    if isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        return all(_equal(actual[key], expected[key]) for key in expected)

    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(_equal(a, e) for a, e in zip(actual, expected))

    return actual == expected


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def store_bytes(path):
    """Bytes a backend occupies on disk, whether it stores one file or a tree."""
    if not os.path.isdir(path):
        return os.path.getsize(path)

    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            total += os.path.getsize(os.path.join(dirpath, filename))
    return total


# --------------------------------------------------------------------------
# holding an eviction open
# --------------------------------------------------------------------------

#: How long a probe keeps a write blocked inside ``serialize``, and how long it
#: waits for anything to happen. HOLD must comfortably exceed the *sum* of the
#: waits performed while the write is held, or the hold expires mid-inspection:
#: the write completes, and the tier state the probe then examines is
#: post-write, where a drop-then-write implementation is indistinguishable from
#: a compliant one. Both numbers are liveness bounds, never discriminators --
#: nothing here decides what happened by how long it took. The ``finally`` in
#: each probe releases the hold, so HOLD is only ever reached if the test itself
#: wedged.
HOLD_SECONDS = 120
WAIT_SECONDS = 10


def xfail_sqlite_held_write(request, backend_cls):
    """Mark the held-open-write probes xfail on SQLite, for a reason of its own.

    Both probes start an evicting write and block it inside ``serialize``. On
    SQLite that write raises the thread-affinity error instead (issue 2.3), so
    it never completes and neither probe observes the thing it is about. They
    assert the writer's outcome rather than swallowing it, which is what makes
    this a marker and not a skip: once issue 2.3 lands, the strict xfail turns
    the silence back into a failure.
    """
    if backend_cls is SqliteBackend:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "the evicting write raises the SQLite thread-affinity error, "
                    "so the held-open write never completes (issue 2.3)"
                ),
            )
        )
