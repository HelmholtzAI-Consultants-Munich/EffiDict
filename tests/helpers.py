"""Value catalogue and comparison helper shared by every Phase 0 spec.

The per-backend tables below were derived by round-tripping each value through
each backend, not by reading the source. Anything asserted here is observed
behaviour on ``main``; ``test_harness.py`` re-derives the tables on every run so
they cannot drift.
"""

from __future__ import annotations

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a dev dependency
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a dev dependency
    pd = None

from effidict import Hdf5Backend, JSONBackend, PickleBackend, SqliteBackend


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
        return True

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
