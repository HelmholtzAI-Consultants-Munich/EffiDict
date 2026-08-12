"""Tests for the harness itself.

The harness is load-bearing for every other Phase 0 spec, so it is verified
rather than assumed. In particular the leak detector is proved to fail on a
deliberate leak -- an autouse fixture that never fires is worse than no fixture.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from .conftest import BACKENDS, POLICIES
from .helpers import (
    UNSUPPORTED,
    VALUE_KINDS,
    assert_equal_value,
    is_lossy,
    is_unsupported,
    supported_kinds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _install_harness(pytester):
    """Point a pytester sandbox at this repo's conftest."""
    pytester.makeconftest(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
                'pytest_plugins = ["tests.conftest"]',
            ]
        )
    )


# --------------------------------------------------------------------------
# matrix
# --------------------------------------------------------------------------


def test_matrix_collects_28_combinations(pytester):
    """Exercise real pytest collection.

    Asserting ``len(BACKENDS) * len(POLICIES) == 28`` in Python would pass even
    if the backend_cls/policy_cls fixtures were broken, which is the thing under
    test. So run a sub-pytest and count what actually ran.
    """
    _install_harness(pytester)
    pytester.makepyfile(
        """
        def test_combo(backend_cls, policy_cls):
            assert backend_cls is not None and policy_cls is not None
        """
    )

    result = pytester.runpytest("-q")

    outcomes = result.parseoutcomes()
    # h5py-conditional params skip rather than vanish, so the total is fixed.
    assert outcomes.get("passed", 0) + outcomes.get("skipped", 0) == 28
    assert outcomes.get("failed", 0) == 0
    assert outcomes.get("errors", 0) == 0


def test_matrix_ids_are_readable_and_unique(pytester):
    _install_harness(pytester)
    pytester.makepyfile(
        """
        def test_combo(backend_cls, policy_cls):
            pass
        """
    )

    result = pytester.runpytest("--collect-only", "-q")

    lines = [line for line in result.outlines if "test_combo[" in line]
    assert len(lines) == 28
    assert len(set(lines)) == 28
    assert any("[SqliteBackend-LRUReplacement]" in line for line in lines)


# --------------------------------------------------------------------------
# make_dict
# --------------------------------------------------------------------------


def test_make_dict_roundtrips_across_the_matrix(backend_cls, policy_cls, make_dict):
    d = make_dict()

    d["a"] = {"nested": [1, 2]}

    assert_equal_value(d["a"], {"nested": [1, 2]})


def test_make_dict_honours_max_in_memory(make_dict):
    d = make_dict(max_in_memory=2)

    for key in ("a", "b", "c"):
        d[key] = key

    assert len(d.replacement_strategy.memory) == 2
    assert len(d.keys()) == 3


def test_make_dict_forwards_max_bytes(make_dict):
    """A dropped max_bytes would let issue 0.4's budget spec test nothing."""
    d = make_dict(max_bytes=4096)

    assert d.max_bytes == 4096


def test_make_dict_defaults_when_unparametrized(make_dict):
    from .conftest import DEFAULT_BACKEND, DEFAULT_POLICY

    d = make_dict()

    assert isinstance(d.disk_backend, DEFAULT_BACKEND)
    assert isinstance(d.replacement_strategy, DEFAULT_POLICY)


def test_make_dict_gives_each_dict_its_own_storage(make_dict):
    first = make_dict()
    second = make_dict()

    assert first.disk_backend.storage_path != second.disk_backend.storage_path


# --------------------------------------------------------------------------
# leak detector
# --------------------------------------------------------------------------


def test_leak_detector_fires_on_a_leaked_file(pytester):
    """The acceptance criterion: prove the detector actually fails."""
    _install_harness(pytester)
    pytester.makepyfile(
        """
        def test_leaks(storage_dir):
            (storage_dir / "leaked.bin").write_bytes(b"x")
        """
    )

    result = pytester.runpytest("-q")

    assert result.ret != 0
    result.stdout.fnmatch_lines(["*storage left behind*leaked.bin*"])


def test_leak_detector_passes_when_storage_is_released(pytester):
    _install_harness(pytester)
    pytester.makepyfile(
        """
        def test_clean(make_dict):
            d = make_dict()
            d["a"] = 1
        """
    )

    result = pytester.runpytest("-q")

    result.assert_outcomes(passed=1)


def test_keeps_storage_marker_opts_out(pytester):
    """Issue 0.5 needs storage to outlive a close, so the opt-out must work."""
    _install_harness(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.keeps_storage
        def test_deliberately_keeps(storage_dir):
            (storage_dir / "kept.bin").write_bytes(b"x")
        """
    )

    result = pytester.runpytest("-q")

    result.assert_outcomes(passed=1)


# --------------------------------------------------------------------------
# assert_equal_value
# --------------------------------------------------------------------------


def test_assert_equal_value_handles_arrays_frames_and_nan():
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")

    left = {"nan": float("nan"), "nested": [VALUE_KINDS["ndarray"]()]}
    right = {"nan": float("nan"), "nested": [VALUE_KINDS["ndarray"]()]}

    assert_equal_value(left, right)
    assert_equal_value(VALUE_KINDS["dataframe"](), VALUE_KINDS["dataframe"]())


def test_assert_equal_value_handles_string_arrays():
    """equal_nan=True raises TypeError on non-numeric dtypes if not guarded."""
    np = pytest.importorskip("numpy")

    assert_equal_value(np.array(["a", "b"]), np.array(["a", "b"]))
    with pytest.raises(AssertionError):
        assert_equal_value(np.array(["a", "b"]), np.array(["a", "c"]))


@pytest.mark.parametrize(
    "left, right",
    [
        ({"a": [1, 2, 3]}, {"a": [1, 2, 4]}),
        (1, True),  # plain == conflates these
        (1, 1.0),
        ((1, 2), [1, 2]),  # the SQLite/JSON tuple defect
        ({"a": 1}, {"a": 1, "b": 2}),
    ],
)
def test_assert_equal_value_is_type_strict(left, right):
    with pytest.raises(AssertionError):
        assert_equal_value(left, right)


def test_assert_equal_value_rejects_numpy_scalar_widening():
    """The HDF5 int -> int64 defect must not compare equal."""
    np = pytest.importorskip("numpy")

    with pytest.raises(AssertionError):
        assert_equal_value(np.int64(7), 7)


# --------------------------------------------------------------------------
# value support tables
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(VALUE_KINDS))
def test_value_support_tables_match_reality(backend_cls, kind, tmp_path):
    """Re-derive UNSUPPORTED/LOSSY every run so the tables cannot go stale."""
    if kind == "ndarray":
        pytest.importorskip("numpy")
    if kind == "dataframe":
        pytest.importorskip("pandas")

    backend = backend_cls(str(tmp_path / backend_cls.__name__))
    value = VALUE_KINDS[kind]()

    try:
        if is_unsupported(backend_cls, kind):
            with pytest.raises(TypeError):
                backend.serialize("k", value)
            return

        backend.serialize("k", value)
        actual = backend.deserialize("k")

        if is_lossy(backend_cls, kind):
            with pytest.raises(AssertionError):
                assert_equal_value(actual, value)
        else:
            assert_equal_value(actual, value)
    finally:
        with contextlib.suppress(Exception):
            backend.destroy()


def test_support_tables_partition_every_kind():
    for backend_cls in BACKENDS:
        buckets = (
            supported_kinds(backend_cls),
            UNSUPPORTED[backend_cls],
            set(k for k in VALUE_KINDS if is_lossy(backend_cls, k)),
        )
        union = set().union(*buckets)
        assert union == set(VALUE_KINDS), backend_cls.__name__
        assert sum(len(b) for b in buckets) == len(VALUE_KINDS), backend_cls.__name__


# --------------------------------------------------------------------------
# backend spy
# --------------------------------------------------------------------------


def test_backend_spy_records_and_delegates(make_dict, backend_spy):
    d = make_dict(max_in_memory=1)
    d["a"] = 1
    d["b"] = 2  # forces 'a' out to disk

    with backend_spy(d.disk_backend) as spy:
        assert d["a"] == 1
        assert spy.count("deserialize") == 1
        assert spy.count("del_item") == 0
        assert spy.calls[0][0] == "deserialize"

        spy.reset()
        assert spy.calls == []
        assert spy.count("deserialize") == 0


def test_backend_spy_restores_the_backend(make_dict, backend_spy):
    d = make_dict()
    original = type(d.disk_backend).serialize

    with backend_spy(d.disk_backend):
        assert d.disk_backend.serialize.__name__ == "recorder"

    assert d.disk_backend.serialize.__func__ is original
    assert "serialize" not in vars(d.disk_backend)


def test_backend_spy_leaves_classmethods_alone(make_dict, backend_spy):
    d = make_dict()

    with backend_spy(d.disk_backend) as spy:
        assert "create" not in spy._patched
        assert "open" not in spy._patched


def test_backend_spy_counts_writes_during_eviction(make_dict, backend_spy):
    d = make_dict(max_in_memory=1)

    with backend_spy(d.disk_backend) as spy:
        d["a"] = 1
        d["b"] = 2
        assert spy.count("serialize") == 1
