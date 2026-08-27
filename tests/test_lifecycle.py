"""Specs for storage ownership: who is allowed to delete the data.

These cover the two defects that destroy real data in oligo-designer-toolsuite.
Both come from the same gap -- nothing models *ownership* -- so both are fixed by
issue 3.1 rather than by special-casing either symptom:

* ``copy.copy(d)`` hands the copy the same backend object. When the copy is
  collected its ``__del__`` destroys the storage, and the **original** loses its
  data.
* A joblib worker unpickles the store, and the child's ``__del__`` deletes the
  parent's cache mid-run.

No test here uses ``keeps_storage``: the autouse leak detector from issue 0.2
checks ``storage_dir`` is empty at teardown, so it backs every test in this file.
Tests that deliberately assert storage *survives* something clean up explicitly
afterwards, which turns the detector into a second assertion that the explicit
cleanup worked.

``__del__`` is always exercised through real garbage collection -- ``del`` plus
``gc.collect()`` -- never by calling ``__del__`` directly, which would prove
nothing about what the interpreter actually does.
"""

from __future__ import annotations

import copy
import gc
import os
import pickle
import shutil

import pytest

from effidict import EffiDict, LRUReplacement

from .conftest import release_store


def _build_unregistered(backend_cls, policy_cls, storage_dir, name="store", **kwargs):
    """Build a store that the ``make_dict`` registry does not hold a reference to.

    ``make_dict`` keeps every store it creates alive for teardown, which would
    stop ``del`` plus ``gc.collect()`` from ever collecting the object. Any test
    that needs real garbage collection has to build its own.
    """
    backend = backend_cls(str(storage_dir / name))
    effidict = EffiDict(
        disk_backend=backend,
        replacement_strategy=policy_cls(disk_backend=backend, max_in_memory=2),
        **kwargs,
    )
    return effidict, backend.storage_path


def _remove(path):
    """Best-effort cleanup for a store the test owns rather than the fixture."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# copying
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "copy.copy() is not refused, so the copy shares the backend and its "
        "__del__ destroys the original's storage (issue 3.2)"
    ),
)
def test_copy_raises_with_actionable_message(backend_cls, policy_cls, make_dict):
    """Shallow-copying a disk-backed store must be refused, and say what to use.

    Silently sharing the storage is the worse outcome: the copy looks independent
    right up to the point where collecting it deletes the original's data.
    """
    d = make_dict()
    d["k"] = "v"

    with pytest.raises(TypeError) as excinfo:
        copy.copy(d)

    assert "clone" in str(excinfo.value).lower(), (
        f"refusal should point at clone(); got: {excinfo.value}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "copy.deepcopy() either succeeds (Pickle/JSON) or fails with an unrelated "
        "pickling error (SQLite/HDF5) instead of naming clone() (issues 3.2, 2.3)"
    ),
)
def test_deepcopy_raises_with_actionable_message(backend_cls, policy_cls, make_dict):
    """Deep-copying must be refused too, with the same pointer to clone().

    SQLite and HDF5 already raise ``TypeError`` here, but for the wrong reason --
    their handles are unpicklable -- so the message has to be asserted, not just
    the exception type.
    """
    d = make_dict()
    d["k"] = "v"

    with pytest.raises(TypeError) as excinfo:
        copy.deepcopy(d)

    assert "clone" in str(excinfo.value).lower(), (
        f"refusal should point at clone(); got: {excinfo.value}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "copy.copy() is not refused and the copy shares the backend, so "
        "collecting it destroys the original's data (issue 3.2)"
    ),
)
def test_copying_does_not_destroy_the_originals_storage(
    backend_cls, policy_cls, make_dict
):
    """The oligo-designer-toolsuite bug, stated as a behaviour.

    Written so it passes once copying is *refused* as well as once it is made
    safe: if ``copy.copy`` raises, the harm is unreachable and the original is
    intact either way.

    The probe key is placed straight on the backend rather than through
    ``__setitem__``, so reading it back has to reach the disk tier. A cached key
    would still be returned from memory after the storage was deleted, and the
    test would pass while the data was already gone.
    """
    d = make_dict()
    d.disk_backend.serialize("on-disk", "v")
    path = d.disk_backend.storage_path

    try:
        duplicate = copy.copy(d)
    except TypeError:
        pass  # refused outright -- the original cannot be harmed
    else:
        del duplicate
        gc.collect()

    assert os.path.exists(path), "collecting a copy destroyed the original's storage"
    assert d["on-disk"] == "v", "the original can no longer read its own data"


@pytest.mark.xfail(
    strict=True,
    reason="clone() is a contract stub (issue 3.2)",
)
def test_clone_produces_independent_store(backend_cls, policy_cls, make_dict, storage_dir):
    """``clone(new_path)`` must produce a store that shares nothing.

    Independence has to hold in both directions, so the test writes to each side
    and checks the other did not move.
    """
    d = make_dict()
    d["shared"] = "original"

    duplicate = d.clone(str(storage_dir / "clone"))
    try:
        assert duplicate["shared"] == "original"

        duplicate["shared"] = "changed-in-clone"
        duplicate["clone-only"] = "x"
        d["origin-only"] = "y"

        assert d["shared"] == "original", "clone wrote through to the original"
        assert "clone-only" not in d
        assert "origin-only" not in duplicate
    finally:
        release_store(duplicate)


# --------------------------------------------------------------------------
# garbage collection
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "__del__ calls close(), which calls destroy(), so collecting a store "
        "deletes its data (issue 3.1)"
    ),
)
def test_del_never_destroys_storage(backend_cls, policy_cls, storage_dir):
    """Collecting a store must release handles, never delete data.

    Uses a store the ``make_dict`` registry does not reference, so ``del`` plus
    ``gc.collect()`` genuinely collects it.
    """
    d, path = _build_unregistered(backend_cls, policy_cls, storage_dir)
    d["k"] = "v"
    assert os.path.exists(path), "precondition: storage was created"

    del d
    gc.collect()

    try:
        assert os.path.exists(path), "garbage collection deleted the storage"
    finally:
        _remove(path)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SQLite and HDF5 cannot be pickled at all, and an unpickled Pickle/JSON "
        "store owns the storage it did not create (issues 2.3, 3.1)"
    ),
)
def test_unpickled_instance_does_not_own_storage(backend_cls, policy_cls, make_dict):
    """A store that arrived over a process boundary must not delete the data.

    This is the joblib failure: the worker's copy is collected when the task ends
    and takes the parent's cache with it.
    """
    d = make_dict()
    d["k"] = "v"
    path = d.disk_backend.storage_path

    child = pickle.loads(pickle.dumps(d))
    assert child["k"] == "v"

    del child
    gc.collect()

    assert os.path.exists(path), "collecting an unpickled store deleted the storage"
    assert d["k"] == "v"


# --------------------------------------------------------------------------
# explicit lifecycle
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EffiDict has no destroy(), and calling close() twice raises "
        "FileNotFoundError (issue 3.1)"
    ),
)
def test_destroy_is_explicit_and_idempotent(backend_cls, policy_cls, storage_dir):
    """Deleting the data is an explicit call, and a second call is a no-op.

    Today a second release raises ``FileNotFoundError`` on all 28 combinations,
    which makes defensive cleanup in a ``finally`` block impossible to write.
    """
    d, path = _build_unregistered(backend_cls, policy_cls, storage_dir)
    d["k"] = "v"

    try:
        assert hasattr(d, "destroy"), "EffiDict has no destroy()"
        d.destroy()
        assert not os.path.exists(path), "destroy() left the storage behind"
        d.destroy()  # must not raise
    finally:
        _remove(path)


@pytest.mark.xfail(
    strict=True,
    reason="__exit__ calls close(), which destroys the storage (issue 3.1)",
)
def test_context_manager_closes_but_does_not_destroy(
    backend_cls, policy_cls, storage_dir
):
    """Leaving a ``with`` block releases handles and keeps the data."""
    d, path = _build_unregistered(backend_cls, policy_cls, storage_dir)

    with d:
        d["k"] = "v"

    try:
        assert os.path.exists(path), "leaving the with-block destroyed the storage"
    finally:
        _remove(path)


@pytest.mark.xfail(
    strict=True,
    reason="temporary() is a contract stub (issues 2.1, 3.1)",
)
def test_temporary_store_is_destroyed_on_close(backend_cls, policy_cls):
    """A scratch store is the one case that *should* clean itself up on close.

    Destroy-on-close becomes opt-in rather than the default, so ``temporary()``
    has to keep today's convenience for callers who genuinely want it.

    Cleans up in a ``finally`` because a temporary store lives outside
    ``storage_dir`` -- under the system temp directory -- where the autouse leak
    detector cannot see it. If close() ever stops destroying, this test would
    otherwise leave files on the machine every run.
    """
    backend = backend_cls.temporary()
    path = backend.storage_path
    try:
        d = EffiDict(
            disk_backend=backend,
            replacement_strategy=policy_cls(disk_backend=backend, max_in_memory=2),
        )
        d["k"] = "v"

        d.close()

        assert not os.path.exists(path), "a temporary store outlived close()"
    finally:
        _remove(path)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "close() destroys the storage instead of writing the cache out; nothing "
        "flushes (issues 1.4, 3.1)"
    ),
)
def test_dirty_entries_are_flushed_on_close(
    backend_cls, policy_cls, make_dict, backend_spy
):
    """close() must write out whatever the cache still holds.

    Asserted with the call spy rather than by reopening, so this stays a statement
    about ``close()`` alone -- issue 0.5 owns the reopen-and-read version.
    """
    d = make_dict(max_in_memory=100)
    for i in range(5):
        d[f"k{i}"] = f"v{i}"
    assert d.disk_backend.keys() == [], "precondition: nothing spilled yet"

    with backend_spy(d.disk_backend) as spy:
        d.close()

        written = spy.count("serialize") + spy.count("write_many")
        assert written > 0, "close() persisted nothing before releasing the store"
