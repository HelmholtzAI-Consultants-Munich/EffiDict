"""Specs for the core promise: data written is data recoverable.

Two groups. The first needs no new API and fails on behaviour that exists today --
``DiskBackend.__init__`` derives its location from ``time.time()`` and ``id(self)``,
so a store cannot be found again and two logically separate stores routinely land
on the same path. The second is written against the ``create``/``open`` lifecycle
from issue 2.1, which is still a contract stub, so those specs fail on
``NotImplementedError`` until it lands.

Everything here is parametrized over all four backends, since persistence is a
backend property.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from effidict import EffiDict, LRUReplacement

from .conftest import release_store

REPO_ROOT = str(Path(__file__).resolve().parents[1])

KEYS = {f"k{i}": f"v{i}" for i in range(5)}


def _build(backend, policy_cls=LRUReplacement, max_in_memory=2):
    """Wrap an already-constructed backend in an EffiDict."""
    return EffiDict(
        disk_backend=backend,
        replacement_strategy=policy_cls(
            disk_backend=backend, max_in_memory=max_in_memory
        ),
    )


# --------------------------------------------------------------------------
# addressability -- can a store be found again at all?
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "storage_path appends time.time() and id(self), so the same argument "
        "resolves to a different location every time (issue 2.1)"
    ),
)
def test_storage_path_is_stable_across_instances(backend_cls, storage_dir):
    """The same path argument must always name the same store.

    Without this nothing else in this file is reachable: a store you cannot
    address twice cannot be reopened, and 'persistent' means nothing.
    """
    path = str(storage_dir / "store")

    first = backend_cls(path)
    second = backend_cls(path)
    try:
        assert first.storage_path == second.storage_path, (
            f"same argument produced two locations:\n"
            f"  {first.storage_path}\n  {second.storage_path}"
        )
    finally:
        for backend in (first, second):
            release_store(backend)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "storage_path embeds id(self), which is only unique among live objects "
        "and is recycled after GC (issue 2.1)"
    ),
)
def test_id_reuse_cannot_collide(backend_cls, storage_dir):
    """A store's location must not depend on a recyclable process-local address.

    Asserted structurally -- the path must not embed ``id(self)`` -- rather than by
    provoking the reuse. Whether a freed address is handed straight back is
    allocator-dependent: measured at 197 collisions in 200 cycles standalone, but
    only about one run in three inside a test process. A probabilistic assertion
    under ``strict=True`` would turn a lucky run into an XPASS and redden CI while
    the defect was still present, so the mechanism is asserted instead of its
    symptom.

    What the symptom costs, measured directly: a store abandoned without
    ``destroy()`` stays on disk, and the next store built from the same argument
    lands on its path. Pickle, SQLite and JSON then read the previous store's
    values; HDF5 opens ``mode="w"`` and truncates it instead, destroying the data
    rather than leaking it. Same root cause, two different failures.
    """
    backend = backend_cls(str(storage_dir / "cache"))
    try:
        assert str(id(backend)) not in backend.storage_path, (
            f"storage path embeds id(self)={id(backend)}, so a later store can "
            f"reuse it verbatim: {backend.storage_path}"
        )
    finally:
        release_store(backend)


# --------------------------------------------------------------------------
# reopening -- written against the issue 2.1 lifecycle API
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "create()/open() and flush() are contract stubs, and close() destroys the storage rather than persisting it (issues 2.1, 1.4, 3.1)"
    ),
)
def test_reopen_after_close_returns_all_values(backend_cls, policy_cls, storage_dir):
    """Write, close, reopen, read everything back.

    The single most important spec in this file: it is the difference between a
    cache and a persistent store.
    """
    path = str(storage_dir / "store")

    writer = _build(backend_cls.create(path), policy_cls)
    for key, value in KEYS.items():
        writer[key] = value
    writer.flush()
    writer.close()

    reader = _build(backend_cls.open(path), policy_cls)
    try:
        for key, value in KEYS.items():
            assert reader[key] == value, f"{key} did not survive the reopen"
        assert set(reader.keys()) == set(KEYS)
    finally:
        release_store(reader)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "open() is a contract stub; Hdf5Backend also opens mode='w', which truncates on construction (issue 2.1)"
    ),
)
def test_open_does_not_truncate_existing_store(backend_cls, storage_dir):
    """Opening an existing store must not empty it.

    ``Hdf5Backend.__init__`` passes ``mode="w"`` to ``h5py.File``, which truncates
    on every construction -- so today the data is destroyed by the act of opening.
    """
    path = str(storage_dir / "store")

    first = backend_cls.create(path)
    first.serialize("kept", "value")
    # Deliberately not closed: Backend.close() does not exist, not even as a
    # stub, and truncation happens on construction -- a second handle is enough
    # to show it. Uses keys() rather than has(), which is a separate stub owned
    # by issue 2.2; depending on it would keep this spec red after its own fix.
    second = backend_cls.open(path)
    try:
        assert "kept" in second.keys(), "opening the store truncated it"
        assert second.deserialize("kept") == "value"
    finally:
        for backend in (second, first):
            release_store(backend)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "create()/open() are contract stubs, nothing flushes on close, and close() destroys the storage (issues 2.1, 1.4, 3.1)"
    ),
)
def test_unflushed_writes_are_not_lost_on_close(backend_cls, policy_cls, storage_dir):
    """close() must persist whatever the cache still holds.

    With a cache larger than the working set nothing ever spills, so today all
    five values live only in memory and ``close()`` -- which currently destroys the
    storage rather than flushing it -- loses every one.
    """
    path = str(storage_dir / "store")

    writer = _build(backend_cls.create(path), policy_cls, max_in_memory=100)
    for key, value in KEYS.items():
        writer[key] = value
    assert writer.disk_backend.keys() == [], "precondition: nothing spilled yet"
    writer.close()

    reader = _build(backend_cls.open(path), policy_cls)
    try:
        for key, value in KEYS.items():
            assert reader[key] == value, f"{key} was lost on close()"
    finally:
        release_store(reader)


# --------------------------------------------------------------------------
# durability across a real process boundary
# --------------------------------------------------------------------------

_WRITER_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {repo!r})
    from effidict import EffiDict, {backend}, LRUReplacement

    backend = {backend}.create({path!r})
    store = EffiDict(
        disk_backend=backend,
        replacement_strategy=LRUReplacement(disk_backend=backend, max_in_memory=2),
    )
    for key, value in {keys!r}.items():
        store[key] = value
    store.flush()
    store.close()
    print("written")
    """
)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "create()/open() are contract stubs, so the writer subprocess cannot build a store at a known path (issue 2.1)"
    ),
)
def test_data_survives_process_restart(backend_cls, storage_dir):
    """A different process must be able to read what this one wrote.

    Uses a real subprocess rather than a fresh object in-process: an in-process
    reopen shares the interpreter's state and would pass even if nothing reached
    the filesystem.
    """
    path = str(storage_dir / "store")
    script = _WRITER_SCRIPT.format(
        repo=REPO_ROOT, backend=backend_cls.__name__, path=path, keys=KEYS
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"writer subprocess failed:\n{result.stderr.strip()[-1500:]}"
    )

    reader = _build(backend_cls.open(path))
    try:
        for key, value in KEYS.items():
            assert reader[key] == value, f"{key} did not survive the restart"
    finally:
        release_store(reader)


# --------------------------------------------------------------------------
# scratch stores
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "temporary() is a contract stub (issue 2.1)"
    ),
)
def test_temporary_stores_are_unique(backend_cls):
    """``temporary()`` must hand out a fresh location every call.

    This is the supported way to get today's scratch-store behaviour once paths
    stop being salted, and it has to be collision-free by construction rather
    than by hoping two objects never share an address.
    """
    first = backend_cls.temporary()
    second = backend_cls.temporary()
    try:
        assert first.storage_path != second.storage_path
    finally:
        for backend in (first, second):
            release_store(backend)


@pytest.mark.xfail(
    strict=True,
    reason="temporary() is a contract stub (issue 2.1)",
)
def test_temporary_stores_clean_up_after_themselves(backend_cls):
    """Releasing a temporary store must leave nothing on disk.

    Asserted explicitly because ``temporary()`` puts storage outside
    ``storage_dir`` -- somewhere under the system temp directory -- where
    conftest's autouse leak detector cannot see it. Without this, a ``destroy()``
    that removed only part of its storage would quietly accumulate files on
    whatever machine ran the suite, and no test in the suite would notice.
    """
    backend = backend_cls.temporary()
    path = backend.storage_path
    assert os.path.exists(path), "temporary() did not create its storage"

    release_store(backend)

    assert not os.path.exists(path), f"temporary store left behind: {path}"
