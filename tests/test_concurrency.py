"""Specs for thread safety and worker-process access.

Everything here is marked ``stress``: these are the specs whose failures depend on
scheduling, so CI runs them repeatedly rather than once.

Which marker each test carries is deliberate, following the rule written into
``test_dict_conformance.py``: a ``strict`` xfail is a claim that the defect shows
up on *every* run, so only deterministic probes get one. Measured here:

* a single cross-thread backend operation fails 5/5 on SQLite -- deterministic,
  strict;
* the blocking-eviction probe is deterministic by construction -- strict;
* joblib destroying the shared storage happens 5/5 -- strict;
* a *particular* worker's read failing happens 2-3/5, because it depends on when
  a sibling's ``__del__`` fires -- non-strict;
* the 8-thread race fails 5/5 at high cache pressure but is still a race --
  non-strict, with the deterministic pins named.

The three multiprocess specs are written against the Phase 5b contract and stay
``xfail`` rather than skipped, so the boundary is documented rather than hidden.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from effidict import (
    EffiDict,
    JSONBackend,
    LRUPolicy,
    PickleBackend,
    SqliteBackend,
)

from .conftest import release_store

pytestmark = pytest.mark.stress

REPO_ROOT = str(Path(__file__).resolve().parents[1])

#: Backends that survive a pickle round-trip today. SQLite holds a live
#: ``sqlite3.Connection`` and HDF5 an open ``h5py.File``, neither of which can be
#: pickled, so they cannot reach a worker process at all (issue 2.3).
PICKLABLE_BACKENDS = {PickleBackend, JSONBackend}

#: Backends whose handle is bound to the thread that opened it.
THREAD_AFFINE_BACKENDS = {SqliteBackend}

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

WORKER_KEYS = {f"k{i}": f"v{i}" for i in range(6)}


def _build(backend, max_in_memory=2):
    return EffiDict(
        disk_backend=backend,
        policy=LRUPolicy(),
        max_in_memory=max_in_memory,
    )


# Module-level so joblib can pickle them by reference.
def _worker_read(store, key):
    try:
        return store[key]
    except Exception as exc:  # noqa: BLE001 - reported, not raised, across the boundary
        return f"<{type(exc).__name__}>"


def _worker_write(store, key, value):
    try:
        store[key] = value
        return "wrote"
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


# --------------------------------------------------------------------------
# threads
# --------------------------------------------------------------------------


def test_backend_is_usable_from_another_thread(request, backend_cls, storage_dir):
    """A backend opened in one thread must be usable from another.

    The deterministic half of thread safety, and the reason the stress test below
    cannot be strict: SQLite binds its connection to the creating thread, so a
    single cross-thread read fails every time, with no race involved.
    """
    if backend_cls in THREAD_AFFINE_BACKENDS:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "sqlite3 connections are bound to the thread that opened "
                    "them; check_same_thread is not set (issue 2.3)"
                ),
            )
        )

    backend = backend_cls(str(storage_dir / "store"))
    try:
        backend.serialize("k", "v")
        observed = {}

        def read():
            try:
                observed["value"] = backend.deserialize("k")
            except Exception as exc:  # noqa: BLE001
                observed["error"] = exc

        # daemon, like every other probe thread here. If the backend call
        # deadlocks, join() below reports the timeout and the assertion fails --
        # but a *non*-daemon thread then keeps the interpreter alive at exit
        # while the fixture closes the backend underneath it, so the CI job
        # hangs forever instead of reporting the failure it already found.
        reader = threading.Thread(target=read, name="cross-thread-reader", daemon=True)
        reader.start()
        reader.join(10)
        assert not reader.is_alive(), "the reading thread did not finish within 10s"

        assert "error" not in observed, (
            f"reading from another thread raised {observed['error']!r}"
        )
        assert observed["value"] == "v"
    finally:
        release_store(backend)


def test_concurrent_writers_see_every_key(request, backend_cls, make_dict):
    """Eight threads writing their own keys must each read their own back.

    Issue 1.2 closed the eviction gap -- the store writes a victim out before
    dropping it, under one lock -- so the three file-backed backends now pass
    outright. SQLite still raises ``ProgrammingError`` across threads, so it
    keeps the marker until issue 2.3. Not strict even there: scheduling decides
    which thread reaches the connection first, and a bet that every run loses is
    the kind of marker that XPASSes in CI at the worst moment. The deterministic
    pins for the same defects are ``test_backend_is_usable_from_another_thread``
    above and ``test_reader_never_observes_a_key_mid_eviction`` below, plus 0.4's
    ``test_no_key_is_ever_absent_from_both_tiers``.

    The cache is kept small on purpose: with room for the whole working set the
    eviction gap never opens and the race cannot be observed at all.
    """
    if backend_cls is SqliteBackend:
        request.applymarker(
            pytest.mark.xfail(
                strict=False,  # a race; see the docstring
                reason=(
                    "concurrent writers lose keys: SQLite raises "
                    "ProgrammingError across threads (issue 2.3)"
                ),
            )
        )

    d = make_dict(max_in_memory=2)
    failures = []

    def worker(index):
        try:
            for i in range(200):
                key = f"t{index}_{i}"
                d[key] = key
                assert d[key] == key, f"{key} read back wrong"
        except Exception as exc:  # noqa: BLE001
            failures.append(f"thread {index}: {type(exc).__name__}: {exc}")

    threads = [
        # daemon: a writer wedged on a backend lock must not outlive the test
        # and hang the interpreter at exit (see the cross-thread reader above).
        threading.Thread(target=worker, args=(i,), name=f"writer-{i}", daemon=True)
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    # join() returning proves nothing on its own -- it also returns on timeout.
    # Asserted before the failure list, because a hung writer means teardown is
    # about to destroy a store that a live thread is still using.
    still_running = [thread.name for thread in threads if thread.is_alive()]
    assert not still_running, f"writers did not finish within 60s: {still_running}"

    assert not failures, f"{len(failures)} of 8 writers failed: {failures[:3]}"

    # Each writer's in-loop read-back only proves the key survived until the next
    # statement in that thread. A sibling can clobber or drop it a moment later
    # and every worker still finishes clean, so the name of this test was a
    # stronger claim than its assertions. Re-read the whole keyspace now that
    # nothing is writing.
    expected = {f"t{index}_{i}": f"t{index}_{i}" for index in range(8) for i in range(200)}

    # Every lookup goes through this, membership included. Both ``in`` and
    # ``[]`` consult the backend, and after eight threads have hammered it those
    # calls themselves raise -- KeyError from the eviction gap, and on SQLite
    # ProgrammingError from thread affinity (issue 2.3). A raw exception here
    # replaces this test's diagnostic with a bare traceback: measured at 4 of 28
    # matrix cases reporting nothing about what the race actually lost. Turning
    # the exception into the observed value keeps it a finding and reports it
    # alongside the keys that went missing.
    def read(key):
        try:
            return d[key]
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            return f"<{type(exc).__name__}: {exc}>"

    observed = {key: read(key) for key in expected}
    wrong = {key: value for key, value in observed.items() if value != expected[key]}
    assert not wrong, (
        f"{len(wrong)} of {len(expected)} keys did not survive the race "
        f"(first few: {dict(list(wrong.items())[:3])})"
    )


def test_reader_never_observes_a_key_mid_eviction(request, backend_cls, make_dict):
    """A reader must never see a key vanish while it is being written out.

    Deterministic by construction: the probe blocks *inside* ``serialize`` on an
    event and the reader runs while the write is held open, so there is no timing
    assumption. Complements 0.4's version of this invariant -- that one asserts
    the tier state, this one asserts what a concurrent reader observes.

    A reader that *blocks* is a pass, not a failure. ``Store`` holds a lock
    across tier updates (I4), so after issue 1.2 the correct behaviour is for
    this reader to wait for the eviction and then succeed -- it never observes
    the key as absent, which is the whole claim. An earlier version of this probe
    asserted ``not reader.is_alive()`` and therefore failed on a *correct* locked
    implementation: measured against a simulated one, it failed with "the reader
    blocked indefinitely" for both a compliant and a non-compliant store. What
    must never happen is the reader returning an error, which is exactly what it
    does today.

    Nothing here waits on a clock to decide *which* of those happened. Two events
    do it: the reader announces that it has entered the read path, and announces
    again when it leaves. A short grace period would be a scheduling assumption --
    a reader that had not been scheduled yet looks identical to one that
    blocked, and releasing the writer at that point lets a drop-then-write
    implementation answer from post-write state and XPASS a strict marker, five
    times per stress run. The waits below are 10s liveness bounds, the same ones
    used elsewhere in this file, not timing discriminators.

    Backend-scoped rather than blanket since issue 1.2, which made eviction
    write *then* drop. On SQLite the evicting write still fails with the
    thread-affinity error, which this probe asserts rather than swallows, so
    that backend stays red until issue 2.3.
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

    d = make_dict(max_in_memory=2)
    d["k0"] = "v0"
    d["k1"] = "v1"

    original = d._store.backend.serialize
    started = threading.Event()
    release = threading.Event()
    victim = {}
    writer_error = {}

    def blocking_serialize(key, value):
        victim["key"] = key
        started.set()
        release.wait(HOLD_SECONDS)
        return original(key, value)

    def write():
        # Recorded rather than raised: on SQLite the real write fails with
        # ProgrammingError (issue 2.3), and an unhandled thread exception would
        # surface as a warning that obscures this test's finding.
        try:
            d["k2"] = "v2"
        except Exception as exc:  # noqa: BLE001
            writer_error["exc"] = exc

    d._store.backend.serialize = blocking_serialize
    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        assert started.wait(WAIT_SECONDS), "eviction never reached the backend"
        key = victim["key"]

        observed = {}
        entered = threading.Event()
        finished = threading.Event()

        # Fired from *inside* the lookup, not before it. Setting an event at the
        # top of the thread body only proves the thread was scheduled: it can be
        # descheduled between that line and the subscript, the writer finishes,
        # and the probe then reports a mid-eviction read that never happened.
        # Patched on the instance and restored in the finally below, like the
        # blocking serialize above. Issue 1.2 moved this delegation to
        # ``Store.get``, and the hook moved with it.
        original_lookup = d._store.get

        def instrumented_lookup(wanted):
            if wanted == key:
                entered.set()
            return original_lookup(wanted)

        d._store.get = instrumented_lookup

        def read():
            try:
                observed["value"] = d[key]
            except Exception as exc:  # noqa: BLE001
                observed["error"] = exc
            finally:
                finished.set()

        reader = threading.Thread(target=read, daemon=True, name="mid-eviction-reader")
        reader.start()

        # Distinguishes "not scheduled yet" from "blocked on the lock", which a
        # timer cannot. If this fails the machine is wedged, not the store.
        #
        # Known residual gap: the hook fires on entry to the lookup, so a reader
        # descheduled between that and the read itself is still indistinguishable
        # from one blocked on a lock -- both leave `observed` empty, and the
        # post-release assertions below then pass against a drop-then-write
        # store. The window is one call wide and the probe waits WAIT_SECONDS
        # with the write held, so it is narrow rather than closed. Closing it
        # needs a hook at the lock boundary itself, which arrives with the
        # `Lock` protocol in issue 5.1; this spec should take it then.
        assert entered.wait(WAIT_SECONDS), "the reader never entered the lookup"

        # The write is still held, so whatever the reader returns here it formed
        # mid-eviction. If it has not returned, it is waiting on the Store lock.
        #
        # This is the one wait that costs something: once a lock exists the
        # reader legitimately never returns, so each backend spends the full
        # bound here -- about 40s for the matrix, and the stress step repeats it
        # five times. Deliberate. Shortening it would turn the bound back into a
        # discriminator, and the probe would then pass a drop-then-write store
        # whose reader simply had not been scheduled yet. Verified both ways:
        # unlocked, a drop-then-write store is caught in 0.0s with KeyError;
        # locked, both orderings pass, which is correct because the lock hides
        # the gap from readers and the ordering itself is 0.4's to pin.
        answered_mid_eviction = finished.wait(WAIT_SECONDS)

        assert "error" not in observed, (
            f"a concurrent reader saw {key!r} as {observed['error']!r} while its "
            f"write was still in flight"
        )

        release.set()
        assert finished.wait(WAIT_SECONDS), (
            "the reader never finished, even after the eviction was released"
        )
        assert "error" not in observed, (
            f"a concurrent reader blocked through the eviction and then still "
            f"failed with {observed.get('error')!r}"
        )
        assert observed["value"] == f"v{key[1:]}", (
            f"reader saw {observed['value']!r}, expected {f'v{key[1:]}'!r} "
            f"(answered mid-eviction: {answered_mid_eviction})"
        )

        # The writer's own outcome is asserted too, and only after the reader
        # checks above so the primary finding is reported first. Recording the
        # error without asserting it would let this probe XPASS once the eviction
        # ordering is fixed while SQLite's write was still failing on thread
        # affinity -- a pass claiming more than it proved.
        release.set()
        writer.join(WAIT_SECONDS)
        assert not writer.is_alive(), (
            f"the evicting writer did not finish within {WAIT_SECONDS}s"
        )
        assert "exc" not in writer_error, (
            f"the eviction itself failed with {writer_error['exc']!r}, so this "
            f"probe never observed a completed write"
        )
    finally:
        release.set()  # idempotent; frees the writer if an assertion above failed
        writer.join(WAIT_SECONDS)
        d._store.backend.serialize = original
        if "instrumented_lookup" in locals():
            d._store.get = original_lookup


# --------------------------------------------------------------------------
# crossing a process boundary
# --------------------------------------------------------------------------


def test_all_backends_are_picklable(request, backend_cls, storage_dir):
    """Every backend must survive a pickle round-trip.

    A store that cannot be pickled cannot reach a joblib worker at all, so this
    gates every other spec in this section.
    """
    if backend_cls not in PICKLABLE_BACKENDS:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{backend_cls.__name__} holds a live handle that pickle "
                    f"cannot serialise (issue 2.3)"
                ),
            )
        )

    backend = backend_cls(str(storage_dir / "store"))
    revived = None
    try:
        backend.serialize("k", "v")

        revived = pickle.loads(pickle.dumps(backend))

        assert revived.deserialize("k") == "v"
    finally:
        # The revived object is a second live handle on the same storage -- a
        # sqlite3.Connection or an h5py.File of its own. Leaving it open leaks a
        # descriptor for the rest of the session and, on Windows, blocks cleanup
        # of the storage this test just wrote.
        for handle in (revived, backend):
            if handle is not None:
                release_store(handle)


def test_picklable_backend_table_matches_reality(backend_cls, storage_dir):
    """Re-derive ``PICKLABLE_BACKENDS`` so the literal set cannot go stale."""
    backend = backend_cls(str(storage_dir / "store"))
    try:
        try:
            pickle.dumps(backend)
            picklable = True
        except Exception:  # noqa: BLE001
            picklable = False

        assert picklable == (backend_cls in PICKLABLE_BACKENDS), (
            f"{backend_cls.__name__}: table says "
            f"{backend_cls in PICKLABLE_BACKENDS}, reality says {picklable}"
        )
    finally:
        release_store(backend)


@pytest.mark.parametrize(
    "backend_cls", sorted(PICKLABLE_BACKENDS, key=lambda c: c.__name__),
    ids=lambda cls: cls.__name__,
)
@pytest.mark.xfail(
    strict=False,  # depends on when a sibling worker is collected; see below
    reason=(
        "a worker's __del__ destroys the shared storage, so sibling workers read "
        "from a store that has already been deleted (issue 3.1)"
    ),
)
def test_joblib_workers_can_read(backend_cls, storage_dir):
    """Worker processes must be able to read the store handed to them.

    Not strict: whether a *given* worker's read fails depends on whether a sibling
    has already been collected, measured at 2-3 failures in 5 runs. The
    deterministic pin for the same defect is
    ``test_joblib_workers_do_not_destroy_storage`` below, which fails 5/5.

    Restricted to the backends that can be pickled at all -- SQLite and HDF5 are
    blocked earlier by issue 2.3, and ``test_all_backends_are_picklable`` owns
    that finding.
    """
    joblib = pytest.importorskip("joblib")

    backend = backend_cls(str(storage_dir / "store"))
    try:
        d = _build(backend, max_in_memory=4)
        for key, value in WORKER_KEYS.items():
            d[key] = value

        results = joblib.Parallel(n_jobs=3)(
            joblib.delayed(_worker_read)(d, key) for key in WORKER_KEYS
        )

        broken = [r for r in results if isinstance(r, str) and r.startswith("<")]
        assert not broken, f"workers could not read: {broken}"
        assert sorted(results) == sorted(WORKER_KEYS.values())
    finally:
        release_store(backend)


@pytest.mark.parametrize(
    "backend_cls", sorted(PICKLABLE_BACKENDS, key=lambda c: c.__name__),
    ids=lambda cls: cls.__name__,
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "a worker process owns the storage it was handed, so its __del__ deletes "
        "the parent's cache mid-run (issue 3.1)"
    ),
)
def test_joblib_workers_do_not_destroy_storage(backend_cls, storage_dir):
    """The oligo-designer-toolsuite bug, as a spec.

    Handing a store to workers must leave it intact. Today the storage is gone
    5/5 by the time ``Parallel`` returns, because each worker's copy runs
    ``__del__`` when its task ends and that path still calls ``destroy()``.
    """
    joblib = pytest.importorskip("joblib")

    backend = backend_cls(str(storage_dir / "store"))
    try:
        d = _build(backend, max_in_memory=4)
        for key, value in WORKER_KEYS.items():
            d[key] = value
        path = backend.storage_path
        assert os.path.exists(path), "precondition: the store exists"

        joblib.Parallel(n_jobs=3)(
            joblib.delayed(_worker_read)(d, key) for key in WORKER_KEYS
        )

        assert os.path.exists(path), "the workers destroyed the parent's storage"
        for key, value in WORKER_KEYS.items():
            assert d[key] == value, f"{key} did not survive the workers"
    finally:
        release_store(backend)


@pytest.mark.parametrize(
    "backend_cls", sorted(PICKLABLE_BACKENDS, key=lambda c: c.__name__),
    ids=lambda cls: cls.__name__,
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "there is no read-only contract, so a worker's write is accepted and "
        "silently lost when the worker exits (issues 3.1, 5.2)"
    ),
)
def test_worker_write_raises_clearly(backend_cls, storage_dir):
    """A write from a worker process must fail loudly, not silently.

    The concurrency model is one writer process with read-only workers, so a
    worker write is a programming error. Accepting it is the damaging outcome: the
    value lands in a cache that is discarded when the task ends, so the caller
    believes the write happened.
    """
    joblib = pytest.importorskip("joblib")

    backend = backend_cls(str(storage_dir / "store"))
    try:
        d = _build(backend, max_in_memory=4)
        d["existing"] = "v"

        results = joblib.Parallel(n_jobs=2)(
            joblib.delayed(_worker_write)(d, f"from-worker-{i}", "v") for i in range(2)
        )

        refused = [r for r in results if isinstance(r, str) and r.startswith("<")]
        assert len(refused) == len(results), (
            f"a worker write was accepted rather than refused: {results}"
        )
        for message in refused:
            assert "read" in message.lower() or "writ" in message.lower(), (
                f"refusal does not explain that workers are read-only: {message}"
            )

        # Refusing is not the same as not writing. A store that persisted the
        # value and *then* raised would satisfy every assertion above while
        # breaking the read-only contract in the way that actually costs the
        # caller something, so check the parent never saw the keys.
        leaked = [f"from-worker-{i}" for i in range(2) if f"from-worker-{i}" in d]
        assert not leaked, (
            f"a worker write was refused but still reached the store: {leaked}"
        )
    finally:
        release_store(backend)


# --------------------------------------------------------------------------
# Phase 5b: multiple processes writing the same store
# --------------------------------------------------------------------------
#
# Written now and left xfail so the boundary is documented rather than absent.
# These use subprocesses rather than multiprocessing, so a failure shows the
# child's own traceback instead of a pickling error from the process pool.

_WRITER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {repo!r})
    from effidict import EffiDict, {backend}, LRUPolicy

    backend = {backend}.open_or_create({path!r})
    store = EffiDict(
        disk_backend=backend,
        policy=LRUPolicy(),
        max_in_memory=4,
    )
    for key, value in {items!r}.items():
        store[key] = value
    store.flush()
    store.close()
    print("ok")
    """
)


def _writer_command(backend_cls, path, items):
    return [
        sys.executable,
        "-c",
        _WRITER.format(
            repo=REPO_ROOT, backend=backend_cls.__name__, path=path, items=items
        ),
    ]


def _run_writer(backend_cls, path, items):
    """Run one writer to completion. For the specs that need ordering."""
    return subprocess.run(
        _writer_command(backend_cls, path, items),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_writers_concurrently(backend_cls, path, items_list):
    """Start every writer *before* waiting on any of them.

    ``subprocess.run`` waits, so collecting results with it runs the writers one
    after another -- which tests sequential persistence, not the concurrent
    contract these specs exist for. Sequential overwrites cannot tear, so an
    implementation with no coordination at all would pass. ``Popen`` starts them
    all first and only then collects.

    Returns ``(returncode, stderr)`` pairs so callers can report the child's own
    traceback rather than just a status.
    """
    processes = [
        subprocess.Popen(
            _writer_command(backend_cls, path, items),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for items in items_list
    ]
    results = []
    for process in processes:
        try:
            _, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
            stderr = f"timed out after 120s\n{stderr}"
        results.append((process.returncode, stderr))
    return results


def _writer_failures(results):
    return [stderr.strip()[-400:] for code, stderr in results if code != 0]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "multiple writer processes are out of scope until Phase 5b, and "
        "open_or_create() is still a contract stub (issues 2.1, 5b)"
    ),
)
def test_multiprocess_disjoint_writes_all_visible(backend_cls, storage_dir):
    """Processes writing disjoint key ranges must all be visible afterwards."""
    path = str(storage_dir / "shared")
    ranges = [{f"p{p}_k{i}": f"v{i}" for i in range(4)} for p in range(3)]

    failures = _writer_failures(_run_writers_concurrently(backend_cls, path, ranges))
    assert not failures, "writer processes failed:\n" + "\n".join(failures)

    reader = _build(backend_cls.open(path), max_in_memory=4)
    try:
        for items in ranges:
            for key, value in items.items():
                assert reader[key] == value, f"{key} written by another process is missing"
    finally:
        release_store(reader)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "concurrent writes to one key are out of scope until Phase 5b, and "
        "open_or_create() is still a contract stub (issues 2.1, 5b)"
    ),
)
def test_multiprocess_same_key_writes_are_not_torn(backend_cls, storage_dir):
    """Concurrent writes to one key must leave a value some writer actually wrote.

    Last-writer-wins is acceptable; a torn or absent value is not.
    """
    path = str(storage_dir / "shared")
    candidates = {f"from-p{p}" for p in range(3)}

    # Launched together, not in sequence: three sequential overwrites cannot tear,
    # so a sequential version of this test would pass on an implementation with no
    # write coordination whatsoever.
    failures = _writer_failures(
        _run_writers_concurrently(
            backend_cls, path, [{"contended": value} for value in sorted(candidates)]
        )
    )
    assert not failures, "writer processes failed:\n" + "\n".join(failures)

    reader = _build(backend_cls.open(path), max_in_memory=4)
    try:
        assert reader["contended"] in candidates, (
            f"the contended key holds {reader['contended']!r}, which no writer wrote"
        )
    finally:
        release_store(reader)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "there is no cross-process cache invalidation until Phase 5b, and "
        "open_or_create() is still a contract stub (issues 2.1, 5b)"
    ),
)
def test_stale_cache_is_invalidated_after_external_write(backend_cls, storage_dir):
    """A cached value must not survive another process overwriting it.

    The hard half of multiprocess support, and the reason Phase 5b needs a
    coherence mechanism rather than just a lock: this process has the old value
    cached and nothing tells it the store moved on.
    """
    # Deliberately sequential: the point is that the first value is *cached*
    # before the second write happens, which needs ordering rather than overlap.
    path = str(storage_dir / "shared")
    first = _run_writer(backend_cls, path, {"shared": "first"})
    assert first.returncode == 0, first.stderr.strip()[-400:]

    reader = _build(backend_cls.open(path), max_in_memory=4)
    try:
        assert reader["shared"] == "first"  # now cached

        second = _run_writer(backend_cls, path, {"shared": "second"})
        assert second.returncode == 0, second.stderr.strip()[-400:]

        assert reader["shared"] == "second", (
            "the reader served a stale cached value after another process "
            "overwrote the key"
        )
    finally:
        release_store(reader)
