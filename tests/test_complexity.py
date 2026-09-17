"""Regression guards for cost: how work grows with the size of the store.

Marked ``slow`` as a file, so the default local loop skips them and CI's plain
``pytest`` run includes them.

Two rules shape every guard here, both learned from the rest of this suite:

* **Count work, do not time it, wherever counting is possible.** A call spy is
  deterministic; a stopwatch is a flake waiting for a busy CI runner. The one
  timing guard present is non-strict and exists only because the issue's table
  quotes a wall-clock ratio -- the deterministic guard beside it is the real pin.
* **Compare two sizes in the same run, never against an absolute number.** A
  threshold in microseconds or bytes measures the machine; a ratio measures the
  implementation.

``test_compact_reclaims_deleted_space`` in ``test_backend_contract.py`` owns the
"store size after deleting every key" row of that table, so it is not repeated
here.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import time

import pytest

from effidict import Hdf5Backend, SqliteBackend

pytestmark = pytest.mark.slow

SMALL = 100
LARGE = 4000


def _count_enumerated(backend):
    """Wrap ``keys()`` to record how many keys each call hands back.

    Counting *calls* is not enough: ``__contains__`` calls ``keys()`` exactly once
    whatever the store size, and the cost is the length of the list it returns. So
    the guard counts keys enumerated, which grows with N today and is zero once
    membership becomes a point lookup.
    """
    enumerated = []
    original = backend.keys

    def counting_keys():
        result = original()
        enumerated.append(len(result))
        return result

    backend.keys = counting_keys
    return enumerated, lambda: setattr(backend, "keys", original)


# --------------------------------------------------------------------------
# membership
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "__contains__ calls disk_backend.keys(), so the work it does grows with "
        "the size of the store (issue 2.2)"
    ),
)
def test_membership_cost_does_not_grow_with_the_store(backend_cls, policy_cls, make_dict):
    """A membership check must cost the same at 4000 keys as at 100.

    Measured by counting the keys the backend is asked to enumerate, not by
    timing: 90 at N=100 and 3990 at N=4000 today, which is the linear scan behind
    the wall-clock ratio quoted in the issue.
    """
    enumerated_by_size = {}

    for size in (SMALL, LARGE):
        d = make_dict(max_in_memory=10)
        for i in range(size):
            d[f"k{i:05d}"] = "v"

        enumerated, restore = _count_enumerated(d.disk_backend)
        try:
            assert ("absent" in d) is False
        finally:
            restore()
        enumerated_by_size[size] = sum(enumerated)

    assert enumerated_by_size[LARGE] == 0, (
        f"one membership check enumerated {enumerated_by_size[LARGE]} keys at "
        f"N={LARGE} (and {enumerated_by_size[SMALL]} at N={SMALL}); it should "
        f"enumerate none"
    )


@pytest.mark.xfail(
    strict=False,  # wall clock; the deterministic pin is the guard above
    reason=(
        "__contains__ scans the keyspace, so it gets slower as the store grows "
        "(issue 2.2)"
    ),
)
def test_membership_time_is_flat(make_dict):
    """The wall-clock form of the guard above, as a ratio within one run.

    Not strict, and no absolute threshold: it exists because the issue's table
    quotes 18x, and a ratio is the only machine-independent way to express that.
    A busy runner can compress the ratio, which is exactly why the key-counting
    guard above carries the strict marker instead.
    """
    timings = {}
    for size in (SMALL, LARGE):
        d = make_dict(max_in_memory=10)
        for i in range(size):
            d[f"k{i:05d}"] = "v"
        start = time.perf_counter()
        for _ in range(50):
            ("absent" in d)
        timings[size] = (time.perf_counter() - start) / 50

    ratio = timings[LARGE] / timings[SMALL]
    assert ratio < 4, (
        f"membership got {ratio:.1f}x slower going from {SMALL} to {LARGE} keys "
        f"({timings[SMALL] * 1e6:.0f}us -> {timings[LARGE] * 1e6:.0f}us)"
    )


# --------------------------------------------------------------------------
# reads must not write
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a promoted entry is never marked clean, so evicting it rewrites a value "
        "the disk already holds (issue 1.3)"
    ),
)
def test_read_only_pass_performs_no_writes(make_dict, backend_spy):
    """Reading 500 keys through a 50-key cache must not write anything.

    Every value is already on disk, so each eviction during the pass rewrites
    bytes that are already there -- 500 writes for 500 reads today. The equivalent
    invariant is pinned per-policy in ``test_tier_invariants.py``; this is the
    at-scale cost guard.
    """
    d = make_dict(max_in_memory=50)
    keys = [f"k{i:04d}" for i in range(500)]
    for key in keys:
        d[key] = f"v-{key}"

    with backend_spy(d.disk_backend) as spy:
        for key in keys:
            assert d[key] == f"v-{key}"

        writes = spy.count("serialize") + spy.count("write_many")
        assert writes == 0, f"a read-only pass over {len(keys)} keys issued {writes} writes"


# --------------------------------------------------------------------------
# bulk writes
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "there is no batch path: load_from_dict loops over serialize, one backend "
        "call and one transaction per item (issue 4.1)"
    ),
)
def test_bulk_write_does_not_issue_one_call_per_item(make_dict, backend_spy):
    """Loading 3000 items must not mean 3000 backend round-trips.

    Counted rather than timed, so the guard states the structural property -- work
    proportional to chunks, not items -- instead of a speed that depends on the
    disk underneath. The 274x penalty in the issue's table is what that costs in
    wall clock.

    Driven through ``load_from_dict``, which is the bulk path that exists today,
    rather than ``update()`` -- that is still a contract stub, so calling it would
    make this guard fail with issue 6.1's ``NotImplementedError`` instead of
    measuring the batch defect it is named for. Issue 4.1 replaces
    ``load_from_dict`` with ``update()``; this call site moves then.
    """
    items = {f"k{i:05d}": "v" for i in range(3000)}
    d = make_dict(max_in_memory=10)

    with backend_spy(d.disk_backend) as spy:
        d.load_from_dict(items)

        per_item = spy.count("serialize")
        batched = spy.count("write_many")

    assert per_item <= len(items) // 100, (
        f"a bulk load of {len(items)} items issued {per_item} single writes and "
        f"{batched} batch calls"
    )


# --------------------------------------------------------------------------
# stored size
# --------------------------------------------------------------------------


def _store_bytes(path):
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(path)
            for filename in filenames
        )
    return os.path.getsize(path)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SQLite stores values as JSON text, so a list of integers costs roughly "
        "2.3x what the same data costs encoded as bytes (issue 4.3)"
    ),
)
def test_stored_payload_is_not_far_larger_than_the_data():
    """A stored value must not cost much more than the data it holds.

    Expressed against the same data pickled, in the same run, rather than a byte
    count: 68 KB of SQLite file for a 10k-integer list whose binary form is 29 KB.
    JSON text also rules out every value that is not JSON-encodable, which is the
    same defect seen from the other side in ``test_backend_contract.py``.
    """
    value = list(range(10000))
    binary_size = len(pickle.dumps(value))

    backend = SqliteBackend(os.path.join(tempfile.mkdtemp(), "store"))
    try:
        backend.serialize("payload", value)
        stored = _store_bytes(backend.storage_path)
    finally:
        try:
            backend.destroy()
        except OSError:
            pass

    assert stored <= binary_size * 1.5, (
        f"storing a 10k-integer list took {stored / 1024:.0f} KiB where its "
        f"binary form is {binary_size / 1024:.0f} KiB ({stored / binary_size:.1f}x)"
    )


#: Backends whose per-item cost rises as the store fills. Measured over 250 vs
#: 1000 items of 200 bytes: Pickle and JSON are exactly linear (215 and 202
#: bytes per item at both sizes), SQLite is sub-linear as its fixed header
#: amortises (279 -> 258), and HDF5 goes the wrong way (343 -> 569) because its
#: group index grows with the number of links.
SUPERLINEAR_BACKENDS = {Hdf5Backend}


def test_stored_size_grows_linearly_with_the_data(request, backend_cls):
    """Four times the data must not cost dramatically more than four times the space.

    A ratio within one run, so it says nothing about absolute efficiency -- that
    is the guard above -- only that cost per item does not degrade as the store
    fills. That distinction matters for a store aimed at data far larger than
    RAM: a layout whose overhead grows with key count gets worse exactly where it
    is needed most.
    """
    if backend_cls in SUPERLINEAR_BACKENDS:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "HDF5 group metadata grows with the number of datasets, so "
                    "per-item cost rises from ~343 to ~569 bytes between 250 and "
                    "1000 keys (issue 7.2)"
                ),
            )
        )

    sizes = {}
    for count in (250, 1000):
        backend = backend_cls(os.path.join(tempfile.mkdtemp(), "store"))
        try:
            for i in range(count):
                backend.serialize(f"k{i:05d}", "x" * 200)
            sizes[count] = _store_bytes(backend.storage_path)
        finally:
            try:
                backend.destroy()
            except OSError:
                pass

    growth = sizes[1000] / sizes[250]
    assert growth < 4 * 1.6, (
        f"4x the items took {growth:.1f}x the space "
        f"({sizes[250]} -> {sizes[1000]} bytes)"
    )
