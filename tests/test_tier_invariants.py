"""White-box specs for the eight tier invariants: where a key is allowed to live.

Each test names its invariant in the docstring. The invariants are owned by
``Store`` once issue 1.2 lands; today they are enforced nowhere, which is the
point of this file.

Tier membership is read through ``helpers.in_cache`` / ``helpers.on_disk`` rather
than by touching ``replacement_strategy.memory`` directly, so issue 1.2 can
repoint them at ``_store`` without rewriting ten tests.

Values are strings throughout. HDF5 widens ``int`` to ``int64`` and SQLite/JSON
turn a tuple into a list -- fidelity defects owned by issue 0.8 -- and a tier test
that tripped over those would be failing for the wrong reason.
"""

from __future__ import annotations

import threading

import pytest

from effidict import (
    DiskBackend,
    EvictionPolicy,
    LFUReplacement,
    LRUReplacement,
    MFUReplacement,
    MRUReplacement,
)

from .helpers import in_cache, on_disk

FILLER = [f"filler{i}" for i in range(12)]

#: Policies whose ``get`` writes a disk value back into the cache. Only these can
#: evict a clean entry, so only these violate I5 on a read-only pass.
PROMOTING_POLICIES = {
    LRUReplacement,
    MRUReplacement,
    LFUReplacement,
    MFUReplacement,
}


def _spill_to_disk(effidict, key):
    """Put ``key``'s current value on disk and drop it from the cache.

    Both halves matter. Stopping at "a disk copy exists" would let a caller read
    the still-cached value and pass without ever consulting the persistent tier --
    and because a reassignment leaves the old value on disk until the new one is
    written out, such a caller could read ``second`` from cache while ``first``
    still sits on disk, proving nothing.

    Deliberately bypasses the eviction policy rather than trying to provoke it.
    Forcing a *chosen* key to be the victim is not achievable across all seven
    policies: LFU and MFU break frequency ties by insertion order into their
    counter, so a freshly rewritten key always loses the tie and is never
    selected. Callers here need the resulting tier *state*, not the code path that
    produced it. ``test_eviction_writes_the_current_value`` covers the real
    eviction path.
    """
    value = effidict[key]
    effidict.disk_backend.serialize(key, value)
    effidict.replacement_strategy.delete(key)
    assert on_disk(effidict, key), "helper failed to place the key on disk"
    assert not in_cache(effidict, key), "helper failed to clear the cache copy"


def _seed_in_both_tiers(effidict, key, value):
    """Arrange for ``key`` to exist in the cache *and* on disk, with one value.

    Spill it, then write it again. The spill leaves the cache with a free slot, so
    the rewrite lands there without triggering an eviction, while the disk copy
    stays put. Callers must pass a freshly built dict.
    """
    effidict[key] = value
    _spill_to_disk(effidict, key)
    effidict[key] = value
    assert in_cache(effidict, key), "seeding failed: key is not cached"
    assert on_disk(effidict, key), "seeding failed: key is not on disk"


# --------------------------------------------------------------------------
# I4 -- delete removes from cache *and* backend
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I4: pop() returns via the self.memory.pop(key) path and never touches "
        "the backend, so the disk copy survives (issue 1.2)"
    ),
)
def test_pop_removes_key_from_both_tiers(backend_cls, policy_cls, make_dict):
    """I4. pop() must clear every tier, not just the cache.

    Today pop() takes the ``self.memory.pop(key)`` path and returns, leaving the
    disk copy behind, so the key is still present afterwards (issue 1.2).
    """
    d = make_dict(max_in_memory=2)
    _seed_in_both_tiers(d, "target", "payload")

    # Precondition: the key really is in both tiers, or this proves nothing.
    assert in_cache(d, "target")
    assert on_disk(d, "target")

    assert d.pop("target") == "payload"

    assert not in_cache(d, "target"), "cache copy survived pop"
    assert not on_disk(d, "target"), "disk copy survived pop"
    assert "target" not in d
    assert "target" not in d.keys()
    with pytest.raises(KeyError):
        d["target"]


def test_delitem_removes_key_from_both_tiers(backend_cls, policy_cls, make_dict):
    """I4. ``del d[key]`` must clear every tier in a single call.

    Regression pin: effidict <= 0.0.9 used an if/else and removed the cache copy
    *or* the disk copy, so deletion needed two calls. Fixed in 0.1.0; this keeps
    it fixed.
    """
    d = make_dict(max_in_memory=2)
    _seed_in_both_tiers(d, "target", "payload")

    assert in_cache(d, "target") and on_disk(d, "target")

    del d["target"]

    assert not in_cache(d, "target"), "cache copy survived del -- the <=0.0.9 bug"
    assert not on_disk(d, "target"), "disk copy survived del -- the <=0.0.9 bug"
    assert "target" not in d


def test_delitem_missing_key_raises(backend_cls, policy_cls, make_dict):
    """I4. Deleting an absent key raises KeyError, like dict."""
    d = make_dict(max_in_memory=2)

    with pytest.raises(KeyError):
        del d["never-written"]


def test_clear_removes_from_both_tiers(backend_cls, policy_cls, make_dict):
    """I4. clear() must empty every tier, including keys spilled to disk."""
    d = make_dict(max_in_memory=2)
    for i in range(6):
        d[f"k{i}"] = f"v{i}"

    assert any(on_disk(d, f"k{i}") for i in range(6)), "test needs a disk spill"

    d.clear()

    assert len(d) == 0
    assert list(d.keys()) == []
    assert d.disk_backend.keys() == []
    for i in range(6):
        assert not in_cache(d, f"k{i}")
        assert not on_disk(d, f"k{i}")


# --------------------------------------------------------------------------
# I5 / I6 -- eviction ordering and read purity
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I5: eviction drops the cache copy before the disk write completes, so "
        "the key is briefly in neither tier (issues 1.3, 5.1)"
    ),
)
def test_no_key_is_ever_absent_from_both_tiers(backend_cls, policy_cls, make_dict):
    """I5. A dirty eviction must write *then* drop, never drop then write.

    Blocks inside ``serialize`` and asks for the victim while the write is in
    flight. Deterministic: the probe waits on an event rather than a sleep, and
    the victim is whichever key the policy actually chose.
    """
    d = make_dict(max_in_memory=2)
    d["k0"] = "v0"
    d["k1"] = "v1"

    original = d.disk_backend.serialize
    started = threading.Event()
    release = threading.Event()
    victim = {}

    def blocking_serialize(key, value):
        victim["key"] = key
        started.set()
        release.wait(10)
        return original(key, value)

    writer_error = {}

    def write():
        # Recorded rather than raised: on SqliteBackend the real write fails with
        # ProgrammingError (thread affinity, issue 2.3), and an unhandled thread
        # exception would surface as a warning that obscures this test's finding.
        try:
            d["k2"] = "v2"
        except Exception as exc:  # noqa: BLE001
            writer_error["exc"] = exc

    d.disk_backend.serialize = blocking_serialize
    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        assert started.wait(10), "eviction never reached the backend"
        key = victim["key"]

        assert key in d, (
            f"{key!r} is in neither tier while its write is in flight "
            f"(cache={in_cache(d, key)}, disk={on_disk(d, key)})"
        )
        assert d[key] == f"v{key[1:]}"
    finally:
        release.set()
        writer.join(10)
        d.disk_backend.serialize = original


def test_clean_eviction_performs_no_io(
    request, backend_cls, policy_cls, make_dict, backend_spy
):
    """I5, I6. Evicting an unmodified entry is a memory drop with zero writes.

    A read-only pass over keys already on disk must not write. Measured with a
    call spy, not a timer.

    The defect is policy-scoped, so the xfail is applied at runtime rather than
    as a blanket decorator: Random/FIFO/LIFO read straight through without
    caching, so they never evict during a read and already satisfy I5. Only the
    promoting policies write back a value the disk already holds.
    """
    if policy_cls in PROMOTING_POLICIES:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "I5/I6: a promoted entry is never marked clean, so evicting "
                    "it rewrites a value the disk already holds (issue 1.3)"
                ),
            )
        )

    d = make_dict(max_in_memory=5)
    keys = [f"k{i}" for i in range(30)]
    for key in keys:
        d[key] = f"v-{key}"

    with backend_spy(d.disk_backend) as spy:
        for key in keys:
            assert d[key] == f"v-{key}"

        assert spy.count("serialize") == 0, (
            f"a read-only pass issued {spy.count('serialize')} writes"
        )


def test_reading_a_cached_key_never_writes(backend_cls, policy_cls, make_dict, backend_spy):
    """I6. Re-reading a key already in the cache must not touch the backend."""
    d = make_dict(max_in_memory=5)
    d["hot"] = "payload"

    with backend_spy(d.disk_backend) as spy:
        for _ in range(5):
            assert d["hot"] == "payload"

        assert spy.count("serialize") == 0


# --------------------------------------------------------------------------
# I3 -- membership is a lookup, never a scan
# --------------------------------------------------------------------------


def test_membership_of_a_cached_key_never_touches_the_backend(
    backend_cls, policy_cls, make_dict, backend_spy
):
    """I3. A cached hit short-circuits before the backend is consulted.

    This half of I3 already holds: ``__contains__`` is
    ``key in self.memory or ...``, so a resident key never reaches the disk tier.
    """
    d = make_dict(max_in_memory=5)
    d["hot"] = "payload"
    assert in_cache(d, "hot")

    with backend_spy(d.disk_backend) as spy:
        assert "hot" in d

        assert spy.count("keys") == 0
        assert spy.count("iter_keys") == 0


@pytest.mark.xfail(
    strict=True,
    reason="I3: __contains__ calls disk_backend.keys(), an O(N) scan (issue 2.2)",
)
@pytest.mark.parametrize("resident", [False, True], ids=["miss", "on-disk"])
def test_membership_never_lists_the_keyspace(
    backend_cls, policy_cls, make_dict, backend_spy, resident
):
    """I3. ``key in d`` is a point lookup; it must never enumerate the keyspace.

    Probes the two cases that have to reach the persistent tier: a key that was
    never written, and one that has been evicted to disk. A cached key is covered
    separately above, since it short-circuits and would XPASS here.
    """
    d = make_dict(max_in_memory=2)
    d["target"] = "payload"

    if resident:
        _spill_to_disk(d, "target")
        probe = "target"
    else:
        probe = "never-written"

    with backend_spy(d.disk_backend) as spy:
        assert (probe in d) is resident

        assert spy.count("keys") == 0, "__contains__ listed the whole keyspace"
        assert spy.count("iter_keys") == 0


# --------------------------------------------------------------------------
# I1 -- disk is authoritative; None is a value, not an absence
# --------------------------------------------------------------------------


def test_none_is_a_storable_value(backend_cls, policy_cls, make_dict):
    """I1. ``None`` round-trips and stays distinguishable from a missing key.

    LRU and MRU guard their write-back with ``if value is not None`` while LFU
    and MFU do not, so the policies disagree about what a stored ``None`` means.
    """
    d = make_dict(max_in_memory=2)
    d["nothing"] = None

    assert d["nothing"] is None
    assert "nothing" in d
    assert "nothing" in d.keys()
    assert len(d) == 1

    del d["nothing"]
    assert "nothing" not in d
    with pytest.raises(KeyError):
        d["nothing"]


def test_none_survives_a_spill_to_disk(backend_cls, policy_cls, make_dict):
    """I1. A stored ``None`` must still read back as ``None`` from the disk tier.

    Separate from the in-cache case so the read genuinely comes from the backend
    rather than short-circuiting on a cache hit.
    """
    d = make_dict(max_in_memory=2)
    d["nothing"] = None

    _spill_to_disk(d, "nothing")

    assert d["nothing"] is None
    assert "nothing" in d
    assert len(d.keys()) == len(set(d.keys()))


def test_eviction_writes_the_current_value(
    backend_cls, policy_cls, make_dict, backend_spy
):
    """I1. Whatever the policy evicts, it must persist the value the cache held.

    Covers the real eviction path, which ``_spill_to_disk`` deliberately bypasses.
    Policy-agnostic: it asserts on whichever victim was chosen rather than trying
    to dictate one.
    """
    d = make_dict(max_in_memory=2)
    d["a"] = "va"
    d["b"] = "vb"

    with backend_spy(d.disk_backend) as spy:
        d["c"] = "vc"
        writes = [call for call in spy.calls if call[0] == "serialize"]

    assert writes, "adding a third key to a 2-slot cache evicted nothing"
    for _, (key, value), _ in writes:
        assert value == f"v{key}", f"evicted {key!r} was written as {value!r}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I1/I7: a reassigned key's stale disk copy can only be superseded by a "
        "write-back, and flush() is still a stub (issue 1.4)"
    ),
)
def test_reassignment_supersedes_the_stale_disk_copy(
    backend_cls, policy_cls, make_dict
):
    """I1, I7. Overwriting a spilled key must not leave the old value on disk.

    Immediately after the overwrite the disk copy is legitimately stale -- that is
    what a dirty cache entry means (I2). The defect is that nothing can make disk
    authoritative again on demand, because ``flush()`` is unimplemented.
    """
    d = make_dict(max_in_memory=2)
    d["target"] = "first"
    _spill_to_disk(d, "target")
    assert d.disk_backend.deserialize("target") == "first"

    d["target"] = "second"
    d.flush()

    assert d.disk_backend.deserialize("target") == "second"
    assert d["target"] == "second"


# --------------------------------------------------------------------------
# I7 -- durability
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="I7: flush() is still a contract stub (issue 1.4)",
)
def test_flush_persists_dirty_entries(backend_cls, policy_cls, make_dict):
    """I7. After flush(), every cached write is on disk.

    Today nothing is ever flushed: values written while the cache has room never
    reach the backend, and there is no way to force them there.
    """
    d = make_dict(max_in_memory=10)
    for i in range(5):
        d[f"k{i}"] = f"v{i}"

    assert d.disk_backend.keys() == [], "precondition: nothing spilled yet"

    d.flush()

    for i in range(5):
        assert on_disk(d, f"k{i}"), f"k{i} still not persisted after flush()"
        assert d.disk_backend.deserialize(f"k{i}") == f"v{i}"


# --------------------------------------------------------------------------
# I8 -- the cache honours a byte budget
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="I8: max_bytes is stored but never enforced (issue 1.4)",
)
def test_cache_respects_byte_budget(backend_cls, policy_cls, make_dict):
    """I8. The cache is bounded by bytes, not only by item count.

    ``max_in_memory`` counts items, so 100 heavy values sit in a '100-item' cache
    regardless of size -- measured at 802 MB with 8 MB values. Scaled down here
    to keep the suite fast; the failure mode is identical.
    """
    budget = 512 * 1024
    payload = "x" * (64 * 1024)

    d = make_dict(max_in_memory=100, max_bytes=budget)
    for i in range(40):
        d[f"k{i}"] = payload

    held = sum(len(v) for v in d.replacement_strategy.memory.values())
    assert held <= budget * 2, (
        f"cache holds ~{held // 1024} KiB against a {budget // 1024} KiB budget; "
        f"{len(d.replacement_strategy.memory)} items resident"
    )


# --------------------------------------------------------------------------
# base classes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("base", [DiskBackend, EvictionPolicy], ids=lambda c: c.__name__)
def test_abstract_bases_cannot_be_instantiated(base):
    """The contract bases are real ABCs.

    ``@abstractmethod`` without ``ABCMeta`` is inert -- it was decorative before
    issue 0.1, so the base classes could be constructed and subclasses were never
    checked for completeness.
    """
    with pytest.raises(TypeError):
        base()


def test_concrete_backends_remain_instantiable(backend_cls, storage_dir):
    """Making the base abstract must not break the four shipped backends."""
    backend = backend_cls(str(storage_dir / f"abc-{backend_cls.__name__}"))
    try:
        assert isinstance(backend, DiskBackend)
    finally:
        backend.destroy()
