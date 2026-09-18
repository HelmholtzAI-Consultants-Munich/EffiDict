"""Specs for what each backend guarantees about values and errors.

The value domain lives in one place the tests read: ``helpers.VALUE_KINDS`` with
``UNSUPPORTED`` and ``LOSSY``, established in issue 0.2 and re-derived on every run
by ``test_harness.test_value_support_tables_match_reality``. What is missing is the
*product* side -- a backend cannot currently be asked what it accepts, so a caller
has to find out by trying. ``test_backend_declares_its_value_domain`` pins that.

``LOSSY`` is deliberately separate from ``UNSUPPORTED``. Folding them together
would let these specs skip the silently-wrong round-trips, which are the more
dangerous half: HDF5 widens ``int`` to ``int64`` and turns a numeric DataFrame
into an ndarray, and SQLite/JSON return a tuple as a list.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from effidict import (
    Hdf5Backend,
    JSONBackend,
    PickleBackend,
    RandomReplacement,
    SqliteBackend,
)

from .helpers import (
    UNSUPPORTED,
    VALUE_KINDS,
    assert_equal_value,
    is_lossy,
    is_unsupported,
)

#: Where each backend decodes a stored payload. Patched to raise, to check an
#: internal decode failure is not reported as a missing key.
DECODE_HOOK = {
    SqliteBackend: "effidict.disk_backend.json.loads",
    JSONBackend: "effidict.disk_backend.json.load",
    PickleBackend: "effidict.disk_backend.pickle.load",
    Hdf5Backend: "effidict.disk_backend.ast.literal_eval",
}

#: Backends whose ``deserialize`` wraps its whole body in ``except KeyError``, so
#: a decode failure is indistinguishable from an absent key.
MASKS_DECODE_FAILURE = {Hdf5Backend}



def _policy_state(store):
    """Everything that decides the policy's next victim.

    Cached *membership* is not enough. FIFO, LIFO, LRU and MRU choose by position,
    so two caches holding the same keys in a different order evict differently;
    LFU and MFU choose by the counts in ``secondary_memory``, which membership
    does not see at all. A bulk load could therefore match on keys and still
    behave differently on the very next write.

    Reads the policy's own attributes, which issue 1.1 moves: the ``*Policy``
    classes keep the ordering and the counts but the cache dict belongs to
    ``Cache``. Update this reader then -- ``test_eviction_policies._instance_state``
    already walks policy state generically and is the model to follow. Left
    concrete here because the generic walk needs the post-1.1 shape to be useful.
    """
    policy = store.replacement_strategy
    state = {"order": list(policy.memory.keys())}
    frequencies = getattr(policy, "secondary_memory", None)
    if frequencies is not None:
        state["frequencies"] = dict(frequencies)
    return state


@pytest.fixture
def backend(backend_cls, storage_dir):
    instance = backend_cls(str(storage_dir / "store"))
    yield instance
    try:
        instance.destroy()
    except OSError:
        pass


def _value_for(kind):
    if kind == "ndarray":
        pytest.importorskip("numpy")
    if kind == "dataframe":
        pytest.importorskip("pandas")
    return VALUE_KINDS[kind]()


# --------------------------------------------------------------------------
# value domain
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="backends do not expose their value domain (issue 7.2)",
)
def test_backend_declares_its_value_domain(backend_cls, backend):
    """A backend must be able to say what it accepts, and must not lie.

    Today the only way to discover that SQLite cannot store an ndarray is to try
    it and read the traceback. A ``supports()`` predicate makes the domain part of
    the contract -- and is cross-checked here against what ``serialize`` actually
    does, because a declaration nobody verifies drifts.

    Called on the *instance*, not the class. Every other backend operation is an
    instance method, so ``def supports(self, value)`` is the natural
    implementation, and a class-level call would hand it ``value`` as ``self``.
    Going through the instance also accepts a ``classmethod`` or ``staticmethod``,
    so it constrains the signature less while still being correct -- and once
    issue 4.3 makes the codec pluggable, the value domain genuinely depends on
    which codec the instance was built with.
    """
    assert hasattr(backend, "supports"), (
        f"{backend_cls.__name__} does not declare its value domain"
    )

    disagreements = []
    for kind in sorted(VALUE_KINDS):
        value = _value_for(kind)
        declared = backend.supports(value)
        try:
            backend.serialize(f"probe-{kind}", value)
            actual = True
        except Exception:  # noqa: BLE001
            actual = False
        if declared != actual:
            disagreements.append(f"{kind}: declared {declared}, actual {actual}")

    assert not disagreements, f"supports() disagrees with reality: {disagreements}"


@pytest.mark.parametrize(
    "backend_cls",
    [cls for cls, kinds in UNSUPPORTED.items() if kinds],
    ids=lambda cls: cls.__name__,
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "an unsupported value raises whatever the codec raises -- 'Object of type "
        "ndarray is not JSON serializable' -- which names neither the backend nor "
        "the key (issue 7.2)"
    ),
)
def test_value_domain_is_enforced(backend_cls, storage_dir):
    """Rejecting an unsupported value must say which backend refused, and why.

    Only parametrized over the backends that actually refuse something: Pickle and
    HDF5 accept every kind in the catalogue, so there is nothing to enforce.
    """
    instance = backend_cls(str(storage_dir / "store"))
    try:
        for kind in sorted(UNSUPPORTED[backend_cls]):
            value = _value_for(kind)

            with pytest.raises(TypeError) as excinfo:
                instance.serialize("k", value)

            message = str(excinfo.value)
            assert backend_cls.__name__ in message, (
                f"{kind}: error does not name the backend: {message}"
            )
            # Naming the backend alone is not actionable -- TypeError("SqliteBackend")
            # would satisfy it. The caller also needs to know *what* was refused.
            assert type(value).__name__ in message, (
                f"{kind}: error does not name the rejected type "
                f"{type(value).__name__!r}: {message}"
            )
    finally:
        try:
            instance.destroy()
        except OSError:
            pass


@pytest.mark.parametrize("kind", sorted(VALUE_KINDS))
def test_value_roundtrip_fidelity(request, backend_cls, backend, kind):
    """A stored value must come back as the same value, or be refused.

    The middle ground -- accepted and quietly changed -- is the defect. Scoped
    per (backend, kind) from ``helpers.LOSSY``, since only some combinations are
    affected.
    """
    if is_unsupported(backend_cls, kind):
        pytest.skip(f"{backend_cls.__name__} does not accept {kind}")
    if is_lossy(backend_cls, kind):
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{backend_cls.__name__} accepts {kind} but does not return "
                    f"it unchanged (issue 7.2)"
                ),
            )
        )

    value = _value_for(kind)
    backend.serialize("k", value)

    assert_equal_value(backend.deserialize("k"), value)


# --------------------------------------------------------------------------
# error contract
# --------------------------------------------------------------------------


def test_missing_key_raises_keyerror_not_something_else(backend, backend_cls):
    """An absent key raises ``KeyError``, never a leaked filesystem error."""
    with pytest.raises(KeyError):
        backend.deserialize("never-written")

    backend.serialize("present", "v")
    with pytest.raises(KeyError):
        backend.deserialize("still-absent")


def test_decode_failure_is_not_reported_as_missing_key(request, backend_cls, backend):
    """A corrupt payload must not look like an absent key.

    ``Hdf5Backend.deserialize`` wraps its entire body in ``except KeyError`` and
    re-raises ``KeyError(key)``, so a decode failure is reported as "no such key"
    -- the caller silently treats real data loss as a cache miss. Injected by
    patching the decode step, which is deterministic and does not depend on
    manufacturing a corrupt file.
    """
    if backend_cls in MASKS_DECODE_FAILURE:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "Hdf5Backend.deserialize catches KeyError around its whole "
                    "body, so a decode failure is reported as a missing key "
                    "(issue 7.2)"
                ),
            )
        )

    hook = DECODE_HOOK.get(backend_cls)
    if hook is None:
        pytest.skip(f"no decode hook identified for {backend_cls.__name__}")

    backend.serialize("present", {"a": 1})
    sentinel = KeyError("internal decode failure")

    with mock.patch(hook, side_effect=sentinel):
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - type is the point
            backend.deserialize("present")

    # The injected failure must reach the caller unchanged. Asserting only that
    # it is not KeyError('present') would accept a backend that swallowed it and
    # raised some other KeyError, or wrapped it in a way that still reads as a
    # miss -- the payload was not decoded either way.
    assert excinfo.value is sentinel, (
        f"a decode failure surfaced as {excinfo.value!r} instead of propagating; "
        f"the caller cannot tell it apart from the key simply not being there"
    )


# --------------------------------------------------------------------------
# storage management
# --------------------------------------------------------------------------


def _store_size(path):
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(path)
            for filename in filenames
        )
    return os.path.getsize(path)


@pytest.mark.xfail(
    strict=True,
    reason="compact() is a contract stub (issue 7.2)",
)
def test_compact_reclaims_deleted_space(backend, backend_cls):
    """Deleting every key then compacting must shrink the store.

    Without this a long-running store only grows: deleting all 2000 rows from a
    SQLite store leaves the file at its full size, and HDF5 never returns space
    at all. ``compact()`` is explicit rather than automatic, because VACUUM on a
    large store is expensive.
    """
    payload = "x" * 2000
    keys = [f"k{i:04d}" for i in range(400)]
    for key in keys:
        backend.serialize(key, payload)
    full = _store_size(backend.storage_path)

    for key in keys:
        backend.del_item(key)
    assert backend.keys() == [], "precondition: every key was deleted"

    backend.compact()

    reclaimed = _store_size(backend.storage_path)
    assert reclaimed < full / 2, (
        f"{backend_cls.__name__} still holds {reclaimed} bytes after deleting "
        f"every key and compacting (was {full})"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "load_from_dict writes straight to the backend, bypassing the cache and "
        "its eviction accounting (issue 4.1)"
    ),
)
def test_batch_and_single_writes_are_equivalent(backend_cls, policy_cls, make_dict):
    """A bulk load must leave the store in the same state as individual writes.

    ``load_from_dict`` currently writes past the cache entirely, so the same data
    loaded in bulk versus item by item gives two different tier layouts -- and a
    bulk load leaves the cache cold, with nothing recorded in the policy.
    """
    items = {f"k{i}": f"v{i}" for i in range(6)}

    one_at_a_time = make_dict(max_in_memory=3)
    for key, value in items.items():
        one_at_a_time[key] = value

    in_bulk = make_dict(max_in_memory=3)
    in_bulk.load_from_dict(dict(items))

    # Captured before anything is read. On a promoting policy, reading a key
    # pulls it into the cache -- so asserting on cache size after the value
    # checks below would measure the reads this test performed rather than what
    # the bulk load did, and would pass on exactly the policies it should catch.
    bulk_state = _policy_state(in_bulk)
    individual_state = _policy_state(one_at_a_time)
    if policy_cls is RandomReplacement:
        # Random picks its victims by coin flip, so two stores given identical
        # writes legitimately retain different keys -- and therefore a different
        # order. Only the population size is a meaningful comparison.
        assert len(bulk_state["order"]) == len(individual_state["order"]), (
            f"bulk load left {len(bulk_state['order'])} entries in the cache "
            f"where the equivalent individual writes left "
            f"{len(individual_state['order'])}"
        )
    else:
        assert bulk_state == individual_state, (
            f"bulk load left the policy in a different state:\n"
            f"  bulk:       {bulk_state}\n"
            f"  individual: {individual_state}"
        )

    assert set(in_bulk.keys()) == set(one_at_a_time.keys())
    for key, value in items.items():
        assert in_bulk[key] == value


# --------------------------------------------------------------------------
# the batch and lookup API added in issue 0.1
# --------------------------------------------------------------------------
#
# ``read_many``, ``write_many``, ``delete_many``, ``has`` and ``count`` were
# added to DiskBackend as part of the target contract and then specified
# nowhere: before these tests, ``read_many``, ``delete_many`` and ``count`` had
# zero references in the suite and ``write_many`` appeared only inside spy call
# counts. All five could ship as stubs, or return wrong results, without a
# single strict xfail ever flipping.
#
# Every guard below is stated as *equivalence with the single-item operation*
# rather than as freshly invented semantics. The single-item behaviour is
# already pinned above -- ``deserialize`` on a missing key raises ``KeyError``,
# a stored value round-trips -- so equivalence both defines the batch contract
# and keeps the two halves of the API from drifting. It also decides the
# question the stub docstrings leave open: a batch operation is not licensed to
# be quieter about a missing key than the single-key call it replaces.

BATCH_ITEMS = {f"b{i}": f"v{i}" for i in range(6)}


def _single_write(backend_cls, storage_dir, items, name):
    """Reference state: the same items written one at a time."""
    reference = backend_cls(str(storage_dir / name))
    for key, value in items.items():
        reference.serialize(key, value)
    return reference


@pytest.mark.xfail(strict=True, reason="write_many is a contract stub (issue 4.1)")
def test_write_many_matches_individual_writes(backend_cls, backend, storage_dir):
    """A batched write must leave exactly what the same writes would leave."""
    backend.write_many(dict(BATCH_ITEMS))

    reference = _single_write(backend_cls, storage_dir, BATCH_ITEMS, "reference")
    try:
        assert sorted(backend.keys()) == sorted(reference.keys()), (
            f"write_many stored {sorted(backend.keys())}, the same writes one "
            f"at a time stored {sorted(reference.keys())}"
        )
        for key, value in BATCH_ITEMS.items():
            assert_equal_value(backend.deserialize(key), value)
    finally:
        try:
            reference.destroy()
        except OSError:
            pass


@pytest.mark.xfail(strict=True, reason="read_many is a contract stub (issue 4.1)")
def test_read_many_matches_individual_reads(backend):
    """A batched read must return what the same reads would return."""
    for key, value in BATCH_ITEMS.items():
        backend.serialize(key, value)

    keys = sorted(BATCH_ITEMS)
    result = backend.read_many(keys)

    assert isinstance(result, dict), f"read_many returned {type(result).__name__}"
    assert sorted(result) == keys, (
        f"read_many returned keys {sorted(result)}, asked for {keys}"
    )
    for key in keys:
        assert_equal_value(result[key], backend.deserialize(key))


@pytest.mark.xfail(strict=True, reason="read_many is a contract stub (issue 4.1)")
def test_read_many_is_not_quieter_than_deserialize(backend):
    """A missing key in a batch must not be silently dropped.

    ``deserialize`` raises ``KeyError``, so a batch that omitted the key instead
    would leave the caller unable to tell "absent" from "stored as None" -- and
    would make the batch path the lossy one to use.
    """
    backend.serialize("present", "v")

    with pytest.raises(KeyError):
        backend.read_many(["present", "absent"])


@pytest.mark.xfail(strict=True, reason="delete_many is a contract stub (issue 4.1)")
def test_delete_many_matches_individual_deletes(backend):
    """A batched delete must remove exactly the keys named, and no others."""
    for key, value in BATCH_ITEMS.items():
        backend.serialize(key, value)

    doomed = sorted(BATCH_ITEMS)[:3]
    survivors = sorted(BATCH_ITEMS)[3:]
    backend.delete_many(doomed)

    assert sorted(backend.keys()) == survivors, (
        f"delete_many left {sorted(backend.keys())}, expected {survivors}"
    )
    for key in survivors:
        assert_equal_value(backend.deserialize(key), BATCH_ITEMS[key])


@pytest.mark.xfail(strict=True, reason="has is a contract stub (issue 2.2)")
def test_has_agrees_with_keys(backend):
    """``has`` must answer what ``keys`` reports, without enumerating it.

    The point of ``has`` is that ``__contains__`` stops being a linear scan
    (issue 2.2); a version that answers correctly by calling ``keys()`` would
    satisfy this guard, which is why ``test_membership_cost_does_not_grow_with_the_store``
    counts the enumeration separately.
    """
    for key, value in BATCH_ITEMS.items():
        backend.serialize(key, value)

    for key in BATCH_ITEMS:
        assert backend.has(key) is True, f"has({key!r}) said False for a stored key"
    assert backend.has("absent") is False, "has() said True for a key never written"


@pytest.mark.xfail(strict=True, reason="count is a contract stub (issue 2.2)")
def test_count_agrees_with_keys(backend):
    """``count`` must agree with ``len(keys())`` before and after a removal.

    Removal goes through ``del_item``, which is the single-key name the backends
    actually have -- the batch method added in 0.1 is ``delete_many``, so the
    pair is spelled inconsistently. Worth aligning in issue 4.1; pinned here
    against the name that exists so this guard fails on ``count`` rather than on
    an attribute error.
    """
    for key, value in BATCH_ITEMS.items():
        backend.serialize(key, value)
    assert backend.count() == len(BATCH_ITEMS), (
        f"count() said {backend.count()} with {len(BATCH_ITEMS)} keys written"
    )
    assert backend.count() == len(backend.keys()), (
        f"count() said {backend.count()}, keys() reports {len(backend.keys())}"
    )

    backend.del_item(sorted(BATCH_ITEMS)[0])
    assert backend.count() == len(backend.keys()) == len(BATCH_ITEMS) - 1, (
        f"after one removal count() said {backend.count()} and keys() reports "
        f"{len(backend.keys())}; expected {len(BATCH_ITEMS) - 1}"
    )
