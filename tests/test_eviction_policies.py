"""Specs for the eviction policies as pure bookkeeping -- no disk, no values.

Written against the ``EvictionPolicy`` contract declared in issue 0.1 and
implemented in issue 1.1. Today's ``*Replacement`` classes cannot satisfy these:
they take a backend in ``__init__`` and write to it during eviction, which is
exactly the coupling 1.1 removes.

**These tests pin the concrete class names too.** Issue 0.1 declared the
``EvictionPolicy`` base but named no implementations, so this spec fixes them as
``effidict.policies.<NAME>Policy`` -- ``FIFOPolicy``, ``LRUPolicy`` and so on.
Cheap to change in review; the point is that it is decided somewhere rather than
being improvised in 1.1.

The semantics being pinned:

* ``on_insert(key)`` starts tracking ``key``: frequency 1, newest in insertion
  order, and counts as its first use.
* ``on_access(key)`` increments its frequency and makes it the most recent use.
* ``victim()`` *peeks* at the next key to evict and raises ``KeyError`` when
  nothing is tracked. The cache calls ``on_remove`` once it has actually evicted,
  so a victim sequence is produced by alternating the two.
* Ties are broken by insertion order, so every policy is deterministic.
"""

from __future__ import annotations

import inspect

import pytest

POLICY_NAMES = ["Random", "FIFO", "LIFO", "LRU", "MRU", "LFU", "MFU"]

DETERMINISTIC = [name for name in POLICY_NAMES if name != "Random"]


def _policy_class(name):
    """Resolve ``effidict.policies.<name>Policy``, or explain what is missing.

    Resolved at call time rather than imported at module scope: a top-level
    import of names that do not exist yet would be a collection error for the
    whole file instead of a clean xfail per test.
    """
    import effidict.policies as policies

    attribute = f"{name}Policy"
    cls = getattr(policies, attribute, None)
    if cls is None:
        raise NotImplementedError(
            f"effidict.policies.{attribute} does not exist yet (issue 1.1)"
        )
    return cls


def _apply(policy, ops):
    for action, key in ops:
        if action == "insert":
            policy.on_insert(key)
        elif action == "access":
            policy.on_access(key)
        else:  # pragma: no cover - typo guard
            raise AssertionError(f"unknown op {action!r}")


def _victim_sequence(policy):
    """Drain the policy, returning the order in which it gives keys up.

    Exercises ``victim()``, ``on_remove()`` and the empty-case ``KeyError`` in one
    pass, which is how the cache will actually drive it.
    """
    order = []
    for _ in range(50):
        try:
            key = policy.victim()
        except KeyError:
            return order
        order.append(key)
        policy.on_remove(key)
    raise AssertionError("victim() never raised KeyError -- policy did not drain")


# Two scenarios: one that only varies recency, one that varies frequency. Each
# expected sequence is written out in full rather than derived, so a wrong answer
# shows up as a diff against a literal.
SCENARIOS = {
    # inserts a, b, c then touches a
    "recency": {
        "ops": [
            ("insert", "a"),
            ("insert", "b"),
            ("insert", "c"),
            ("access", "a"),
        ],
        "expected": {
            "FIFO": ["a", "b", "c"],
            "LIFO": ["c", "b", "a"],
            "LRU": ["b", "c", "a"],
            "MRU": ["a", "c", "b"],
            "LFU": ["b", "c", "a"],
            "MFU": ["a", "b", "c"],
        },
    },
    # inserts a, b, c then touches b twice and c once -> freqs a=1, b=3, c=2
    "frequency": {
        "ops": [
            ("insert", "a"),
            ("insert", "b"),
            ("insert", "c"),
            ("access", "b"),
            ("access", "b"),
            ("access", "c"),
        ],
        "expected": {
            "FIFO": ["a", "b", "c"],
            "LIFO": ["c", "b", "a"],
            "LRU": ["a", "b", "c"],
            "MRU": ["c", "b", "a"],
            "LFU": ["a", "c", "b"],
            "MFU": ["b", "c", "a"],
        },
    },
}


@pytest.mark.xfail(
    strict=True,
    reason="the EvictionPolicy implementations do not exist yet (issue 1.1)",
)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("name", DETERMINISTIC)
def test_victim_order(name, scenario):
    """Each policy gives up keys in one specific, literal order."""
    ops = SCENARIOS[scenario]["ops"]
    expected = SCENARIOS[scenario]["expected"][name]

    policy = _policy_class(name)()
    _apply(policy, ops)

    assert _victim_sequence(policy) == expected


@pytest.mark.xfail(
    strict=True,
    reason="the EvictionPolicy implementations do not exist yet (issue 1.1)",
)
def test_lfu_evicts_the_least_frequently_used():
    """LFU gives up the coldest key first, regardless of when it arrived."""
    policy = _policy_class("LFU")()
    for key in ("hot", "warm", "cold"):
        policy.on_insert(key)
    for _ in range(5):
        policy.on_access("hot")
    policy.on_access("warm")

    assert policy.victim() == "cold"


@pytest.mark.xfail(
    strict=True,
    reason="the EvictionPolicy implementations do not exist yet (issue 1.1)",
)
def test_mfu_evicts_the_most_frequently_used():
    """MFU gives up the hottest key first.

    Regression pin for the duplicated ``MFUReplacement`` class body, whose
    surviving ``put`` resets ``secondary_memory[key]`` to 1 on every write. That
    makes the frequency count meaningless, so the policy does not actually
    evict by frequency at all.
    """
    policy = _policy_class("MFU")()
    for key in ("hot", "warm", "cold"):
        policy.on_insert(key)
    for _ in range(5):
        policy.on_access("hot")
    policy.on_access("warm")

    assert policy.victim() == "hot"


@pytest.mark.xfail(
    strict=True,
    reason="the EvictionPolicy implementations do not exist yet (issue 1.1)",
)
@pytest.mark.parametrize("name", POLICY_NAMES)
def test_policy_never_touches_the_backend(name):
    """A policy must have no way to reach storage.

    Asserted structurally, because 'does not write to disk' is not observable
    from the outside once the coupling exists: today every ``*Replacement`` takes
    a backend and serializes its own victims, which is what makes them impossible
    to unit-test without a filesystem.
    """
    cls = _policy_class(name)

    parameters = set(inspect.signature(cls.__init__).parameters) - {"self"}
    forbidden = {"disk_backend", "backend", "storage", "store"}
    assert not (parameters & forbidden), (
        f"{cls.__name__}.__init__ accepts {sorted(parameters & forbidden)}"
    )

    policy = cls()
    policy.on_insert("a")
    attributes = {
        name: value
        for name, value in vars(policy).items()
        if hasattr(value, "serialize") or hasattr(value, "deserialize")
    }
    assert not attributes, f"{cls.__name__} holds a backend-like object: {sorted(attributes)}"


@pytest.mark.xfail(
    strict=True,
    reason="the EvictionPolicy implementations do not exist yet (issue 1.1)",
)
@pytest.mark.parametrize("name", POLICY_NAMES)
def test_forget_removes_all_policy_state(name):
    """``on_remove`` must drop every trace of a key.

    LFU and MFU keep a second dict of frequency counts alongside the ordering.
    Removing a key from one and not the other leaves a phantom entry that can be
    selected as a victim later, when the cache no longer holds it.
    """
    policy = _policy_class(name)()
    for key in ("a", "b"):
        policy.on_insert(key)
    policy.on_access("a")
    policy.on_access("a")

    policy.on_remove("a")

    assert policy.victim() == "b", "a survived on_remove and was chosen as victim"

    policy.on_remove("b")
    with pytest.raises(KeyError):
        policy.victim()

    leftovers = {
        attribute: value
        for attribute, value in vars(policy).items()
        if isinstance(value, (dict, set, list)) and value
    }
    assert not leftovers, f"state survived removal of every key: {leftovers}"


@pytest.mark.xfail(
    strict=True,
    reason="the EvictionPolicy implementations do not exist yet (issue 1.1)",
)
def test_random_policy_only_ever_names_tracked_keys():
    """Random cannot be pinned to a sequence, so pin what must still hold.

    Every victim is a key the policy is tracking, the policy drains completely,
    and over enough draws every key is reachable -- otherwise it is not random,
    it is just unpredictable.
    """
    cls = _policy_class("Random")
    keys = {"a", "b", "c", "d"}

    chosen = set()
    for _ in range(200):
        policy = cls()
        for key in sorted(keys):
            policy.on_insert(key)
        victim = policy.victim()
        assert victim in keys, f"victim {victim!r} was never tracked"
        chosen.add(victim)

    assert chosen == keys, f"never selected {sorted(keys - chosen)} in 200 draws"

    policy = cls()
    for key in sorted(keys):
        policy.on_insert(key)
    assert sorted(_victim_sequence(policy)) == sorted(keys)


@pytest.mark.xfail(
    strict=True,
    reason="the EvictionPolicy implementations do not exist yet (issue 1.1)",
)
@pytest.mark.parametrize("name", POLICY_NAMES)
def test_clear_drops_everything(name):
    """``clear()`` must leave the policy indistinguishable from a fresh one."""
    policy = _policy_class(name)()
    for key in ("a", "b", "c"):
        policy.on_insert(key)
    policy.on_access("a")

    policy.clear()

    with pytest.raises(KeyError):
        policy.victim()
