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
from collections import abc

import pytest

POLICY_NAMES = ["Random", "FIFO", "LIFO", "LRU", "MRU", "LFU", "MFU"]

DETERMINISTIC = [name for name in POLICY_NAMES if name != "Random"]

#: Constructor parameters a policy may accept. Empty on purpose: bookkeeping
#: needs no configuration, and an empty allowlist is the only version of this
#: check that cannot be walked around. If a policy ever needs something -- a
#: ``seed`` for Random, say -- that is a deliberate amendment here, not an
#: incidental widening.
ALLOWED_INIT_PARAMS = frozenset()


#: Distinctive key names. Single letters would make the repr check below
#: unusable: "a" appears in "defaultdict", so an empty defaultdict attribute
#: would read as a leak.
KEY_ALPHA = "alpha-9f2c"
KEY_BETA = "beta-9f2c"


def _residual_state(policy):
    """Attributes still holding anything after every key has been removed.

    Type-agnostic on purpose. Checking ``dict``/``set``/``list`` by name would
    miss ``collections.deque`` -- the obvious way to implement FIFO and LIFO
    ordering -- along with tuple, frozenset and any custom container, so a policy
    could retain every key it was told to forget and this test would still pass.
    ``Sized`` covers all of those and anything else defining ``__len__``.
    """
    residual = {}
    for attribute, value in vars(policy).items():
        if isinstance(value, (str, bytes)):
            continue
        if isinstance(value, abc.Sized) and len(value) > 0:
            residual[attribute] = value
    return residual



def _reachable_keys(root, targets, max_depth=8):
    """Which ``targets`` are still reachable from ``root``.

    Walks mappings, sequences, sets and plain object ``__dict__``s. The Sized
    check above misses state with no ``__len__`` -- a hand-rolled linked list of
    nodes, for instance -- and a repr scan misses it too, because a custom class
    with the default repr never shows the key it holds. Reachability is the only
    formulation that does not depend on how the state is shaped.

    Deliberately does not iterate arbitrary iterables: consuming a generator
    would mutate the thing under test.
    """
    found = set()
    seen = set()
    stack = [(root, 0)]
    while stack:
        obj, depth = stack.pop()
        if depth > max_depth or id(obj) in seen:
            continue
        seen.add(id(obj))

        if isinstance(obj, str):
            if obj in targets:
                found.add(obj)
            continue
        if isinstance(obj, (bytes, bytearray, int, float, bool, type(None))):
            continue

        if isinstance(obj, abc.Mapping):
            for key, value in obj.items():
                stack.append((key, depth + 1))
                stack.append((value, depth + 1))
            continue
        if isinstance(obj, (abc.Sequence, abc.Set)):
            for item in obj:
                stack.append((item, depth + 1))
            continue

        state = getattr(obj, "__dict__", None)
        if isinstance(state, dict):
            for value in state.values():
                stack.append((value, depth + 1))
    return found


class _FakeBackend:
    """Quacks like a ``DiskBackend`` so an accidental injection is detectable.

    Every method records itself rather than raising, so a policy that both
    accepts *and uses* a backend is reported with the call it made instead of
    failing somewhere further down the stack.
    """

    def __init__(self):
        self.calls = set()

    def _record(self, name):
        self.calls.add(name)

    def serialize(self, key, value):
        self._record("serialize")

    def deserialize(self, key):
        self._record("deserialize")
        raise KeyError(key)

    def del_item(self, key):
        self._record("del_item")

    def keys(self):
        self._record("keys")
        return []

    def has(self, key):
        self._record("has")
        return False

    def destroy(self):
        self._record("destroy")


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

    Today every ``*Replacement`` takes a backend and serializes its own victims,
    which is what makes them impossible to unit-test without a filesystem.

    Checked four ways, because no single one is sufficient. A denylist of
    parameter names is the weakest: it misses ``*args``/``**kwargs``, which can
    smuggle a backend past any name check, and it misses an unlisted name like
    ``be`` or ``sink``. So the signature checks use an allowlist, and the load
    bearing assertion is the third -- actually attempting the injection and
    demanding it be refused.
    """
    cls = _policy_class(name)
    signature = inspect.signature(cls.__init__)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    }

    # 1. No variadics. Without this the allowlist below means nothing, since
    #    anything at all can be passed through *args/**kwargs. Skipped when the
    #    class does not define __init__ at all: object.__init__ reports
    #    (*args, **kwargs) but refuses every argument, which check 3 confirms.
    if cls.__init__ is not object.__init__:
        variadic = sorted(
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )
        assert not variadic, (
            f"{cls.__name__}.__init__ accepts {variadic}, so a backend can be "
            f"passed regardless of what the parameters are called"
        )

        # 2. Allowlist, not denylist: a policy needs no construction arguments.
        unexpected = sorted(set(parameters) - ALLOWED_INIT_PARAMS)
        assert not unexpected, (
            f"{cls.__name__}.__init__ accepts {unexpected}; a pure bookkeeping "
            f"policy takes no arguments"
        )

    # 3. Attempt the injection. This is what closes the *args/**kwargs loophole:
    #    a policy that quietly swallows and stores a backend fails here even
    #    though its signature looks clean.
    fake = _FakeBackend()
    attempts = {
        "positionally": lambda: cls(fake),
        "as disk_backend=": lambda: cls(disk_backend=fake),
        "as backend=": lambda: cls(backend=fake),
    }
    for description, attempt in attempts.items():
        with pytest.raises(TypeError):
            attempt()
        assert not fake.calls, (
            f"{cls.__name__} accepted a backend {description} and called "
            f"{sorted(fake.calls)} on it"
        )

    # 4. Nothing backend-like survives real use.
    policy = cls()
    policy.on_insert("a")
    policy.on_access("a")
    policy.victim()
    held = sorted(
        attribute
        for attribute, value in vars(policy).items()
        if hasattr(value, "serialize") or hasattr(value, "deserialize")
    )
    assert not held, f"{cls.__name__} holds a backend-like object: {held}"


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
    for key in (KEY_ALPHA, KEY_BETA):
        policy.on_insert(key)
    policy.on_access(KEY_ALPHA)
    policy.on_access(KEY_ALPHA)

    policy.on_remove(KEY_ALPHA)

    assert policy.victim() == KEY_BETA, (
        f"{KEY_ALPHA} survived on_remove and was chosen as victim"
    )

    policy.on_remove(KEY_BETA)
    with pytest.raises(KeyError):
        policy.victim()

    residual = _residual_state(policy)
    assert not residual, f"state survived removal of every key: {residual}"

    # Catch-all for state that is not Sized -- a hand-rolled linked list, say --
    # where the keys stay reachable but len() does not exist.
    reachable = _reachable_keys(policy, {KEY_ALPHA, KEY_BETA})
    assert not reachable, (
        f"{sorted(reachable)} are still reachable from the policy after removal"
    )


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
