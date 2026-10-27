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

import copy
import inspect
from collections import abc

import pytest

from effidict import MFUReplacement

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



def _instance_state(obj):
    """Attribute name -> value for ``obj``, covering ``__dict__`` and ``__slots__``.

    ``vars()`` alone is not enough: a policy declaring ``__slots__`` for
    compactness has no ``__dict__``, and ``vars()`` raises ``TypeError`` on it --
    which would make these specs fail for a reason that has nothing to do with
    state leaking.
    """
    state = {}
    if hasattr(obj, "__dict__"):
        try:
            state.update(vars(obj))
        except TypeError:  # pragma: no cover - defensive
            pass
    for klass in type(obj).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots or ():
            if hasattr(obj, slot):
                state[slot] = getattr(obj, slot)
    return state


def _state_snapshot(obj):
    """A *detached* copy of ``_instance_state``, safe to compare against later.

    ``_instance_state`` hands back the live containers -- ``vars()`` returns the
    policy's own dict and list objects -- so ``before = _instance_state(p)``
    followed by ``assert _instance_state(p) == before`` compares a thing with
    itself and is true however much ``p`` mutated in between. Any "did this
    operation leave the policy alone?" check needs this instead; the only
    before/after comparison in this file that uses the same object was
    vacuous until it did.

    Falls back to ``repr`` for anything that cannot be deep-copied, which still
    detects a change while never failing the spec for a reason unrelated to it.
    """
    snapshot = {}
    for attribute, value in _instance_state(obj).items():
        try:
            snapshot[attribute] = copy.deepcopy(value)
        except Exception:  # noqa: BLE001 - uncopyable: fall back to its repr
            snapshot[attribute] = repr(value)
    return snapshot


def _residual_state(policy):
    """Attributes still holding anything after every key has been removed.

    Type-agnostic on purpose. Checking ``dict``/``set``/``list`` by name would
    miss ``collections.deque`` -- the obvious way to implement FIFO and LIFO
    ordering -- along with tuple, frozenset and any custom container, so a policy
    could retain every key it was told to forget and this test would still pass.
    ``Sized`` covers all of those and anything else defining ``__len__``.
    """
    residual = {}
    for attribute, value in _instance_state(policy).items():
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

        # Pushed as a mapping rather than iterating its values, so attribute
        # *names* are examined too: a policy doing setattr(self, key, ...) keeps
        # every key as an attribute name, where a values-only walk sees nothing.
        state = _instance_state(obj)
        if state:
            stack.append((state, depth + 1))
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


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("name", DETERMINISTIC)
def test_victim_order(name, scenario):
    """Each policy gives up keys in one specific, literal order."""
    ops = SCENARIOS[scenario]["ops"]
    expected = SCENARIOS[scenario]["expected"][name]

    policy = _policy_class(name)()
    _apply(policy, ops)

    assert _victim_sequence(policy) == expected


def test_lfu_evicts_the_least_frequently_used():
    """LFU gives up the coldest key first, regardless of when it arrived."""
    policy = _policy_class("LFU")()
    for key in ("hot", "warm", "cold"):
        policy.on_insert(key)
    for _ in range(5):
        policy.on_access("hot")
    policy.on_access("warm")

    assert policy.victim() == "cold"


def test_mfu_evicts_the_most_frequently_used():
    """MFU gives up the hottest key first.

    States the contract issue 1.1 satisfies, against ``MFUPolicy``. It does not
    reach the duplicated-class-body defect it was originally written for: that
    lives on the legacy ``MFUReplacement``, which 1.1 leaves alone and 1.2
    deletes. ``test_mfu_replacement_keeps_frequency_across_a_value_update``
    below is the pin that does reach it.
    """
    policy = _policy_class("MFU")()
    for key in ("hot", "warm", "cold"):
        policy.on_insert(key)
    for _ in range(5):
        policy.on_access("hot")
    policy.on_access("warm")

    assert policy.victim() == "hot"


class _NullBackend:
    """Absorbs the write an eviction performs. The shipped policies do their own
    I/O, which issue 1.1 removes; until then a victim cannot be observed without
    one."""

    def serialize(self, key, value):
        pass


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MFUReplacement's surviving put() resets secondary_memory[key] to 1, so "
        "a value update discards the key's recorded frequency and the policy "
        "evicts by stale counts (issue 1.1)"
    ),
)
def test_mfu_replacement_keeps_frequency_across_a_value_update():
    """The *shipped* MFU policy must not forget a key's frequency on rewrite.

    Aimed at ``MFUReplacement`` rather than the future ``MFUPolicy``, so it
    actually executes the duplicated class body. Derived from the MFU rule
    stated in this module's docstring, not from the implementation: ``a`` is used
    four times and ``b`` twice, so ``a`` is the hottest key and MFU must give it
    up. Rewriting ``a``'s *value* changes no usage count, so the victim is
    unchanged by it.

    The counts are what discriminate here. With the reset, ``a`` drops to 1
    against ``b``'s 2, and the policy evicts ``b`` -- the coldest key, the exact
    opposite of MFU.
    """
    policy = MFUReplacement(disk_backend=_NullBackend(), max_in_memory=2)
    policy.put("a", "va")
    policy.put("b", "vb")
    for _ in range(3):
        policy.get("a")
    policy.get("b")

    policy.put("a", "va-updated")
    policy.put("c", "vc")

    assert sorted(policy.memory) == ["b", "c"], (
        f"MFU kept {sorted(policy.memory)}; 'a' was the hottest key (4 uses "
        f"against 2) and should have been the victim"
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
        for attribute, value in _instance_state(policy).items()
        if hasattr(value, "serialize") or hasattr(value, "deserialize")
    )
    assert not held, f"{cls.__name__} holds a backend-like object: {held}"


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


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_clear_drops_everything(name):
    """``clear()`` must leave the policy indistinguishable from a fresh one.

    "Indistinguishable" is the claim, so it is what gets asserted. An earlier
    version checked only that ``victim()`` then raised ``KeyError``, which a
    policy that drops its ordering while keeping its frequency counters
    satisfies -- and LFU/MFU would then pick victims from counts belonging to
    keys the cache no longer holds, the whole defect ``clear`` exists to avoid.
    Compared against a fresh instance rather than against a list of attribute
    names, so a counter added later is covered without anyone remembering to
    extend this test.
    """
    cls = _policy_class(name)
    policy = cls()
    for key in ("a", "b", "c"):
        policy.on_insert(key)
    # Access twice: on LFU/MFU this is what separates a frequency table that was
    # reset from one that merely lost its keys.
    policy.on_access("a")
    policy.on_access("a")

    policy.clear()

    with pytest.raises(KeyError):
        policy.victim()

    residual = _residual_state(policy)
    assert not residual, (
        f"{name} kept state after clear(): {residual}"
    )
    # Compared on comparable state only. The residual check above covers
    # containers; this adds plain counters -- a ``self.tick`` left at 57 is not
    # ``Sized``, so nothing else would see it. Attributes that do not compare by
    # value (a lock, a handle) are skipped rather than failing: two fresh ones
    # are never equal, which would make this fail for a reason that has nothing
    # to do with clear().
    comparable = (int, float, bool, str, bytes, type(None), abc.Sized)

    def differs(left, right):
        """``left != right`` as a plain bool, or ``False`` if it is not one.

        An attribute whose ``!=`` returns something elementwise -- an ndarray is
        ``Sized``, so it reaches here -- would raise ValueError on the truth
        test and fail this spec for a reason unrelated to ``clear()``.
        """
        try:
            return bool(left != right)
        except Exception:  # noqa: BLE001 - uncomparable means "cannot judge"
            return False

    cleared_state = _instance_state(policy)
    fresh_state = _instance_state(cls())
    differing = {
        attribute: (value, fresh_state.get(attribute))
        for attribute, value in cleared_state.items()
        if isinstance(value, comparable)
        and isinstance(fresh_state.get(attribute), comparable)
        and differs(value, fresh_state.get(attribute))
    }
    assert not differing, (
        f"{name} after clear() differs from a fresh instance "
        f"(attribute: cleared vs fresh): {differing}"
    )


# --------------------------------------------------------------------------
# enumerating victims without disturbing the policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_victims_enumerates_in_eviction_order(name):
    """``victims()`` must list every tracked key, most-evictable first.

    ``Cache.evict_candidates`` promises several keys in victim order *without*
    mutating, and ``victim()`` cannot supply that: it names one key and only
    advances when that key is removed, so asking twice returns the same answer.
    Removing before the caller has written a dirty value out is what I5 forbids,
    so the enumeration has to exist on the policy -- otherwise a conforming
    ``Cache`` would have to keep a second copy of the ordering.

    Checked against ``victim()`` itself rather than a hardcoded order, so the two
    cannot disagree: repeatedly taking ``victim()`` and removing it produces the
    sequence ``victims()`` must have returned in one pass.
    """
    policy = _policy_class(name)()
    for key in ("a", "b", "c", "d"):
        policy.on_insert(key)
    policy.on_access("b")

    listed = list(policy.victims())

    # Through the bounded helper rather than an inline ``while True``. An
    # ``on_remove`` that fails to deregister the key makes ``victim()`` return
    # it forever: an unbounded loop appends until the machine runs out of
    # memory, so a broken policy hangs the suite instead of failing it.
    # ``_victim_sequence`` stops after 50 and says what went wrong.
    drained = _victim_sequence(policy)

    assert listed == drained, (
        f"{name}.victims() gave {listed}, draining victim() gives {drained}"
    )


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_victims_records_nothing(name):
    """Walking ``victims()`` must leave the policy exactly as it was.

    It is an inspection, not an access: a version that routed through
    ``on_access`` would reorder the very thing the caller is reading, and a
    ``Store`` deciding what to shed would change its answer by asking.
    """
    policy = _policy_class(name)()
    for key in ("a", "b", "c"):
        policy.on_insert(key)
    policy.on_access("a")
    before = _state_snapshot(policy)

    # Partially first. A policy that promoted every key it listed would walk an
    # LRU ordering right back to where it started, so comparing only after a
    # full pass cannot see it -- the promotions cancel. Stopping after one key
    # breaks the symmetry, and stopping early is what a Store does anyway once
    # it is back inside budget.
    sentinel = object()
    partial = next(iter(policy.victims()), sentinel)
    assert partial is not sentinel, (
        f"{name}.victims() was empty while the policy tracked three keys"
    )
    assert partial in ("a", "b", "c"), f"{name}.victims() yielded {partial!r}"
    assert _state_snapshot(policy) == before, (
        f"{name}.victims() changed the policy after a single step:\n"
        f"  before: {before}\n  after:  {_state_snapshot(policy)}"
    )

    listed = list(policy.victims())
    assert sorted(listed) == ["a", "b", "c"], (
        f"{name}.victims() did not list every tracked key: {listed}"
    )

    assert _state_snapshot(policy) == before, (
        f"{name}.victims() changed the policy:\n"
        f"  before: {before}\n  after:  {_state_snapshot(policy)}"
    )
