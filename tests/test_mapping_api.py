"""Specs for the mapping surface, pinned explicitly.

The differential oracle in issue 0.3 catches behavioural drift across random
operation sequences; these state the individual guarantees so a failure names one
thing. ``test_iteration_order_is_insertion_order`` in particular is the
deterministic pin for issue 4.2 that let the oracle's order machine drop to
``strict=False`` -- whether that machine sees the defect depends on hypothesis
generating a case where set-union order happens to differ, which is not reliable
enough for a strict marker.

Values and keys are plain strings, so nothing here can fail for a backend-fidelity
or keyspace reason. That is also why most specs here take neither matrix fixture:
the mapping surface is plain Python semantics, and running it over all 28
backend/policy combinations asserts the same thing 28 times. The exceptions take
the fixtures they actually read -- iteration order depends on which keys sit in
which tier, and the purity spec branches on whether the policy promotes.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
from collections.abc import MutableMapping
from pathlib import Path

import pytest

from effidict import EffiDict

from .helpers import (
    EAGER_ITERATOR_TYPES,
    cached_keys,
    in_cache,
    on_disk,
)

REPO_ROOT = str(Path(__file__).resolve().parents[1])

ORDERED_KEYS = [f"k{i:02d}" for i in range(12)]


# --------------------------------------------------------------------------
# iteration
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "__iter__ returns self and stores its cursor in a single shared "
        "self._iter_keys, so every iterator is the same object (issue 6.2)"
    ),
)
def test_iter_returns_independent_iterators(make_dict):
    """Each call to ``iter()`` must hand back a fresh, independent cursor."""
    d = make_dict(max_in_memory=10)
    for key in "abcd":
        d[key] = key

    first = iter(d)
    second = iter(d)

    assert first is not second, "iter(d) returned the same object twice"
    assert next(first) == next(second), "independent iterators disagreed on the first key"
    assert list(first) == list(second), "advancing one iterator moved the other"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EffiDict is its own iterator, so the inner loop exhausts the shared "
        "cursor and the outer loop stops early (issue 6.2)"
    ),
)
def test_nested_iteration(make_dict):
    """Nested loops must produce the full cross product.

    Silently truncating is worse than raising: four keys yield four pairs instead
    of sixteen, and nothing signals that the loop ended early.
    """
    d = make_dict(max_in_memory=10)
    for key in "abcd":
        d[key] = key

    pairs = [(outer, inner) for outer in d for inner in d]

    assert len(pairs) == 16, f"nested iteration produced {len(pairs)} pairs, expected 16"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "keys() returns list(set(memory) | set(disk)), so order is whatever set "
        "iteration gives (issue 4.2)"
    ),
)
def test_iteration_order_is_insertion_order(backend_cls, policy_cls, make_dict):
    """Iteration must yield keys in the order they were first written.

    Uses twelve keys deliberately: with two or three, set-union order coincides
    with insertion order often enough to make the assertion unreliable. At twelve
    the chance of an accidental match is negligible, which is what makes this a
    safe ``strict=True`` pin where the oracle's order machine is not.
    """
    d = make_dict(max_in_memory=4)
    for key in ORDERED_KEYS:
        d[key] = key

    assert list(d.keys()) == ORDERED_KEYS
    assert list(d) == ORDERED_KEYS
    assert [key for key, _ in d.items()] == ORDERED_KEYS


_ORDER_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {repo!r})
    from effidict import EffiDict, {backend}, FIFOPolicy

    backend = {backend}({path!r})
    store = EffiDict(
        disk_backend=backend,
        policy=FIFOPolicy(),
        max_in_memory=4,
    )
    for key in {keys!r}:
        store[key] = key
    print(",".join(store.keys()))
    """
)

HASH_SEEDS = ["0", "1", "42"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "key order comes from set iteration, which is salted by PYTHONHASHSEED, "
        "so it changes between processes (issue 4.2)"
    ),
)
def test_iteration_order_is_stable_across_processes(backend_cls, tmp_path):
    """The same writes must iterate identically in any process.

    Runs real subprocesses under three ``PYTHONHASHSEED`` values. String hashing
    is salted per process, so anything derived from set iteration order changes
    run to run -- which makes stored key order unreproducible for a caller who
    reasonably expects a mapping to be deterministic.
    """
    observed = {}
    for seed in HASH_SEEDS:
        script = _ORDER_SCRIPT.format(
            repo=REPO_ROOT,
            backend=backend_cls.__name__,
            path=str(tmp_path / f"store-{seed}"),
            keys=ORDERED_KEYS,
        )
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert result.returncode == 0, (
            f"seed {seed} subprocess failed:\n{result.stderr.strip()[-1000:]}"
        )
        observed[seed] = result.stdout.strip().split(",")

    distinct = {tuple(order) for order in observed.values()}
    assert len(distinct) == 1, (
        "key order differed between processes:\n"
        + "\n".join(f"  PYTHONHASHSEED={seed}: {order}" for seed, order in observed.items())
    )
    assert observed[HASH_SEEDS[0]] == ORDERED_KEYS, "order was stable but not insertion order"


# --------------------------------------------------------------------------
# surface
# --------------------------------------------------------------------------

MAPPING_METHODS = [
    "get",
    "setdefault",
    "pop",
    "popitem",
    "update",
    "clear",
    "keys",
    "values",
    "items",
    "__contains__",
    "__iter__",
    "__len__",
    "__reversed__",
    "__or__",
    "fromkeys",
]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EffiDict is not registered as a MutableMapping and get/setdefault/"
        "popitem/update raise NotImplementedError (issue 6.1)"
    ),
)
def test_mutablemapping_surface_is_complete(make_dict):
    """The full mapping surface must be present and working.

    Registering as a ``MutableMapping`` is what makes conformance structural
    rather than a list somebody has to remember to extend.
    """
    d = make_dict(max_in_memory=10)
    d["a"] = "va"

    assert isinstance(d, MutableMapping), (
        "EffiDict is not a MutableMapping, so the mapping surface has to be "
        "hand-maintained instead of inherited"
    )

    missing = [name for name in MAPPING_METHODS if not hasattr(d, name)]
    assert not missing, f"missing from the mapping surface: {missing}"

    # Present is not enough -- the headline request was a working get().
    assert d.get("a") == "va"
    assert d.get("absent") is None
    assert d.get("absent", "fallback") == "fallback"
    assert d.setdefault("b", "vb") == "vb"
    assert d["b"] == "vb"
    d.update({"c": "vc"})
    assert d["c"] == "vc"
    key, value = d.popitem()
    assert key not in d and value is not None


def test_update_mapping_parameter_is_positional_only():
    """``update`` must accept ``other`` as a key, the way ``dict`` does.

    The expectation is taken from ``dict`` itself rather than asserted as a
    literal, since ``dict`` is the contract this suite compares against. A
    positional-or-keyword parameter named ``other`` silently swallows that key,
    and the oracle will not find it: ``update_from_keywords`` draws its key from
    a 27-character alphabet, so generating exactly ``"other"`` is a ~1-in-14-
    million event per draw. A signature check is the only guard that fires.
    """
    reference = {}
    reference.update(other=5)
    assert reference == {"other": 5}, "dict.update stores 'other' as a key"

    parameter = inspect.signature(EffiDict.update).parameters["other"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_ONLY, (
        f"EffiDict.update takes 'other' as {parameter.kind.description}, so "
        f"update(other=...) binds the mapping parameter instead of storing the key"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "__reversed__, __or__ and fromkeys raise NotImplementedError; the three "
        "entries in MAPPING_METHODS were only ever hasattr-checked (issue 6.1). "
        "They are declared stubs now rather than missing attributes, which is "
        "itself the fix for a sharper problem: with no __reversed__ at all, "
        "reversed() fell back to the sequence protocol and subscripted the store "
        "with integers, so the caller got a TypeError from inside os.path.join "
        "on Pickle/HDF5 or a KeyError on SQLite/JSON instead of anything about "
        "reversal"
    ),
)
def test_dict_extras_behave_like_dict(make_dict):
    """The three non-``Mapping`` entries in the surface list must actually work.

    ``MAPPING_METHODS`` claims ``__reversed__``, ``__or__`` and ``fromkeys``, but
    the surface test above only checks they exist and the differential oracle
    never calls them -- so a stub, or a wrong answer, satisfied the "complete
    mapping surface" spec. None of the three come from ``MutableMapping``; they
    are ``dict`` features, which is exactly why inheriting the ABC does not
    supply them and why they need exercising rather than listing.

    All three are declared stubs on the facade, so this now fails with an
    issue-linked ``NotImplementedError``. Before they were declared, the
    failures came from wherever Python's fallbacks happened to land -- which is
    the state this spec was written against and no longer describes.

    Each is compared against ``dict``'s own answer. ``fromkeys`` is asserted on
    *contents* only: where the new store's storage lives is a real open question
    (``clone`` takes a path, ``dict.fromkeys`` takes none) and belongs to issue
    6.1, so pinning a signature here would be inventing a decision.
    """
    d = make_dict(max_in_memory=10)
    reference = {}
    for index, key in enumerate(["a", "b", "c"]):
        d[key] = f"v{index}"
        reference[key] = f"v{index}"

    assert list(reversed(d)) == list(reversed(reference)), (
        f"reversed() gave {list(reversed(d))}, dict gives "
        f"{list(reversed(reference))}"
    )

    merged = d | {"c": "override", "e": "new"}
    expected = reference | {"c": "override", "e": "new"}
    assert dict(merged) == expected, f"| gave {dict(merged)}, dict gives {expected}"
    assert dict(d) == reference, "| mutated the left operand"

    filled = d.fromkeys(["x", "y"], "z")
    assert dict(filled) == dict.fromkeys(["x", "y"], "z"), (
        f"fromkeys gave {dict(filled)}, dict gives {dict.fromkeys(['x', 'y'], 'z')}"
    )


# --------------------------------------------------------------------------
# equality and hashing
# --------------------------------------------------------------------------


#: ``(other, expected)`` pairs, where ``expected`` is what a plain ``dict``
#: holding ``BASE`` answers. Comparing against ``dict``'s own answer rather than
#: a hand-written True/False keeps the table honest.
BASE = {"x": "1", "y": "2"}
EQUALITY_CASES = [
    ("equal mapping", dict(BASE)),
    ("different value", {"x": "1", "y": "9"}),
    ("missing key", {"x": "1"}),
    ("extra key", {"x": "1", "y": "2", "z": "3"}),
    ("empty mapping", {}),
    ("not a mapping", 42),
    ("none", None),
    ("list of pairs", [("x", "1"), ("y", "2")]),
]


@pytest.mark.parametrize("label, other", EQUALITY_CASES, ids=[c[0] for c in EQUALITY_CASES])
def test_equality_matches_dict(request, label, other, make_dict):
    """``==`` must give the answer a ``dict`` would, against any operand.

    Nothing pinned this. The two existing ``__eq__`` specs cover *purity* -- that
    comparing does not mutate either operand -- and unhashability; neither looks
    at what the comparison returns, so an ``__eq__`` answering False for
    identical contents satisfied the whole "complete mapping surface" claim.

    That it must equal a plain ``dict`` is not an invention: this suite's premise
    is that an ``EffiDict`` is indistinguishable from a ``dict``, the
    differential oracle asserts exactly that for every other operation, and
    issue 6.1 registers the class as a ``MutableMapping``. A type check that
    rejects every other mapping is the deviation, and it is the one defect this
    case table exposes -- the other seven rows already agree with ``dict``.
    """
    expected = BASE == other
    if expected:
        # Marked per case, not on the parametrization. The type check only shows
        # up where equality *should* hold: on the seven rows that compare
        # unequal, returning False happens to be the right answer, so they pass
        # today and are live guards. Blanket-xfailing them would have reported
        # seven XPASSes and said nothing about the one real defect.
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "__eq__ opens with `if not isinstance(other, EffiDict): "
                    "return False`, so a store never equals a plain mapping "
                    "holding the same items (issue 6.1)"
                ),
            )
        )

    d = make_dict(max_in_memory=10)
    for key, value in BASE.items():
        d[key] = value

    assert (d == other) is expected, (
        f"{label}: EffiDict == {other!r} gave {d == other!r}, dict gives "
        f"{expected!r}"
    )
    assert (d != other) is (not expected), (
        f"{label}: != disagrees with == (got {d != other!r})"
    )
    # Reflected: Python tries dict.__eq__ first, gets NotImplemented, then asks
    # EffiDict. A store that only answers when it is on the left is still wrong.
    assert (other == d) is expected, (
        f"{label}: {other!r} == EffiDict gave {other == d!r}, dict gives "
        f"{expected!r}"
    )


def test_equality_between_stores_is_by_contents(make_dict):
    """Two stores holding the same items must compare equal.

    Already true, and the half of ``__eq__`` that works; pinned so the fix for
    the dict case above cannot regress it.
    """
    left = make_dict(max_in_memory=10)
    right = make_dict(max_in_memory=10)
    for key, value in BASE.items():
        left[key] = value
        right[key] = value

    assert left == right, "two stores with identical contents compared unequal"
    assert left == left, "a store did not equal itself"

    right["y"] = "changed"
    assert left != right, "stores with differing contents compared equal"


def test_eq_does_not_mutate_either_operand(
    backend_cls, policy_cls, make_dict, backend_spy
):
    """Comparing two stores must leave both exactly as they were.

    ``__eq__`` reads through ``items()``, which on a store that pulled disk
    keys into the cache as it read them would evict others -- an observation
    that changes what it observes.

    Cache membership alone is too weak to catch that. MRU promotes a key and
    then immediately evicts it again, because its victim is the entry just
    inserted, so the set of resident keys is unchanged while a disk write has
    still happened mid-comparison. The backend spy is what actually detects it.

    Every policy satisfies this since issue 1.2, which stopped ``Store.get``
    promoting. The runtime xfail that scoped this to the promoting policies
    went with them: the set it tested has been empty ever since, so it could
    never fire.
    """
    left = make_dict(max_in_memory=3)
    right = make_dict(max_in_memory=3)
    for key in ORDERED_KEYS[:8]:
        left[key] = key
        right[key] = key

    before_left = cached_keys(left)
    before_right = cached_keys(right)

    # Both operands are spied. __eq__ walks self.items() *and* reads other[key],
    # so the right-hand store is promoted too -- watching only the left would let
    # a mutation of the right slip past the assertion this test makes.
    with backend_spy(left._store.backend) as left_spy:
        with backend_spy(right._store.backend) as right_spy:
            assert left == right

            left_writes = left_spy.count("serialize") + left_spy.count("write_many")
            assert left_writes == 0, (
                f"comparing issued {left_writes} writes to the "
                f"left operand"
            )
            right_writes = right_spy.count("serialize") + right_spy.count("write_many")
            assert right_writes == 0, (
                f"comparing issued {right_writes} writes to the "
                f"right operand"
            )

    assert cached_keys(left) == before_left, (
        "comparing changed the left operand's cache contents"
    )
    assert cached_keys(right) == before_right, (
        "comparing changed the right operand's cache contents"
    )


def test_unhashable_by_design(make_dict):
    """A mutable mapping must not be hashable.

    Already true, and pinned so it stays true: ``EffiDict`` defines ``__eq__``,
    which makes Python set ``__hash__`` to ``None`` on the class. That is the
    right outcome, but it is a side effect rather than a decision, so it is the
    kind of thing a later refactor can silently undo.
    """
    from effidict import EffiDict

    d = make_dict()

    assert EffiDict.__hash__ is None
    with pytest.raises(TypeError):
        hash(d)
    with pytest.raises(TypeError):
        {d: "not allowed"}


@pytest.mark.xfail(
    strict=True,
    reason="EffiDict.iter_keys() is a contract stub (issue 4.2)",
)
def test_facade_iter_keys_streams_the_whole_keyspace(make_dict, backend_spy):
    """``d.iter_keys()`` must stream every key, across both tiers.

    The facade method had no test of its own. ``DiskBackend.iter_keys`` is
    specified in ``test_backend_contract``, ``keys()`` and ``iter(d)`` here --
    all three could be fixed while this public method stayed a stub, because
    nothing called it.

    It is also the only one of the four that has to merge tiers: the backend
    sees what spilled, the cache holds the rest, and the caller must get the
    union exactly once each. The cache is kept smaller than the keyspace on
    purpose so both tiers are non-empty.
    """
    d = make_dict(max_in_memory=5)
    expected = [f"k{index:02d}" for index in range(20)]
    for key in expected:
        d[key] = f"v-{key}"

    # One key put in *both* tiers on purpose. Without this the de-duplication
    # assertion below is dead: nothing is flushed today, so the cached keys are
    # not on disk and the two tiers happen not to overlap -- an implementation
    # that simply concatenated them would pass. Dual residency is the intended
    # steady state once I1 holds (disk authoritative, not exclusive), so the
    # union has to be a union rather than a concatenation.
    dual = expected[-1]
    assert in_cache(d, dual), "precondition: the last write is still cached"
    d._store.backend.serialize(dual, f"v-{dual}")
    assert on_disk(d, dual), "precondition: the probe key is now in both tiers"

    with backend_spy(d._store.backend) as spy:
        stream = d.iter_keys()

        assert iter(stream) is stream, (
            f"iter_keys returned {type(stream).__name__}, not an iterator"
        )
        assert type(stream).__name__ not in EAGER_ITERATOR_TYPES, (
            f"iter_keys returned a {type(stream).__name__}, so the keyspace was "
            f"materialised before iteration began"
        )
        assert spy.names() == [], (
            f"creating the iterator already went to the backend: {spy.names()}"
        )

        streamed = list(stream)

    assert sorted(streamed) == sorted(expected), (
        f"iter_keys yielded {len(streamed)} keys, expected {len(expected)}"
    )
    assert len(streamed) == len(set(streamed)), (
        f"a key present in both tiers was yielded twice: "
        f"{sorted(k for k in set(streamed) if streamed.count(k) > 1)}"
    )
