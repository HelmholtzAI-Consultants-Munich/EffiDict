"""Specs for the mapping surface, pinned explicitly.

The differential oracle in issue 0.3 catches behavioural drift across random
operation sequences; these state the individual guarantees so a failure names one
thing. ``test_iteration_order_is_insertion_order`` in particular is the
deterministic pin for issue 4.2 that let the oracle's order machine drop to
``strict=False`` -- whether that machine sees the defect depends on hypothesis
generating a case where set-union order happens to differ, which is not reliable
enough for a strict marker.

Values and keys are plain strings, so nothing here can fail for a backend-fidelity
or keyspace reason.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import MutableMapping
from pathlib import Path

import pytest

from .helpers import PROMOTING_POLICIES

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
def test_iter_returns_independent_iterators(backend_cls, policy_cls, make_dict):
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
def test_nested_iteration(backend_cls, policy_cls, make_dict):
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
    from effidict import EffiDict, {backend}, FIFOReplacement

    backend = {backend}({path!r})
    store = EffiDict(
        disk_backend=backend,
        replacement_strategy=FIFOReplacement(disk_backend=backend, max_in_memory=4),
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
def test_mutablemapping_surface_is_complete(backend_cls, policy_cls, make_dict):
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


# --------------------------------------------------------------------------
# equality and hashing
# --------------------------------------------------------------------------


def test_eq_does_not_mutate_either_operand(
    request, backend_cls, policy_cls, make_dict, backend_spy
):
    """Comparing two stores must leave both exactly as they were.

    ``__eq__`` reads through ``items()``, so on a promoting policy it pulls disk
    keys into the cache and evicts others -- an observation that changes what it
    observes.

    Cache membership alone is too weak to catch this. MRU promotes a key and then
    immediately evicts it again, because its victim is the entry just inserted, so
    the set of resident keys is unchanged while a disk write has still happened
    mid-comparison. The backend spy is what actually detects it, and the xfail is
    scoped at runtime to the promoting policies -- Random, FIFO and LIFO read
    straight through without caching and already satisfy this.
    """
    if policy_cls in PROMOTING_POLICIES:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "__eq__ walks items(), which promotes every key through the "
                    "policy and writes evicted entries back out, so comparing "
                    "mutates both operands (issue 6.2)"
                ),
            )
        )

    left = make_dict(max_in_memory=3)
    right = make_dict(max_in_memory=3)
    for key in ORDERED_KEYS[:8]:
        left[key] = key
        right[key] = key

    before_left = set(left.replacement_strategy.memory)
    before_right = set(right.replacement_strategy.memory)

    with backend_spy(left.disk_backend) as spy:
        assert left == right

        assert spy.count("serialize") == 0, (
            f"comparing issued {spy.count('serialize')} writes to the left operand"
        )

    assert set(left.replacement_strategy.memory) == before_left, (
        "comparing changed the left operand's cache contents"
    )
    assert set(right.replacement_strategy.memory) == before_right, (
        "comparing changed the right operand's cache contents"
    )


def test_unhashable_by_design(backend_cls, policy_cls, make_dict):
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
