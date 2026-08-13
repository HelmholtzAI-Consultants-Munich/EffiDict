"""Differential oracle: EffiDict must be indistinguishable from ``dict``.

Every rule applies the same operation to a reference ``dict`` and to an
``EffiDict``, then asserts an identical result *or* an identical exception type.
An ``@invariant`` re-checks the whole contents after every step.

Three deliberate restrictions keep this test measuring what it claims to:

* **Keys are safe strings.** Non-string keys corrupt the store and ``'a/b'`` /
  ``'..'`` / ``''`` raise raw ``OSError`` from the file backends. Those are
  keyspace defects owned by issues 0.8 and 7.1, and letting them in here would
  make the oracle fail for reasons unrelated to the mapping protocol.
* **Values are drawn per backend** from ``helpers.supported_kinds``. HDF5 widens
  ``int`` to ``int64`` and SQLite/JSON turn a tuple into a list; those are
  round-trip fidelity defects owned by issue 0.8. A protocol test can only use
  values the backend stores faithfully.
* **Rules are split into groups**, one machine each, so a known failure can be
  ``xfail(strict=True)`` against the specific finding that causes it. A single
  all-rules machine would collapse five independent defects into one marker and
  give no signal when the first of them is fixed.

The core machine runs the full 4 x 7 matrix at every ``max_in_memory``; the
rule-group machines run the full matrix at one cache size, plus a deep ``slow``
run. See ``_settings``.
"""

from __future__ import annotations

import copy

import pytest
from hypothesis import HealthCheck, settings as hyp_settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    invariant,
    rule,
    run_state_machine_as_test,
)

from effidict import LRUReplacement, PickleBackend

from .conftest import BACKENDS, POLICIES, release_store
from .helpers import assert_equal_value, supported_kinds

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# Lowercase ASCII, digits and underscore only. Deliberately excludes uppercase
# and non-ASCII: on a case-insensitive or normalizing filesystem (macOS APFS,
# Windows NTFS) the file backends collide distinct keys onto one file, so
# d["A"] and d["a"] become a single entry and NFC/NFD spellings of the same
# character merge. That is a real defect owned by issue 7.1 -- see the module
# docstring -- and it is platform-dependent, so it cannot be xfailed here.
KEYS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=8)

# HDF5 rejects NUL (ValueError) and lone surrogates (UnicodeEncodeError) in
# string values; the other three backends accept both. Value-domain finding
# owned by issue 0.8.
_TEXT = st.text(
    alphabet=st.characters(exclude_categories=("Cs",), exclude_characters="\x00"),
    max_size=8,
)
_SCALARS = st.integers(-1000, 1000) | _TEXT | st.none() | st.booleans()

#: One strategy per value kind named in helpers.VALUE_KINDS.
KIND_STRATEGIES = {
    "int": st.integers(-1000, 1000),
    "float": st.floats(allow_nan=False, allow_infinity=False, width=32),
    "bool": st.booleans(),
    "str": _TEXT,
    "none": st.none(),
    "empty_dict": st.just({}),
    "dict": st.dictionaries(KEYS, st.integers(-100, 100), max_size=3),
    "nested_dict": st.dictionaries(
        KEYS, st.lists(st.integers(-100, 100), max_size=3), max_size=2
    ),
    "list": st.lists(_SCALARS, max_size=4),
    "tuple": st.lists(st.integers(-100, 100), max_size=3).map(tuple),
    # ndarray/dataframe are handled by _values(): they need numpy/pandas and are
    # only faithful on PickleBackend (and ndarray on HDF5).
}


def _values(backend_cls):
    """Value strategy limited to kinds ``backend_cls`` round-trips faithfully."""
    faithful = supported_kinds(backend_cls)
    strategies = [
        KIND_STRATEGIES[kind] for kind in sorted(faithful & set(KIND_STRATEGIES))
    ]

    # Plain imports rather than importorskip: this runs inside a hypothesis
    # example, where raising Skipped would be reported as an error.
    if "ndarray" in faithful and np is not None:
        strategies.append(
            st.lists(st.integers(-100, 100), min_size=1, max_size=4).map(np.array)
        )
    if "dataframe" in faithful and pd is not None:
        strategies.append(
            st.lists(st.integers(-100, 100), min_size=1, max_size=3).map(
                lambda col: pd.DataFrame({"a": col})
            )
        )

    assert strategies, f"no faithful value kinds for {backend_cls.__name__}"
    return st.one_of(strategies)


# --------------------------------------------------------------------------
# machines
# --------------------------------------------------------------------------


class CoreConformance(RuleBasedStateMachine):
    """__setitem__, __getitem__, __delitem__, __len__, __contains__, keys()."""

    # Bound by _bind() before hypothesis instantiates the class.
    backend_cls = PickleBackend
    policy_cls = LRUReplacement
    max_in_memory = 2
    dict_factory = None

    keys = Bundle("keys")

    def __init__(self):
        super().__init__()
        cls = type(self)
        self.sut = cls.dict_factory(
            backend=cls.backend_cls,
            policy=cls.policy_cls,
            max_in_memory=cls.max_in_memory,
        )
        self.ref = {}
        self._value_strategy = _values(cls.backend_cls)

    def teardown(self):
        release_store(self.sut)

    # -- helpers ---------------------------------------------------------

    def _mirror(self, op):
        """Apply ``op`` to the reference and the store; demand identical outcomes."""
        ref_exc = None
        ref_value = None
        try:
            ref_value = op(self.ref)
        except Exception as exc:  # noqa: BLE001 - the type is the assertion
            ref_exc = type(exc)

        sut_exc = None
        sut_value = None
        try:
            sut_value = op(self.sut)
        except Exception as exc:  # noqa: BLE001
            sut_exc = type(exc)

        assert sut_exc is ref_exc, (
            f"exception mismatch: dict raised {ref_exc.__name__ if ref_exc else None}, "
            f"EffiDict raised {sut_exc.__name__ if sut_exc else None}"
        )
        if ref_exc is None:
            assert_equal_value(sut_value, ref_value)

    # -- rules -----------------------------------------------------------

    @rule(target=keys, key=KEYS, data=st.data())
    def setitem(self, key, data):
        value = data.draw(self._value_strategy)
        self.ref[key] = copy.deepcopy(value)
        self.sut[key] = value
        return key

    @rule(key=st.one_of(keys, KEYS))
    def getitem(self, key):
        self._mirror(lambda d: d[key])

    @rule(key=st.one_of(keys, KEYS))
    def delitem(self, key):
        def op(d):
            del d[key]

        self._mirror(op)

    @rule(key=st.one_of(keys, KEYS))
    def contains(self, key):
        self._mirror(lambda d: key in d)

    @rule()
    def length(self):
        self._mirror(len)

    @invariant()
    def same_contents(self):
        assert set(self.sut.keys()) == set(self.ref), (
            f"key sets differ: EffiDict {sorted(set(self.sut.keys()))!r} "
            f"vs dict {sorted(self.ref)!r}"
        )
        assert len(self.sut) == len(self.ref)
        for key, expected in self.ref.items():
            assert key in self.sut, f"{key!r} missing from EffiDict"
            assert_equal_value(self.sut[key], expected)


#: The Bundle is a class attribute, so it is not in scope inside subclass bodies.
KEY_BUNDLE = CoreConformance.keys
#: Either a key the machine has already written, or a fresh one, so every rule
#: exercises both the hit and the miss path.
ANY_KEY = st.one_of(KEY_BUNDLE, KEYS)


class PopConformance(CoreConformance):
    """Adds ``pop``. Two defects today, both issue 1.2:

    1. ``pop(key)`` with no default returns ``None`` where ``dict`` raises
       ``KeyError``. This is the shorter reproducer, so it is what shrinking
       finds first.
    2. ``pop`` removes only the memory copy, so a key present in both tiers
       survives and a second call is needed.
    """

    @rule(key=ANY_KEY)
    def pop_with_default(self, key):
        self._mirror(lambda d: d.pop(key, "__missing__"))

    @rule(key=ANY_KEY)
    def pop_without_default(self, key):
        self._mirror(lambda d: d.pop(key))


class ClearConformance(CoreConformance):
    """Adds ``clear``."""

    @rule()
    def clear(self):
        def op(d):
            d.clear()

        self._mirror(op)


class MappingApiConformance(CoreConformance):
    """Adds get/setdefault/update/popitem. All raise NotImplementedError (6.1)."""

    @rule(key=ANY_KEY)
    def get(self, key):
        self._mirror(lambda d: d.get(key))

    @rule(key=ANY_KEY)
    def get_with_default(self, key):
        self._mirror(lambda d: d.get(key, "__missing__"))

    @rule(target=KEY_BUNDLE, key=KEYS, data=st.data())
    def setdefault(self, key, data):
        value = data.draw(self._value_strategy)
        self._mirror(lambda d: d.setdefault(key, copy.deepcopy(value)))
        return key

    @rule(key=KEYS, data=st.data())
    def update(self, key, data):
        value = data.draw(self._value_strategy)
        self._mirror(lambda d: d.update({key: copy.deepcopy(value)}))

    @rule()
    def popitem(self):
        self._mirror(lambda d: d.popitem())


class OrderConformance(CoreConformance):
    """Iteration must yield insertion order. Fails today (issue 4.2)."""

    @invariant()
    def same_order(self):
        assert list(self.sut) == list(self.ref), (
            f"iteration order differs: EffiDict {list(self.sut)!r} "
            f"vs dict {list(self.ref)!r}"
        )

    @rule()
    def keys_are_ordered(self):
        assert list(self.sut.keys()) == list(self.ref.keys())


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------

_COMMON = dict(
    deadline=None,  # real disk I/O; a wall-clock deadline would just be flaky
    suppress_health_check=[
        # make_dict is function-scoped on purpose: every example builds and
        # destroys its own store, which is what we want to exercise.
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
FAST = hyp_settings(max_examples=8, stateful_step_count=12, **_COMMON)
DEEP = hyp_settings(max_examples=120, stateful_step_count=40, **_COMMON)

MAX_IN_MEMORY = [1, 2, 10, 1000]

#: Cross-section used by the rule-group machines: every backend against LRU,
#: plus every policy against Pickle. Cheaper than the full product while still
#: touching each backend and each policy at least once.
CROSS_SECTION = [
    pytest.param(backend, LRUReplacement, id=f"{backend.__name__}-LRUReplacement")
    for backend in BACKENDS
] + [
    pytest.param(PickleBackend, policy, id=f"PickleBackend-{policy.__name__}")
    for policy in POLICIES
    if policy is not LRUReplacement
]


def _bind(base, backend_cls, policy_cls, max_in_memory, factory):
    """Subclass ``base`` with its configuration baked in as class attributes."""
    return type(
        f"{base.__name__}_{backend_cls.__name__}_{policy_cls.__name__}_{max_in_memory}",
        (base,),
        {
            "backend_cls": backend_cls,
            "policy_cls": policy_cls,
            "max_in_memory": max_in_memory,
            "dict_factory": staticmethod(factory),
        },
    )


def _run(base, backend_cls, policy_cls, max_in_memory, factory, settings=FAST):
    machine = _bind(base, backend_cls, policy_cls, max_in_memory, factory)
    run_state_machine_as_test(machine, settings=settings)


# -- default run: cross-section, every cache size -------------------------


@pytest.mark.parametrize("max_in_memory", MAX_IN_MEMORY)
@pytest.mark.parametrize("backend_cls, policy_cls", CROSS_SECTION)
def test_core_conformance(backend_cls, policy_cls, max_in_memory, make_dict):
    _run(CoreConformance, backend_cls, policy_cls, max_in_memory, make_dict)


@pytest.mark.parametrize("max_in_memory", MAX_IN_MEMORY)
@pytest.mark.parametrize("backend_cls, policy_cls", CROSS_SECTION)
def test_clear_conformance(backend_cls, policy_cls, max_in_memory, make_dict):
    _run(ClearConformance, backend_cls, policy_cls, max_in_memory, make_dict)


# -- full 4 x 7 matrix: CI only ------------------------------------------
#
# 28 combinations x 4 cache sizes is too slow for an edit-run loop, so the
# exhaustive sweep is marked `slow`. CI runs it; locally use -m "not slow".


@pytest.mark.slow
@pytest.mark.parametrize("max_in_memory", MAX_IN_MEMORY)
def test_core_conformance_full_matrix(
    backend_cls, policy_cls, max_in_memory, make_dict
):
    _run(CoreConformance, backend_cls, policy_cls, max_in_memory, make_dict)


@pytest.mark.slow
@pytest.mark.parametrize("max_in_memory", MAX_IN_MEMORY)
def test_clear_conformance_full_matrix(
    backend_cls, policy_cls, max_in_memory, make_dict
):
    _run(ClearConformance, backend_cls, policy_cls, max_in_memory, make_dict)


# -- rule groups: cross-section, one cache size --------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "two pop defects (issue 1.2): pop(key) with no default returns None "
        "instead of raising KeyError, and pop removes only the memory copy so a "
        "key present in both tiers survives"
    ),
)
@pytest.mark.parametrize("backend_cls, policy_cls", CROSS_SECTION)
def test_pop_conformance(backend_cls, policy_cls, make_dict):
    _run(PopConformance, backend_cls, policy_cls, 2, make_dict)


@pytest.mark.xfail(
    strict=True,
    reason="get/setdefault/update/popitem raise NotImplementedError (issue 6.1)",
)
@pytest.mark.parametrize("backend_cls, policy_cls", CROSS_SECTION)
def test_mapping_api_conformance(backend_cls, policy_cls, make_dict):
    _run(MappingApiConformance, backend_cls, policy_cls, 2, make_dict)


@pytest.mark.xfail(
    strict=True,
    reason="keys() returns set-union order, not insertion order (issue 4.2)",
)
@pytest.mark.parametrize("backend_cls, policy_cls", CROSS_SECTION)
def test_order_conformance(backend_cls, policy_cls, make_dict):
    _run(OrderConformance, backend_cls, policy_cls, 2, make_dict)


# -- deep runs: one combination, many more examples ----------------------


@pytest.mark.slow
@pytest.mark.parametrize("max_in_memory", MAX_IN_MEMORY)
def test_core_conformance_deep(max_in_memory, make_dict):
    _run(CoreConformance, PickleBackend, LRUReplacement, max_in_memory, make_dict, DEEP)


@pytest.mark.slow
def test_clear_conformance_deep(make_dict):
    _run(ClearConformance, PickleBackend, LRUReplacement, 2, make_dict, DEEP)
