"""Spec for the two target-architecture types that carry the refactor: ``Store``
and ``Cache``.

Issue 0.1 gave both a fully documented surface in which every method raises
``NotImplementedError("see issue #1.x")``, and phases 0.2-0.10 then specified
``EffiDict``, the backends and the policies -- but never these two. Their names
appeared in the suite only inside prose, so Phase 1 could implement ``Store.get``
and ``set`` while silently dropping ``in_cache`` or ``owns_storage`` and nothing
would fail. That matters more than it sounds: ``in_cache`` is what
``helpers.in_cache`` is documented to be rewritten against, so dropping it takes
the tier-invariant specs' only white-box probe with it.

Two kinds of guard here, and the split is deliberate:

* **Structural pins that pass today.** The surface exists as stubs, so asserting
  its shape is a live guard against Phase 1 narrowing it -- the same role
  ``test_key_table_matches_reality`` plays for the keyspace tables.
* **Behavioural specs that xfail today.** ``Store`` and ``Cache`` cannot even be
  constructed yet, so everything about what they *do* is ``xfail(strict=True)``
  against the issue that implements it.

Unlike ``DiskBackend`` and ``EvictionPolicy``, neither type is an ABC: they are
concrete classes awaiting bodies, so they are absent from
``test_abstract_bases_cannot_be_instantiated`` on purpose.
"""

from __future__ import annotations

import inspect

import pytest

from effidict import Cache, EffiDict, Store

#: The documented ``Store`` surface, as issue 0.1 wrote it. Each entry is the
#: name plus the positional parameter names after ``self``, so a Phase 1
#: implementation that keeps the name but changes the contract still trips.
STORE_SURFACE = {
    # The constructor is part of the surface. Both tests that *call* these
    # constructors are expected failures, so without this row a Phase 1
    # implementation that renamed or dropped ``cache`` would stay XFAIL rather
    # than trip a passing guard -- the composition root would be wrong and
    # nothing would say so.
    "__init__": ["backend", "cache", "lock", "owns_storage"],
    "get": ["key"],
    "set": ["key", "value"],
    "delete": ["key"],
    "pop": ["key", "default"],
    "clear": [],
    "update": ["items"],
    "contains": ["key"],
    "count": [],
    "iter_keys": [],
    "in_cache": ["key"],
    "flush": [],
    "close": [],
    "destroy": [],
    "clone": ["new_path"],
}

#: Read-only accessors, checked separately because ``inspect.signature`` on a
#: property object describes the descriptor rather than the getter.
STORE_PROPERTIES = ["backend", "cache", "owns_storage"]

#: The documented ``Cache`` surface.
CACHE_SURFACE = {
    "__init__": ["policy", "max_items", "max_bytes", "size_estimator"],
    "has": ["key"],
    "get": ["key"],
    "peek": ["key"],
    "put": ["key", "value", "dirty"],
    "discard": ["key"],
    "clear": [],
    "__len__": [],
    "nbytes": [],
    "is_dirty": ["key"],
    "mark_clean": ["key"],
    "dirty_items": [],
    "evict_candidates": [],
}


def _positional_names(func):
    """Parameter names after ``self``, excluding ``*args``/``**kwargs``."""
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    parameters = list(inspect.signature(func).parameters.values())
    return [p.name for p in parameters if p.name != "self" and p.kind in kinds]


# --------------------------------------------------------------------------
# structural pins -- these pass today and must keep passing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "owner, surface, properties",
    [
        pytest.param(Store, STORE_SURFACE, STORE_PROPERTIES, id="Store"),
        pytest.param(Cache, CACHE_SURFACE, [], id="Cache"),
    ],
)
def test_surface_is_complete(owner, surface, properties):
    """Every documented method and property must still be there, unchanged.

    Passes today against the stubs. Its whole job is to fail the moment Phase 1
    lands a narrower type than issue 0.1 specified, which is the failure mode
    nothing else in the suite can see.
    """
    missing = [name for name in surface if not hasattr(owner, name)]
    assert not missing, f"{owner.__name__} lost: {missing}"

    wrong = {}
    for name, expected in surface.items():
        actual = _positional_names(getattr(owner, name))
        if actual != expected:
            wrong[name] = f"expected {expected}, got {actual}"
    assert not wrong, f"{owner.__name__} signatures changed: {wrong}"

    for name in properties:
        attribute = inspect.getattr_static(owner, name)
        assert isinstance(attribute, property), (
            f"{owner.__name__}.{name} is no longer a property "
            f"({type(attribute).__name__})"
        )
        # A property with a setter is still a property, so isinstance alone
        # would not hold the "read-only" half of the claim. Where a key lives is
        # the Store's to decide; letting a caller assign the backend or the cache
        # out from under it puts key placement back in two places, which is the
        # root cause this whole epic exists to remove.
        assert attribute.fset is None, (
            f"{owner.__name__}.{name} gained a setter, so callers can swap it"
        )
        assert attribute.fdel is None, (
            f"{owner.__name__}.{name} gained a deleter"
        )


@pytest.mark.parametrize("owner", [Store, Cache], ids=lambda c: c.__name__)
def test_unimplemented_methods_name_their_issue(owner):
    """A stub must say which issue fills it in.

    Only checks methods that still raise ``NotImplementedError``, so this stays
    valid as Phase 1 implements them one at a time: an implemented method is
    simply not examined, and a stub that forgets its reference is.
    """
    unreferenced = []
    for name in list(STORE_SURFACE) + list(CACHE_SURFACE) + STORE_PROPERTIES:
        attribute = inspect.getattr_static(owner, name, None)
        if attribute is None:
            continue
        function = attribute.fget if isinstance(attribute, property) else attribute
        if not callable(function):
            continue
        try:
            function(object.__new__(owner), *[None] * len(_positional_names(function)))
        except NotImplementedError as exc:
            if "issue #" not in str(exc):
                unreferenced.append(f"{name}: {str(exc)!r}")
        except Exception:  # noqa: BLE001 - implemented, or failing for its own reasons
            continue
    assert not unreferenced, f"{owner.__name__} stubs with no issue reference: {unreferenced}"


# --------------------------------------------------------------------------
# behavioural specs -- Store
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Store.__init__ and Cache.__init__ are contract stubs (issue 1.2)",
)
def test_store_can_be_built_over_a_backend_and_a_cache(backend_cls, make_policy, storage_dir):
    """The composition root must actually compose.

    ``Store(backend, cache)`` is the whole point of Phase 1 -- one object owning
    where a key lives -- and today neither constructor runs. Asserted across the
    matrix because a ``Store`` that only composes with some backends is not a
    composition root.
    """
    backend = backend_cls(str(storage_dir / "store"))
    try:
        cache = Cache(policy=make_policy(max_in_memory=4), max_items=4)
        store = Store(backend=backend, cache=cache)

        assert store.backend is backend
        assert store.cache is cache
        assert store.owns_storage is True
    finally:
        backend.destroy()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EffiDict holds its own disk_backend and aliases the policy's dict as "
        "self.memory, so nothing owns key placement (issue 1.2)"
    ),
)
def test_effidict_delegates_placement_to_a_store(make_dict):
    """``EffiDict`` must be a facade over one ``Store``, not a second owner.

    This is the root cause the whole epic is built on: ``EffiDict.__init__`` sets
    both ``self.disk_backend`` and ``self.memory = replacement_strategy.memory``,
    so the facade and the policy both hold the cache and both write to the
    backend. The ``memory`` alias in particular is what let the retired
    ``test_crud.py`` assert tier placement through the facade.
    """
    d = make_dict(max_in_memory=4)
    d["a"] = "va"

    assert isinstance(d._store, Store), (
        f"EffiDict._store is {type(d._store).__name__}, not a Store"
    )
    assert not hasattr(d, "memory"), (
        "EffiDict still aliases the policy's cache dict as .memory, so the "
        "cache has two owners"
    )
    assert d._store.in_cache("a") is True


# --------------------------------------------------------------------------
# behavioural specs -- Cache
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Cache is a contract stub; clean/dirty tagging does not exist (issue 1.3)",
)
def test_cache_tags_entries_clean_or_dirty(make_policy):
    """I2. A cache entry must know whether disk already has it.

    The tag is what makes I5's "evicting a clean entry is a pure memory drop"
    decidable at all. Without it every eviction has to assume dirty, which is
    exactly the 500-writes-for-500-reads cost the complexity guards measure.
    """
    cache = Cache(policy=make_policy(max_in_memory=4), max_items=4)

    cache.put("written", "v", dirty=True)
    cache.put("loaded", "v", dirty=False)

    assert cache.is_dirty("written") is True
    assert cache.is_dirty("loaded") is False
    assert [key for key, _ in cache.dirty_items()] == ["written"]

    cache.mark_clean("written")
    assert cache.is_dirty("written") is False
    assert list(cache.dirty_items()) == []


@pytest.mark.xfail(
    strict=True,
    reason="Cache is a contract stub; neither budget is enforced (issue 1.4)",
)
def test_cache_enforces_item_and_byte_budgets_independently(make_policy):
    """I8. Either budget alone must be able to trigger eviction.

    Two budgets, two separate failure modes: today ``max_in_memory`` counts items
    and ``max_bytes`` is accepted and dropped, which is how a hundred 1 MB values
    fit a "hundred-item" cache. The docstring on ``Cache.__init__`` already says
    eviction happens while *either* budget is exceeded; this pins it.
    """
    by_items = Cache(policy=make_policy(max_in_memory=2), max_items=2)
    for index in range(5):
        by_items.put(f"k{index}", "v")
    assert len(by_items) <= 2, f"item budget ignored: {len(by_items)} entries resident"

    by_bytes = Cache(
        policy=make_policy(max_in_memory=1000),
        max_items=1000,
        max_bytes=4096,
    )
    for index in range(20):
        by_bytes.put(f"k{index}", "x" * 1024)
    assert by_bytes.nbytes() <= 4096 * 1.5, (
        f"byte budget ignored: {by_bytes.nbytes()} bytes resident in "
        f"{len(by_bytes)} entries"
    )


@pytest.mark.xfail(
    strict=True,
    reason="Cache is a contract stub (issue 1.2)",
)
def test_cache_peek_does_not_promote(make_policy):
    """I6. ``peek`` must read without telling the policy.

    The distinction between ``get`` and ``peek`` is the only way a flush or a
    full scan can walk the cache without reordering it -- the defect
    ``test_full_scan_does_not_evict_the_working_set`` measures from the outside.
    """
    policy = make_policy(max_in_memory=2)
    cache = Cache(policy=policy, max_items=2)
    cache.put("a", "va")
    cache.put("b", "vb")

    before = list(cache.evict_candidates())
    assert cache.peek("a") == "va"
    assert list(cache.evict_candidates()) == before, (
        "peek reordered the eviction queue, so it is not a pure read"
    )
