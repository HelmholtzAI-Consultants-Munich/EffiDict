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

from effidict import Cache, EffiDict, EvictionPolicy, Store

#: The documented ``Store`` surface, as issue 0.1 wrote it. Each entry is the
#: name plus the positional parameter names after ``self``, so a Phase 1
#: implementation that keeps the name but changes the contract still trips.
STORE_SURFACE = {
    # The constructor is part of the surface. Both tests that *call* these
    # constructors are expected failures, so without this row a Phase 1
    # implementation that renamed or dropped ``cache`` would stay XFAIL rather
    # than trip a passing guard -- the composition root would be wrong and
    # nothing would say so.
    "__init__": ["backend", "cache", "lock=", "owns_storage="],
    "get": ["key"],
    "set": ["key", "value"],
    "delete": ["key"],
    "pop": ["key", "default="],
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
    "__init__": ["policy", "max_items=", "max_bytes=", "size_estimator="],
    "has": ["key"],
    "get": ["key"],
    "peek": ["key"],
    "put": ["key", "value", "dirty="],
    "discard": ["key"],
    "clear": [],
    "__len__": [],
    "nbytes": [],
    "is_dirty": ["key"],
    "mark_clean": ["key"],
    "dirty_items": [],
    "evict_candidates": [],
}


def _signature_spec(func):
    """Every parameter after ``self``, with its kind and whether it has a default.

    Names alone are not a signature. Three things the earlier name-only form
    could not see, each of which changes what callers may write:

    * **Defaults.** ``pop(key, default=MISSING)`` and ``pop(key, default)``
      produce identical name lists, so Phase 1 could make the default required
      and break ``pop(key)`` -- the precise ambiguity the ``MISSING`` sentinel
      was added to remove -- without this guard noticing.
    * **Catch-alls.** ``*args``/``**kwargs`` were filtered out, so every method
      expecting no parameters -- eleven of them across the two tables -- was
      satisfied by ``def method(self, *args, **kwargs)``.
    * **Keyword-only.** Those parameters were dropped silently, so the table
      could neither require nor forbid them.

    Encoding, chosen to stay readable in the tables below::

        key          positional-or-keyword, required
        key=         positional-or-keyword, has a default
        kw:key       keyword-only, required
        kw:key=      keyword-only, has a default
        pos:key      positional-only
        *args        var-positional
        **kwargs     var-keyword
    """
    parameter = inspect.Parameter
    spec = []
    for item in inspect.signature(func).parameters.values():
        if item.name == "self":
            continue
        default = "=" if item.default is not parameter.empty else ""
        if item.kind is parameter.VAR_POSITIONAL:
            spec.append(f"*{item.name}")
        elif item.kind is parameter.VAR_KEYWORD:
            spec.append(f"**{item.name}")
        elif item.kind is parameter.KEYWORD_ONLY:
            spec.append(f"kw:{item.name}{default}")
        elif item.kind is parameter.POSITIONAL_ONLY:
            spec.append(f"pos:{item.name}{default}")
        else:
            spec.append(f"{item.name}{default}")
    return spec


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
        actual = _signature_spec(getattr(owner, name))
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
            function(object.__new__(owner), *[None] * len(_signature_spec(function)))
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
def test_store_can_be_built_over_a_backend_and_a_cache(backend_cls, storage_dir):
    """The composition root must actually compose.

    ``Store(backend, cache)`` is the whole point of Phase 1 -- one object owning
    where a key lives -- and today neither constructor runs. Asserted across the
    matrix because a ``Store`` that only composes with some backends is not a
    composition root.
    """
    backend = backend_cls(str(storage_dir / "store"))
    try:
        cache = Cache(policy=RecordingPolicy(), max_items=4)
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
def test_cache_tags_entries_clean_or_dirty():
    """I2. A cache entry must know whether disk already has it.

    The tag is what makes I5's "evicting a clean entry is a pure memory drop"
    decidable at all. Without it every eviction has to assume dirty, which is
    exactly the 500-writes-for-500-reads cost the complexity guards measure.
    """
    cache = Cache(policy=RecordingPolicy(), max_items=4)

    cache.put("written", "v", dirty=True)
    cache.put("loaded", "v", dirty=False)

    assert cache.is_dirty("written") is True
    assert cache.is_dirty("loaded") is False
    assert [key for key, _ in cache.dirty_items()] == ["written"]

    cache.mark_clean("written")
    assert cache.is_dirty("written") is False
    assert list(cache.dirty_items()) == []


#: A policy that records what it was told, so a spec can assert that a cache
#: consulted it -- or did not. Real policies keep their bookkeeping private, and
#: "peek did not promote" is a statement about a call that must *not* happen,
#: which is unobservable from the outside.
#:
#: Used by every Cache spec here in place of the matrix ``policy_cls``, and not
#: only for observability: ``Cache`` takes an ``EvictionPolicy``, while the
#: matrix still supplies the legacy ``*Replacement`` classes, which do not grow
#: ``on_insert``/``victim`` until issue 1.1. Handing one to a correct ``Cache``
#: raises ``AttributeError``, so these specs were unsatisfiable by the very
#: implementation they are meant to describe -- found by running them against a
#: reference ``Cache`` rather than by reading them.
class RecordingPolicy(EvictionPolicy):
    def __init__(self):
        self.calls = []
        self.order = []

    def on_insert(self, key):
        self.calls.append(("on_insert", key))
        self.order.append(key)

    def on_access(self, key):
        self.calls.append(("on_access", key))
        if key in self.order:                      # move-to-most-recent
            self.order.remove(key)
            self.order.append(key)

    def on_remove(self, key):
        self.calls.append(("on_remove", key))
        if key in self.order:
            self.order.remove(key)

    def victim(self):
        if not self.order:
            raise KeyError("empty")
        return self.order[0]

    def victims(self):
        # A snapshot, so walking it records nothing and is unaffected by any
        # discard the caller makes while iterating.
        return iter(list(self.order))

    def clear(self):
        self.calls.append(("clear", None))
        self.order.clear()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Cache is a contract stub, so this dies at the constructor (issue 1.2) "
        "before it can reach the thing it pins -- neither budget is enforced "
        "(issue 1.4). It also exercises has/discard, which are 1.2 as well, so "
        "it stays red until both land"
    ),
)
def test_cache_reports_what_must_be_shed_for_each_budget():
    """I8. Either budget alone must put entries on the eviction list.

    Two budgets, two separate failure modes: today ``max_in_memory`` counts items
    and ``max_bytes`` is accepted and dropped, which is how a hundred 1 MB values
    fit a "hundred-item" cache.

    Asserted as *what the cache says must go*, not as the cache having already
    dropped it. An earlier version of this spec demanded ``len(cache) <= 2``
    after over-filling, which encodes the opposite ownership model and is
    unimplementable alongside I5: if ``put`` drops its own victim, a dirty value
    is gone before anything could write it out, and ``Cache`` has no backend to
    write it to. ``evict_candidates`` naming the surplus is the form that lets
    ``Store`` peek, write, then discard.

    The exact count is asserted, not a lower bound, and that is a real
    constraint worth stating: ``Cache`` promises *keys that must be shed to get
    back inside budget*, so shedding to a low-water mark to avoid thrashing
    falls outside the contract as written. If issue 1.4 wants watermark
    eviction, this assertion and that docstring change together -- which is the
    point of pinning it here rather than discovering the disagreement later.
    """
    by_items = Cache(policy=RecordingPolicy(), max_items=2)
    for index in range(5):
        by_items.put(f"k{index}", "v")
    assert len(by_items) == 5, (
        f"put dropped entries by itself: {len(by_items)} of 5 held, so a dirty "
        f"victim would be unrecoverable"
    )
    candidates = list(by_items.evict_candidates())
    # Distinct, and actually present. ``len(...) == 3`` alone is satisfied by a
    # cache that names one key three times, which sheds a single entry and
    # leaves the budget broken.
    assert len(set(candidates)) == len(candidates), (
        f"evict_candidates repeated a key: {candidates}"
    )
    assert all(by_items.has(key) for key in candidates), (
        f"evict_candidates named keys the cache does not hold: "
        f"{[key for key in candidates if not by_items.has(key)]}"
    )
    # And sufficient: discarding exactly what it named must reach the budget,
    # which is the property the caller depends on. A list that is merely
    # non-empty leaves the owner guessing how many times to ask.
    assert len(candidates) == 3, (
        f"3 entries are over a 2-item budget but evict_candidates named "
        f"{len(candidates)}: {candidates}"
    )
    for key in candidates:
        by_items.discard(key)
    assert len(by_items) <= 2, (
        f"after discarding every named candidate the cache still holds "
        f"{len(by_items)} entries against a 2-item budget"
    )
    assert not list(by_items.evict_candidates()), (
        "the cache is inside its budget but still names candidates to shed"
    )

    by_bytes = Cache(
        policy=RecordingPolicy(),
        max_items=1000,
        max_bytes=4096,
    )
    for index in range(20):
        by_bytes.put(f"k{index}", "x" * 1024)
    assert by_bytes.nbytes() > 4096, (
        "nbytes() did not account the values it was given"
    )
    byte_candidates = list(by_bytes.evict_candidates())
    assert byte_candidates, (
        f"{by_bytes.nbytes()} bytes are resident against a 4096-byte budget and "
        f"evict_candidates named nothing, so the byte budget is not tracked"
    )
    assert len(set(byte_candidates)) == len(byte_candidates), (
        f"evict_candidates repeated a key: {byte_candidates}"
    )
    # Non-empty is far too weak here: 20 KiB against a 4 KiB budget needs most
    # of the entries gone, and a cache naming one victim would pass that alone.
    for key in byte_candidates:
        by_bytes.discard(key)
    assert by_bytes.nbytes() <= 4096, (
        f"after discarding every named candidate {by_bytes.nbytes()} bytes are "
        f"still resident against a 4096-byte budget"
    )
    assert not list(by_bytes.evict_candidates()), (
        "the cache is inside its byte budget but still names candidates to shed"
    )


@pytest.mark.xfail(
    strict=True,
    reason="Cache is a contract stub; size_estimator is never consulted (issue 1.4)",
)
def test_cache_measures_values_with_the_size_estimator():
    """``size_estimator`` must actually be used, not assumed to be ``len``.

    It is the only way the byte budget can work for values whose footprint is
    not their length -- an ndarray, a DataFrame, anything where ``len`` is a row
    count. Every other byte-budget guard in the suite uses strings, so all of
    them pass against a hard-coded ``len``, and this parameter was on the public
    contract with nothing exercising it.

    The values here report ten bytes each through the estimator while ``len``
    says one, so the two implementations cannot both satisfy this.
    """
    cache = Cache(
        policy=RecordingPolicy(),
        max_items=1000,
        max_bytes=25,
        size_estimator=lambda value: 10,
    )
    for index in range(4):
        cache.put(f"k{index}", "x")

    assert cache.nbytes() == 40, (
        f"nbytes() reported {cache.nbytes()} for 4 values the estimator sizes at "
        f"10 bytes each; len() would give 4"
    )
    assert list(cache.evict_candidates()), (
        "40 estimated bytes against a 25-byte budget shed nothing, so eviction "
        "ignores the estimator even if nbytes() honours it"
    )


@pytest.mark.xfail(
    strict=True,
    reason="Cache is a contract stub (issue 1.2)",
)
def test_cache_peek_does_not_promote():
    """I6. ``peek`` must read without telling the policy.

    The distinction between ``get`` and ``peek`` is the only way a flush or a
    full scan can walk the cache without reordering it -- the defect
    ``test_full_scan_does_not_evict_the_working_set`` measures from the outside.

    Two things make this observable, and the earlier version of this spec had
    neither. It compared ``evict_candidates()`` before and after on a cache
    holding exactly its budget, where the correct answer is ``[]`` both times --
    so ``[] == []`` passed however much ``peek`` had reordered. Here the cache is
    deliberately over budget, so the candidate order is non-empty and visible,
    and the policy records its calls, so "did not tell the policy" is asserted
    directly rather than inferred.
    """
    policy = RecordingPolicy()
    cache = Cache(policy=policy, max_items=2)
    for key in ("a", "b", "c", "d"):
        cache.put(key, f"v{key}")

    before = list(cache.evict_candidates())
    assert before, "precondition: an over-budget cache has candidates to name"

    policy.calls.clear()
    assert cache.peek("a") == "va"

    promoted = [call for call in policy.calls if call[0] == "on_access"]
    assert not promoted, f"peek told the policy about the access: {promoted}"
    assert list(cache.evict_candidates()) == before, (
        f"peek reordered the eviction queue: {before} -> "
        f"{list(cache.evict_candidates())}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="Cache is a contract stub (issue 1.2)",
)
def test_cache_get_reads_and_records_the_access():
    """``get`` must return the value *and* tell the policy.

    The complement of ``test_cache_peek_does_not_promote``, and without it the
    pair said only half of what it needs to: nothing exercised ``get`` at all,
    so an implementation where ``get`` is literally ``peek`` passed every Cache
    spec in this file. That store would hold a cache whose recency and frequency
    data never moved -- LRU degenerating to FIFO, LFU to insertion order --
    which is invisible to any test that only checks the value came back.

    So the recording is asserted directly through ``RecordingPolicy``, the same
    way its absence is asserted for ``peek``.
    """
    policy = RecordingPolicy()
    cache = Cache(policy=policy, max_items=4)
    cache.put("a", "va")
    cache.put("b", "vb")

    policy.calls.clear()
    assert cache.get("a") == "va"

    accesses = [call for call in policy.calls if call[0] == "on_access"]
    assert accesses == [("on_access", "a")], (
        f"get did not record the access as exactly one on_access('a'): "
        f"{policy.calls}"
    )

    # A miss must raise rather than return a sentinel -- Store.get distinguishes
    # "not cached" from "cached as None" on this, and a cache that returned None
    # for both would make a stored None indistinguishable from an absent key.
    with pytest.raises(KeyError):
        cache.get("absent")


@pytest.mark.xfail(
    strict=True,
    reason="Store is a contract stub (issue 1.2)",
)
def test_store_pop_distinguishes_an_omitted_default_from_none(backend_cls, storage_dir):
    """``pop`` must tell ``pop(key)`` from ``pop(key, None)``.

    This is the entire reason ``Store.pop`` carries a ``MISSING`` sentinel
    instead of ``default=None``, and nothing tested it: no spec invoked
    ``Store.pop`` at all, and the structural guard cannot help because
    ``default=MISSING`` and ``default=None`` produce the identical signature
    spec ``['key', 'default=']``. Phase 1 could therefore restore
    ``default=None`` -- reintroducing exactly the missing-key defect
    ``test_pop_without_default_raises_keyerror`` pins at the facade -- with every
    Store spec still green.
    """
    backend = backend_cls(str(storage_dir / "store"))
    try:
        store = Store(backend=backend, cache=Cache(policy=RecordingPolicy(), max_items=4))
        store.set("present", "value")
        store.set("holds-none", None)

        assert store.pop("present") == "value"
        assert not store.contains("present"), "pop returned the value but kept the key"

        # Omitted: raise. Given explicitly: return it, even when it is None.
        with pytest.raises(KeyError):
            store.pop("absent")
        assert store.pop("absent", None) is None
        assert store.pop("absent", "fallback") == "fallback"

        # And a stored None is a hit, not a miss.
        assert store.pop("holds-none", "fallback") is None
    finally:
        backend.destroy()
