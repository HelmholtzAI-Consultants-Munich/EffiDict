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

Unlike ``DiskBackend`` and ``EvictionPolicy``, neither type is an ABC, and that
is deliberate rather than an oversight: an abstract base earns its place where
several implementations have to be kept honest, and PLAN.md plans exactly one of
each -- ``Store`` is "THE single owner of key location", and both vary by what is
injected into them rather than by subclassing. They are therefore absent from
``test_abstract_bases_cannot_be_instantiated``, which covers the three bases that
do have implementations. Making them abstract would also contradict the specs
below, which construct ``Cache(...)`` and ``Store(...)`` directly.
"""

from __future__ import annotations

import gc
import inspect
import threading

import pytest

from effidict import Cache, EffiDict, EvictionPolicy, PickleBackend, Store

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


def test_both_tiers_are_required_to_construct():
    """No constructor argument may carry a default it cannot honour.

    Three are required, for two different reasons. The backend and the policy
    are the tiers themselves: before issue 1.2 they defaulted to ``None`` and
    ``__init__`` ended with ``self.memory = replacement_strategy.memory``, so
    either default could only raise ``AttributeError`` from inside the object.
    ``max_in_memory`` is required because a default would be *usable* and wrong
    -- an unbounded cache never spills, which is the one thing this library
    exists to do, and the store would silently become a dict that also writes
    files.

    A caller who omits one should be told which, by the signature, before any
    attribute lookup happens.
    """
    parameters = inspect.signature(EffiDict.__init__).parameters
    for name in ("disk_backend", "policy", "max_in_memory"):
        assert parameters[name].default is inspect.Parameter.empty, (
            f"{name} declares a default that __init__ cannot work with"
        )

    with pytest.raises(TypeError) as raised:
        EffiDict()
    assert "policy" in str(raised.value), (
        f"constructing without arguments reported {raised.value!r}, which does "
        f"not name the missing argument"
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
#: Used by every Cache spec here in place of the matrix ``policy_cls``. The
#: matrix now supplies real ``*Policy`` classes, which a ``Cache`` accepts, but
#: those keep their bookkeeping private: what these specs assert is which calls
#: the cache made, and that is only visible from a policy that records them.
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


def test_store_works_with_a_non_reentrant_lock(storage_dir):
    """Every operation must acquire the lock exactly once.

    ``lock=`` is part of the ``Store`` surface, and PLAN.md 3.6 lets a caller
    inject a ``Lock`` protocol that need not be reentrant -- one shared across
    stores, or an object that is not a ``threading`` lock at all. An
    implementation whose public methods call each other re-acquires, which a
    reentrant default hides: ``delete``, ``pop`` and ``count`` each hung
    indefinitely under ``threading.Lock`` and nothing failed, because a deadlock
    is not an exception.

    Run on a worker with a bounded wait, since the failure mode is a hang, and
    the worker's exception is re-raised rather than swallowed -- a call that
    died still sets the "finished" event, so liveness alone would pass on a
    broken call.

    One backend, not the matrix: the claim is about ``Store`` acquiring its own
    lock, which no backend participates in. Across the matrix SQLite would fail
    here for an unrelated reason -- its connection cannot be used off the thread
    that opened it (issue 2.3) -- and that noise would say nothing about locking.
    """
    backend = PickleBackend(str(storage_dir / "store"))
    try:
        store = Store(
            backend=backend,
            cache=Cache(policy=RecordingPolicy(), max_items=4),
            lock=threading.Lock(),
        )
        store.set("a", "va")
        store.set("b", "vb")

        for label, call in (
            ("contains", lambda: store.contains("a")),
            ("get", lambda: store.get("a")),
            ("count", lambda: store.count()),
            ("iter_keys", lambda: list(store.iter_keys())),
            ("pop", lambda: store.pop("a", None)),
            ("delete", lambda: store.delete("b")),
            ("clear", lambda: store.clear()),
        ):
            finished = threading.Event()
            failure = {}

            def run(call=call):
                try:
                    call()
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    failure["exc"] = exc
                finally:
                    finished.set()

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            assert finished.wait(5.0), (
                f"Store.{label} did not return within 5s under a plain Lock, "
                f"so it acquires the lock more than once"
            )
            if "exc" in failure:
                raise AssertionError(
                    f"Store.{label} raised {failure['exc']!r} on a worker thread"
                ) from failure["exc"]
    finally:
        backend.destroy()


#: Distinct from every value a test stores, so "the finalizer got KeyError"
#: cannot be confused with "the finalizer read ``None``".
_ABSENT = object()


#: Read by :class:`_ReentrantFinalizer`, so the stored value itself holds
#: nothing but a string and stays picklable -- the eviction case below writes it
#: to disk. Keyed by tag, and each probe removes its own entry as it fires, so
#: the copy that unpickling creates cannot set the recording off a second time.
_FINALIZER_PROBES = {}


class _ReentrantFinalizer:
    """A stored value whose ``__del__`` queries the store that held it."""

    def __init__(self, tag):
        self.tag = tag

    def __del__(self):
        probe = _FINALIZER_PROBES.pop(self.tag, None)
        if probe is not None:
            probe()


@pytest.mark.parametrize(
    "operation, survives",
    [
        ("overwrite", True),
        ("evict", True),
        ("delete", False),
        ("clear", False),
    ],
)
def test_a_finalizer_never_runs_inside_a_tier_operation(
    operation, survives, storage_dir
):
    """Displacing a value must not hand control to it mid-operation.

    CPython runs ``__del__`` the instant the last reference goes, on the thread
    that dropped it. Every mutating path drops one: an overwrite displaces the
    old value, an eviction and a delete drop the cached one, ``clear`` drops
    them all. Each of those drops sits inside the lock, so without care the
    cache hands a tier operation it has half-applied to arbitrary user code.

    Both halves are asserted, because the two plausible implementations each
    fail only one. Keeping a plain lock and dropping the value in place
    **hangs** -- the finalizer's query blocks on a lock its own thread holds --
    which the bounded wait catches. Making the lock reentrant instead lets the
    query through into the gap: measured on ``delete``, after the cache had
    dropped the key and before the backend had, ``get`` returned the disk copy
    of a key being deleted. So liveness alone is not the property; the
    finalizer must also find the store whole, which is what ``survives``
    pins per operation.

    ``pop`` is absent deliberately: it returns the value, so the caller's
    reference outlives the lock and nothing is dropped inside it.

    One backend and the **default** lock. Default, because that is what a
    reverted fix would change and what users actually get; injected locks are
    ``test_store_works_with_a_non_reentrant_lock``'s subject. One backend,
    for that test's reason -- SQLite's thread affinity (issue 2.3) would fail
    here while saying nothing about finalizers.
    """
    backend = PickleBackend(str(storage_dir / "store"))
    observed = []
    tag = f"probe-{operation}"
    try:
        store = Store(
            backend=backend,
            cache=Cache(
                policy=RecordingPolicy(),
                max_items=1 if operation == "evict" else 4,
            ),
        )

        def probe():
            # Whatever the finalizer can see of the store it was dropped from.
            try:
                present = store.contains("k")
            except BaseException as exc:  # noqa: BLE001 - reported below
                observed.append(("contains-raised", exc))
                return
            try:
                observed.append(("seen", present, store.get("k")))
            except KeyError:
                observed.append(("seen", present, _ABSENT))

        _FINALIZER_PROBES[tag] = probe
        # A disk copy as well as the cached one, so that "the store is whole"
        # is a real question during a delete: a finalizer let into the gap
        # reads this value, where one running afterwards gets ``KeyError``.
        backend.serialize("k", "on-disk")
        store.set("k", _ReentrantFinalizer(tag))

        actions = {
            "overwrite": lambda: store.set("k", "replacement"),
            "evict": lambda: store.set("other", "v"),
            "delete": lambda: store.delete("k"),
            "clear": lambda: store.clear(),
        }
        finished = threading.Event()
        failure = {}

        def run():
            try:
                actions[operation]()
            except Exception as exc:  # noqa: BLE001 - re-raised below
                failure["exc"] = exc
            finally:
                finished.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert finished.wait(5.0), (
            f"{operation} did not return within 5s: the displaced value's "
            f"__del__ ran while the lock was held and blocked on it"
        )
        if "exc" in failure:
            raise AssertionError(
                f"{operation} raised {failure['exc']!r} on a worker thread"
            ) from failure["exc"]

        assert len(observed) == 1, (
            f"the finalizer ran {len(observed)} times during {operation}; the "
            f"test proves nothing unless the value was actually dropped"
        )
        record = observed[0]
        assert record[0] == "seen", (
            f"the finalizer could not query the store during {operation}: "
            f"{record}"
        )
        _, present, value = record
        assert present is survives, (
            f"during {operation} the finalizer saw contains('k') == {present}, "
            f"expected {survives}: it was let into a half-applied operation"
        )
        if not survives:
            assert value is _ABSENT, (
                f"during {operation} the finalizer read {value!r} for a key "
                f"being removed, so it ran between the two tiers"
            )
    finally:
        _FINALIZER_PROBES.pop(tag, None)
        backend.destroy()


class _CollectsOnWrite(PickleBackend):
    """A backend whose write runs a cyclic collection.

    Stands in for the one thing a test cannot schedule: the collector tripping
    its threshold partway through a locked operation. Writing allocates, so the
    real collector may do exactly this at any time -- but *when* is not
    controllable, and waiting for it does not make a guard. Measured on this
    tree: 400 store operations under ``gc.set_threshold(1, 1, 1)`` put zero
    collections inside the critical section, because the surrounding code
    allocated first every time. Forcing the collection moves the rarity out of
    the test without changing what is being tested.
    """

    def serialize(self, key, value):
        gc.collect()
        return super().serialize(key, value)


def test_a_cyclic_value_cannot_deadlock_the_store(storage_dir):
    """Re-entry the deferral cannot reach must report itself, not hang.

    The ``released`` lists in ``Store`` hold a displaced value past the unlock,
    which covers every finalizer that *refcounting* would run. A value in a
    reference cycle is not one of them: dropping the last reference does not
    free it, and the collector does, whenever it next runs -- which may be deep
    inside an unrelated locked operation. No refcount-side strategy reaches
    that, so the lock refuses same-thread re-entry instead.

    Refusing is what is asserted, not merely surviving. A bare ``Lock`` hangs
    here, which the bounded wait catches; a reentrant one would let the
    finalizer through to read a half-applied operation, and then ``contains``
    returns a bool rather than raising. Only the guard gives ``RuntimeError``,
    so the error *is* the evidence that the finalizer ran inside the lock.

    Automatic collection is off for the duration, which removes scheduling
    nondeterminism rather than the hazard: with it on, an incidental collection
    between the ``delete`` below and the worker could reap the cycle early and
    the test would pass without having tested anything.
    """
    backend = _CollectsOnWrite(str(storage_dir / "store"))
    recorded = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        store = Store(
            backend=backend,
            cache=Cache(policy=RecordingPolicy(), max_items=1),
        )

        class Cyclic:
            """Refers to itself, so only the collector can free it."""

            def __init__(self):
                self.self_ref = self

            def __del__(self):
                try:
                    recorded.append(("returned", store.contains("a")))
                except BaseException as exc:  # noqa: BLE001 - inspected below
                    recorded.append(("raised", exc))

        store.set("cyclic", Cyclic())
        store.delete("cyclic")  # unreachable now; the cycle keeps it alive

        finished = threading.Event()
        failure = {}

        def run():
            try:
                store.set("a", "va")
                store.set("b", "vb")  # over budget -> serialize -> collect
            except Exception as exc:  # noqa: BLE001 - re-raised below
                failure["exc"] = exc
            finally:
                finished.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert finished.wait(5.0), (
            "a cyclic value's __del__ ran inside the lock and blocked on it: "
            "the write never returned"
        )
        if "exc" in failure:
            raise AssertionError(
                f"the write raised {failure['exc']!r} on a worker thread"
            ) from failure["exc"]

        assert len(recorded) == 1, (
            f"the cyclic finalizer ran {len(recorded)} times; the test proves "
            f"nothing unless the collector reaped it during the write"
        )
        kind, detail = recorded[0]
        assert kind == "raised", (
            f"the finalizer re-entered the store and was served: contains() "
            f"returned {detail!r} from inside a half-applied operation"
        )
        assert isinstance(detail, RuntimeError), (
            f"re-entry raised {detail!r}, which says nothing a caller can act "
            f"on; it should name the re-entry"
        )
        assert "re-entered" in str(detail), (
            f"the re-entry error does not explain itself: {detail}"
        )

        # The refused call must leave the operation under it alone.
        assert store.get("b") == "vb", "the write that was re-entered was lost"
        assert not store.contains("cyclic"), "the deleted key came back"
    finally:
        if was_enabled:
            gc.enable()
        backend.destroy()


def _cycle_that_probes(recorded, probe):
    """Return a self-referential value whose finalizer runs ``probe``.

    Only the collector can free it, so the finalizer fires wherever the next
    collection lands -- which is the whole point: that may be inside a locked
    operation. What the probe did is appended to ``recorded``, either as
    ``("returned", answer)`` or as ``("raised", exception)``.
    """

    class Cyclic:
        def __init__(self):
            self.self_ref = self

        def __del__(self):
            try:
                recorded.append(("returned", probe()))
            except BaseException as exc:  # noqa: BLE001 - inspected by the test
                recorded.append(("raised", exc))

    return Cyclic()


def _assert_reentry_was_refused(recorded, during):
    """Assert the probe ran once, inside the lock, and was turned away.

    Refusal is what is asserted, not merely survival. A bare ``Lock`` hangs,
    which the caller's bounded wait catches; a reentrant one serves the
    finalizer a half-applied operation, and the probe then returns an answer
    instead of raising. Only the guard raises, so the error *is* the evidence
    that the finalizer ran inside the lock.
    """
    assert len(recorded) == 1, (
        f"the finalizer ran {len(recorded)} times during {during}; the test "
        f"proves nothing unless the collector reaped the cycle in there"
    )
    kind, detail = recorded[0]
    assert kind == "raised", (
        f"the finalizer re-entered during {during} and was served: the probe "
        f"returned {detail!r} from inside a half-applied operation"
    )
    assert isinstance(detail, RuntimeError), (
        f"re-entry during {during} raised {detail!r}, which says nothing a "
        f"caller can act on; it should name the re-entry"
    )
    assert "re-entered" in str(detail), (
        f"the re-entry error does not explain itself: {detail}"
    )


def test_stores_sharing_one_lock_share_re_entry_detection(storage_dir):
    """One lock means one critical section, so it must mean one answer.

    ``Store`` takes an injected lock precisely so several stores can be held
    consistent under it. A finalizer collected inside store A's operation and
    querying store B is then re-entering the *lock*, not the store: B's
    acquire blocks on a lock the calling thread already holds through A, and
    the thread waits for itself. Detection has to be a property of the lock
    rather than of either store, which is why the guard tracks the lock's
    identity and not its own.
    """
    a_backend = _CollectsOnWrite(str(storage_dir / "a"))
    b_backend = PickleBackend(str(storage_dir / "b"))
    recorded = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        lock = threading.Lock()
        a = Store(
            backend=a_backend,
            cache=Cache(policy=RecordingPolicy(), max_items=1),
            lock=lock,
        )
        b = Store(
            backend=b_backend,
            cache=Cache(policy=RecordingPolicy(), max_items=4),
            lock=lock,
        )
        b.set("other", "untouched")

        a.set("cyclic", _cycle_that_probes(recorded, lambda: b.contains("other")))
        a.delete("cyclic")  # unreachable now; the cycle keeps it alive

        finished = threading.Event()
        failure = {}

        def run():
            try:
                a.set("x", "vx")
                a.set("y", "vy")  # over budget -> serialize -> collect
            except Exception as exc:  # noqa: BLE001 - re-raised below
                failure["exc"] = exc
            finally:
                finished.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert finished.wait(5.0), (
            "a finalizer collected inside one store's operation queried the "
            "other store sharing its lock, and blocked on it: the write never "
            "returned"
        )
        if "exc" in failure:
            raise AssertionError(
                f"the write raised {failure['exc']!r} on a worker thread"
            ) from failure["exc"]

        _assert_reentry_was_refused(recorded, "a store sharing the lock")

        # Neither store may be left disturbed by the call that was refused.
        assert a.get("y") == "vy", "the write that was re-entered was lost"
        assert b.get("other") == "untouched", "the queried store was disturbed"
    finally:
        if was_enabled:
            gc.enable()
        a_backend.destroy()
        b_backend.destroy()


class _CollectingLock:
    """A lock written in Python that collects inside ``acquire`` or ``release``.

    The ``Lock`` protocol of PLAN.md 3.6 lets a caller inject any object with
    these two methods, and a Python one runs Python in both -- so it allocates,
    so the collector can trip there, while the underlying lock is already held
    (inside ``acquire``) or still held (inside ``release``). Those are two
    windows a guard that claims ownership only *between* the calls does not
    cover. Collecting on purpose puts a finalizer in the window every time
    rather than once in a long while; the stdlib locks cannot reach it at all,
    which is why this takes a lock of its own to show.
    """

    def __init__(self, collect_in):
        self._lock = threading.Lock()
        self._collect_in = collect_in

    def acquire(self, *args):
        acquired = self._lock.acquire(*args)
        if self._collect_in == "acquire":
            gc.collect()
        return acquired

    def release(self):
        if self._collect_in == "release":
            gc.collect()
        self._lock.release()


@pytest.mark.parametrize("collect_in", ["acquire", "release"])
def test_an_injected_python_lock_cannot_hide_re_entry(collect_in, storage_dir):
    """Ownership has to span the lock calls, not sit between them.

    A finalizer run by a collection inside the injected lock's own ``acquire``
    or ``release`` arrives while that lock is held, but outside the span a
    guard would claim if it marked ownership after acquiring and dropped it
    before releasing. It would then be waved through to block on a lock its
    own thread holds. Marking before the acquire and clearing after the
    release closes both ends.
    """
    backend = PickleBackend(str(storage_dir / "store"))
    recorded = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        store = Store(
            backend=backend,
            cache=Cache(policy=RecordingPolicy(), max_items=4),
            lock=_CollectingLock(collect_in),
        )
        store.set(
            "cyclic", _cycle_that_probes(recorded, lambda: store.contains("x"))
        )
        store.delete("cyclic")  # unreachable now; the cycle keeps it alive

        finished = threading.Event()
        failure = {}

        def run():
            try:
                store.set("x", "vx")
            except Exception as exc:  # noqa: BLE001 - re-raised below
                failure["exc"] = exc
            finally:
                finished.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert finished.wait(5.0), (
            f"a finalizer collected inside the lock's own {collect_in}() "
            f"re-entered the store and blocked: the write never returned"
        )
        if "exc" in failure:
            raise AssertionError(
                f"the write raised {failure['exc']!r} on a worker thread"
            ) from failure["exc"]

        _assert_reentry_was_refused(recorded, f"the lock's {collect_in}()")
        assert store.get("x") == "vx", "the write that was re-entered was lost"
    finally:
        if was_enabled:
            gc.enable()
        backend.destroy()


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
