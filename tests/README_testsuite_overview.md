# EffiDict test suite

This directory is the executable specification for EffiDict. It combines ordinary regression tests with expected-failure specifications for behavior tracked in `PLAN.md`.

This document explains the test suite itself: how it is organized, how it runs, and what each file verifies. Product implementation details are intentionally out of scope.

## High-level overview

The suite is organized into four layers.

### 1. Shared test infrastructure

- `conftest.py` defines the backend/policy matrix, object factories, teardown, leak detection, and backend call spying.
- `helpers.py` defines shared values, backend capability tables, tier inspection, strict equality, streaming checks, and storage-size measurement.
- `test_harness.py` tests that this infrastructure works.

### 2. Core correctness specifications

- `test_architecture.py` pins the public surface and responsibilities of the target `Store` and `Cache` types.
- `test_tier_invariants.py` checks consistency between cache and disk.
- `test_eviction_policies.py` specifies policy bookkeeping and victim ordering.
- `test_backend_contract.py` defines behavior common to every disk backend.
- `test_dict_conformance.py` compares generated EffiDict operation sequences with Python `dict`.
- `test_mapping_api.py` pins individual mapping, iteration, and equality rules.

### 3. Storage and ownership specifications

- `test_keyspace.py` covers key fidelity and storage-path safety.
- `test_persistence.py` covers stable addressing, close, reopen, and restart.
- `test_lifecycle.py` covers copying, ownership, garbage collection, close, and destruction.

### 4. Non-functional specifications

- `test_complexity.py` guards asymptotic work and storage growth.
- `test_memory.py` guards cache memory, process RSS, and streaming scans.
- `test_concurrency.py` covers threads, worker processes, and multiprocess access.

## How to interpret results

The suite is specification-first, so a healthy run contains more than passes.

- **PASS** protects behavior that currently works.
- **XFAIL** records a known deterministic gap or an API contract not yet implemented.
- **XPASS** means an expected-failure test passed.
- With `xfail(strict=True)`, XPASS fails the run so a completed specification cannot remain hidden behind a stale marker.
- `xfail(strict=False)` is used only where Hypothesis generation or scheduling makes reproduction probabilistic. Important defects also have deterministic strict tests.
- **SKIP** is used for unavailable optional dependencies or platform facilities.

Expected-failure reasons name their owning issue. They are part of test traceability, not permission to ignore arbitrary failures.

## Test matrix

`conftest.py` defines four backend classes:

- `SqliteBackend`
- `PickleBackend`
- `JSONBackend`
- `Hdf5Backend`

and seven replacement-policy classes:

- `RandomReplacement`
- `FIFOReplacement`
- `LIFOReplacement`
- `LRUReplacement`
- `MRUReplacement`
- `LFUReplacement`
- `MFUReplacement`

A test requesting both `backend_cls` and `policy_cls` runs over all 28 combinations. Parameter IDs contain class names, for example `SqliteBackend-LRUReplacement`.

HDF5 cases remain collected but are skipped when `h5py` is unavailable.

Tests use the full matrix only when behavior can vary by backend, policy, or tier placement. Plain mapping semantics generally use the default Pickle/LRU combination.

The Hypothesis suite additionally uses a cheaper cross-section:

- Every backend with LRU.
- Every remaining policy with Pickle.

Slow runners expand this to the full matrix and several cache sizes.

## Markers

- `slow`: long-running memory, complexity, or expanded conformance checks. Plain `pytest` includes them; use `-m "not slow"` for a shorter local run.
- `stress`: concurrent or scheduling-sensitive checks. CI repeats these tests.
- `keeps_storage`: disables the final storage-leak assertion for a test that intentionally preserves storage.

Marker names are registered with strict checking in `pyproject.toml`.

## Running the suite

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Common commands:

```bash
# Complete suite
python -m pytest

# Shorter local loop
python -m pytest -m "not slow"

# Concurrency specifications
python -m pytest -m stress

# One file
python -m pytest tests/test_backend_contract.py

# One test and all of its parameters
python -m pytest tests/test_tier_invariants.py::test_none_is_a_storable_value

# Detailed parameters and expected-failure reasons
python -m pytest -vv -rxX
```

## File-by-file guide

The order below is the recommended reading order: infrastructure first, then core behavior, storage semantics, and finally non-functional properties.

### `conftest.py`

This file provides the shared pytest fixtures.

#### Optional HDF5 handling

`h5py` is imported conditionally. `backend_marks()` adds a skip mark to HDF5
parameters when the dependency is absent, and `_backend_param()` gives every
backend a readable test ID.

#### Matrix fixtures

- `backend_cls` yields one backend class per parameter.
- `policy_cls` yields one policy class per parameter.
- Requesting both creates the 4 × 7 Cartesian product.

The fixtures return classes, not instances. Tests or object factories decide
where and how to construct them.

#### Temporary storage

`storage_dir` creates a fresh directory under pytest’s `tmp_path`.

`_open_stores` is a per-test registry of objects requiring teardown.

A `pytest_runtest_makereport` hook records any spec that dies on one issue's tagged stub while its xfail marker blames another, and the run then ends non-zero naming each one. The state lives on the `Config` rather than the module, because `pytester` runs its sub-pytest in the same process.

`storage_leak_detector` is autouse. After each test it:

1. Releases registered stores in reverse creation order.
2. Honors the `keeps_storage` opt-out.
3. Fails if anything remains in `storage_dir`.

This verifies library cleanup before pytest eventually removes the surrounding
temporary tree.

`release_store()` supports the lifecycle transition by trying `destroy()` first
and `close()` as a fallback. It propagates unrelated cleanup errors.

#### EffiDict factory

`make_dict` returns a factory that:

- Uses the active matrix classes when requested.
- Falls back to default Pickle/LRU classes otherwise.
- Allocates a unique path for every created store.
- Forwards item and byte limits and additional arguments.
- Registers the backend and facade for teardown.

#### Backend call spy

`backend_spy` exposes `BackendSpy` as a context-manager factory.

`BackendSpy` temporarily wraps public instance methods, records ordered calls
and per-method counts, delegates to real behavior, supports reset, and restores
the instance on exit. It deliberately leaves properties, static methods, and
class methods unchanged.

#### Nested pytest

The `pytester` plugin allows the harness tests to create temporary tests and run
an isolated pytest invocation.

### `helpers.py`

This file centralizes test data and assertions.

#### Value catalog

`VALUE_KINDS` contains fresh factories for:

- Integers, floats, booleans, strings, and `None`
- Empty, simple, and nested dictionaries
- Lists and tuples
- NumPy arrays and pandas DataFrames

`UNSUPPORTED` lists backend/value combinations that raise.

`LOSSY` lists combinations that are accepted but do not round-trip faithfully.

`supported_kinds`, `is_unsupported`, and `is_lossy` query these tables.

#### Compatibility helpers

`build_policy` inspects a policy constructor and supplies only parameters it
declares. This keeps test construction valid across the policy API transition.

`in_cache`, `cached_keys`, `cached_bytes`, and `on_disk` provide one shared
place for white-box tier inspection.

#### Equality

`assert_equal_value` performs recursive, type-strict comparison. It handles:

- pandas DataFrames and Series
- NumPy arrays and dtypes
- NaN values
- Nested dictionaries, lists, and tuples
- Scalar types that normal equality conflates

This ensures a storage round trip did not silently change representation.

#### Streaming and size

`EAGER_ITERATOR_TYPES` identifies iterators over containers that were already
fully built in memory.

`store_bytes` measures one storage file or all files in a storage tree.


# Test Harness

`test_harness.py` tests the testing infrastructure itself rather than primarily testing EffiDict features. Its purpose is to ensure broken fixtures or helpers cannot produce misleading results elsewhere.

### `_install_harness(pytester)`

Creates an isolated pytest sandbox and loads this repository’s `tests.conftest` plugin into it.

This lets tests run a second, temporary pytest session to verify collection, teardown failures, and markers without disrupting the main test run.

## Backend/policy matrix tests

### `test_matrix_collects_28_combinations`

Creates a temporary test using both `backend_cls` and `policy_cls`, then runs pytest against it.

It verifies that pytest actually executes:

- 4 backends
- 7 policies
- 28 total combinations

HDF5 cases may be skipped when `h5py` is unavailable, so passed and skipped results are added together.

This tests real pytest parametrization rather than merely checking `4 × 7 == 28`.

### `test_matrix_ids_are_readable_and_unique`

Runs pytest in collection-only mode and inspects generated test IDs.

It verifies:

- Exactly 28 cases are collected.
- Every case has a unique name.
- Names are readable, such as `SqliteBackend-LRUReplacement`.

## `make_dict` fixture tests

### `test_make_dict_roundtrips_across_the_matrix`

Runs across every backend/policy combination.

It stores and retrieves a nested value, confirming that `make_dict()` correctly connects the selected backend, policy, and `EffiDict`.

### `test_make_dict_honours_max_in_memory`

Creates a dictionary with space for two cached values and inserts three.

It verifies:

- Only two remain in memory.
- All three remain logically present in the complete dictionary.

### `test_make_dict_forwards_max_bytes`

Checks that the byte-budget argument reaches the created `EffiDict`. Otherwise, byte-budget tests could appear to run while testing nothing.

### `test_make_dict_defaults_when_unparametrized`

Checks that a test which does not request matrix fixtures receives `DEFAULT_BACKEND` and `DEFAULT_POLICY`.

### `test_make_dict_gives_each_dict_its_own_storage`

Creates two dictionaries and verifies they have different storage paths, preventing accidental test interference.

## Storage leak detector tests

These use isolated pytest runs because they intentionally cause teardown behavior that would otherwise fail the main suite.

### `test_leak_detector_fires_on_a_leaked_file`

Creates a file and deliberately leaves it behind.

It verifies that:

- The temporary pytest run fails.
- The failure names the leaked file.

This proves the autouse leak detector genuinely runs.

### `test_leak_detector_passes_when_storage_is_released`

Creates and uses an `EffiDict` without manually cleaning it up.

It verifies that the harness automatically releases the store and the test passes.

### `test_keeps_storage_marker_opts_out`

Deliberately leaves a file behind under `@pytest.mark.keeps_storage`.

It verifies that this marker disables the leak assertion for persistence tests that intentionally preserve storage.

## Equality-helper tests

### `test_assert_equal_value_handles_arrays_frames_and_nan`

Checks that `assert_equal_value()` correctly handles:

- Nested NumPy arrays
- pandas DataFrames
- `NaN`, even though normal `NaN == NaN` is false

It skips if NumPy or pandas is unavailable.

### `test_assert_equal_value_handles_string_arrays`

Checks string arrays separately because NumPy’s `equal_nan=True` cannot be used safely with string dtypes.

It verifies equal arrays pass and different arrays fail.

### `test_assert_equal_value_is_type_strict`

Parametrized with several unequal cases:

- Different nested values
- `1` versus `True`
- `1` versus `1.0`
- Tuple versus list
- Dictionaries with different keys

Each comparison must raise `AssertionError`.

### `test_assert_equal_value_rejects_numpy_scalar_widening`

Confirms that NumPy `int64(7)` is not considered the same type as Python `int(7)`, despite normal equality returning true.

### `test_assert_equal_value_rejects_array_dtype_changes`

Checks arrays whose element values are equal but dtypes differ, including:

- `float64 → float32`
- `float32 → float64`
- `int64 → int32`
- `int64 → float64`

This ensures silent precision or representation changes are detected.

## Backend value-support table tests

### `test_value_support_tables_match_reality`

Runs every value kind against every backend.

Depending on the capability tables, it expects one of three outcomes:

- **Unsupported:** serialization raises `TypeError`.
- **Lossy:** serialization succeeds, but strict equality fails after retrieval.
- **Supported:** the retrieved value equals the original exactly.

This re-tests the `UNSUPPORTED` and `LOSSY` declarations against actual backend behavior so the tables cannot silently become outdated.

The `finally` block attempts to destroy the backend regardless of success or failure.

### `test_support_tables_partition_every_kind`

Checks that every value kind belongs to exactly one category for each backend:

- Faithfully supported
- Unsupported
- Lossy

The union assertion catches missing kinds. The combined-size assertion catches overlap between categories.

## Backend spy tests

### `test_backend_spy_records_and_delegates`

Forces a value onto disk, then reads it while spying.

It verifies that:

- `deserialize()` was recorded once.
- Unrelated methods were not recorded.
- Call order is retained.
- The real backend operation still works.
- `reset()` clears recorded history.

### `test_backend_spy_restores_the_backend`

Checks that methods are wrapped only inside the `with` block.

After leaving the block:

- The original class method is active again.
- No patched method remains on the backend instance.

### `test_backend_spy_leaves_classmethods_alone`

Verifies that class methods such as `create()` and `open()` are not patched.

### `test_backend_spy_counts_writes_during_eviction`

Uses a one-entry cache and inserts two values.

The second insertion evicts the first, so it verifies exactly one backend `serialize()` call occurred.

### `test_drifted_xfail_reason_fails_the_run`

A sandbox spec whose marker cites issue 3.2 while the test dies on 1.4's stub.

The run must end non-zero and name the mismatch. A strict xfail is satisfied by *any* failure, so without this check the marker passes while the spec it guards is stopped at a gate nobody recorded — and the day 3.2 lands, nothing says the spec is still untested. Two markers in this suite had drifted that way.

### `test_xfail_reason_naming_its_own_cause_is_left_alone`

The same shape, with the marker naming the issue that actually raised.

The run must stay green. Most of the 580 expected failures in this suite die on a tagged stub, so a check that fired on all of them would be useless.

In short, this file proves six foundational pieces work correctly: matrix parametrization, dictionary construction, leak detection, value comparison, backend call spying, and the xfail reason drift check.

# Test Architecture
`test_architecture.py` specifies the intended Phase 1 architecture:

- `EffiDict` should be a public facade.
- `Store` should own key placement and coordinate disk with memory.
- `Cache` should own cached entries and eviction-policy bookkeeping.
- Eviction policies should not directly access storage.

Most behavioral tests are strict `xfail` specifications because `Store` and `Cache` currently exist mainly as stubs.

## API surface definitions

### `STORE_SURFACE` and `CACHE_SURFACE`

These tables define every required method and its exact signature.

For example:

```python
"put": ["key", "value", "dirty="]
```

means `Cache.put()` must accept required `key` and `value` parameters and an optional `dirty` parameter.

`STORE_PROPERTIES` requires `backend`, `cache`, and `owns_storage` to be read-only properties.

### `_signature_spec()`

Converts a Python signature into the compact format used by those tables.

It distinguishes:

- Required and optional parameters
- Positional-only parameters
- Keyword-only parameters
- `*args`
- `**kwargs`

This prevents an implementation from preserving method names while silently changing how callers must use them.

## Structural tests

These pass against the current stubs and protect the documented API during implementation.

### `test_surface_is_complete`

Runs once for `Store` and once for `Cache`.

It verifies:

- Every documented method still exists.
- Every method has the expected signature.
- Required properties remain properties.
- Store properties have no setters or deleters.

The read-only requirement prevents callers from replacing the backend or cache after construction, which would undermine `Store`’s ownership of key placement.

### `test_unimplemented_methods_name_their_issue`

Examines methods that still raise `NotImplementedError`.

It requires their error message to contain an issue reference such as:

```python
NotImplementedError("see issue #1.2")
```

Implemented methods are ignored. This allows the stubs to be replaced incrementally while keeping unfinished work traceable.

## Store and facade tests

### `test_store_can_be_built_over_a_backend_and_a_cache`

Strictly expected to fail until issue 1.2.

It verifies that a `Store` can be constructed from:

- A disk backend
- A cache

It then checks that `Store` exposes those exact objects and owns the underlying storage by default.

This is tested across all backends.

### `test_both_tiers_are_required_to_construct`

Checks the current `EffiDict` constructor.

Both `disk_backend` and `replacement_strategy` must be required arguments. Calling `EffiDict()` without them must raise a clear `TypeError` identifying a missing argument.

This prevents misleading defaults such as `None` that only fail later with an internal `AttributeError`.

### `test_effidict_delegates_placement_to_a_store`

Strictly expected to fail until issue 1.2.

It requires the future `EffiDict` to:

- Have a `_store` that is a `Store`.
- Stop exposing its own `memory` alias.
- Ask the store whether a key is cached.

The test enforces one owner of key placement. Currently, both `EffiDict` and replacement strategies participate in memory/disk placement.

## Cache clean/dirty state

### `test_cache_tags_entries_clean_or_dirty`

Expected to fail until clean/dirty tracking is implemented.

It inserts:

- A dirty entry representing a new or modified value.
- A clean entry representing a value already persisted on disk.

It checks:

- `is_dirty()` reports each state correctly.
- `dirty_items()` returns only dirty entries.
- `mark_clean()` changes the state.

This matters because clean eviction should simply discard memory, while dirty eviction requires writing to disk.

## `RecordingPolicy`

This is a small test implementation of `EvictionPolicy`.

It records:

- Insertions
- Accesses
- Removals
- Clears

It also maintains an ordered key list and names victims from the front.

Tests use it because they need to observe whether `Cache` notified its policy. Real policy state is private, making operations such as “peek without promoting” otherwise difficult to verify.

## Cache budget tests

### `test_cache_reports_what_must_be_shed_for_the_item_budget`

Passes.

- Inserts five entries into a cache limited to two.
- Requires the cache to retain all five initially.
- Requires `evict_candidates()` to name exactly three distinct, existing keys.
- Discards those keys.
- Verifies the cache is then within budget and names nothing further.

The cache must not evict entries itself because a dirty entry might need to be persisted first. Instead, it tells `Store` which entries must be shed; `Store` writes dirty values and then discards them.

Split out of a combined budget test. The byte half dies at `nbytes()`, and a single failure marks a whole test expected-to-fail, so this half reported nothing for as long as they shared an id — while the behaviour it pins already worked.

### `test_cache_reports_what_must_be_shed_for_the_byte_budget`

Expected to fail until issue 1.4.

- Inserts roughly 20 KiB into a cache limited to 4 KiB.
- Requires `nbytes()` to report the excess.
- Requires enough eviction candidates to return within budget.

### `test_cache_sheds_exactly_the_surplus_in_victim_order`

Passes. Parametrized over FIFO, LIFO, LRU, MRU, LFU and MFU.

- Fills a two-item cache with keys `a` to `e`, then reads `b` twice and `d` once.
- Requires `evict_candidates()` to return exactly the first three keys of that policy's own ordering, written out as a literal list per policy.
- Requires the cache to still hold all five afterwards, because naming a key is not removing it.

This covers the branch `Store` never reaches today: it spills after every `put`, so the surplus is always one and the single-key fast path answers. A bulk `update` (issue 4.1) or the byte budget (issue 1.4) would reach it.

### `test_cache_candidates_survive_being_discarded_while_walked`

Passes.

- Uses `_LiveOrderPolicy`, a conforming policy whose `victims()` walks its live `OrderedDict` rather than a snapshot.
- Discards each candidate as it is yielded, the way `Store._spill` does.
- Requires the walk to survive, which it can only do if `Cache` read the candidates out of the policy before handing them over.

The seven shipped policies and `RecordingPolicy` all copy their ordering first, so none of them can tell an eager hand-off from a lazy one.

### `test_cache_measures_values_with_the_size_estimator`

Expected to fail until issue 1.4.

It supplies a custom estimator that reports every value as 10 bytes.

After inserting four values, the cache must report 40 bytes and identify eviction candidates against a 25-byte budget.

This proves the cache uses the provided estimator instead of incorrectly using `len(value)`.

## Cache read behavior

### `test_cache_peek_does_not_promote`

Expected to fail until issue 1.2.

It verifies that `peek("a")`:

- Returns the value.
- Does not call `policy.on_access()`.
- Does not reorder eviction candidates.

`peek()` is intended for internal inspection, flushing, and scanning without changing recency or frequency.

### `test_cache_get_reads_and_records_the_access`

The complement of the `peek()` test.

It requires `get("a")` to:

- Return the value.
- Call `policy.on_access("a")` exactly once.
- Raise `KeyError` for a missing key.

Raising `KeyError` is important because returning `None` would make a missing key indistinguishable from a cached value whose actual value is `None`.

## Store `pop()` semantics

### `test_store_pop_distinguishes_an_omitted_default_from_none`

Expected to fail until issue 1.2.

It requires dictionary-compatible behavior:

- `pop(existing)` returns and removes the value.
- `pop(missing)` raises `KeyError`.
- `pop(missing, None)` returns `None`.
- `pop(missing, "fallback")` returns `"fallback"`.
- A stored `None` is returned as a real value, not treated as missing.

This requires an internal sentinel such as `MISSING`; `default=None` alone cannot distinguish an omitted default from an explicitly supplied `None`.

Overall, this file pins both the shape and responsibilities of the target architecture before the refactor is implemented.

# Test Tier Invariants

`test_tier_invariants.py` checks correctness across EffiDict’s two storage tiers:

- In-memory cache
- Persistent disk backend

It uses white-box helpers to inspect each tier independently, because public operations can hide temporary inconsistencies.

## Setup helpers

### `_spill_to_disk(d, key)`

Forces a selected key into disk only:

1. Reads its current value.
2. Serializes it directly to the backend.
3. Removes it from the cache.
4. Verifies it is on disk and not cached.

It bypasses normal eviction because different policies would choose different victims.

### `_seed_in_both_tiers(d, key, value)`

Places the same key and value in both tiers:

1. Writes it normally.
2. Spills it to disk.
3. Writes it into the cache again.

This creates the exact precondition needed for deletion tests.

## I4: deletion affects every tier

### `test_pop_removes_key_from_both_tiers`

Strict `xfail`.

Starts with a key in both tiers, calls `pop()`, then verifies:

- The returned value is correct.
- Neither tier retains the key.
- Membership and lookup report it missing.

The current `pop()` removes the cache copy but can leave the disk copy behind.

### `test_delitem_removes_key_from_both_tiers`

Regression test for a pre-0.1.0 bug.

A single `del d[key]` must remove both cache and disk copies. Previously an `if/else` removed only one copy.

### `test_delitem_missing_key_raises`

Checks dictionary-compatible behavior: deleting an absent key raises `KeyError`.

### `test_clear_removes_from_both_tiers`

Creates enough entries to force disk spills, calls `clear()`, and verifies:

- Logical length is zero.
- Iteration returns no keys.
- The backend is empty.
- Every key is absent from both tiers.

## I5: safe eviction ordering

### `test_no_key_is_ever_absent_from_both_tiers`

Strict `xfail`.

A dirty eviction must:

1. Write the value to disk.
2. Only then remove it from cache.

The test temporarily replaces `serialize()` with a blocking function. While the disk write is paused, it inspects the chosen victim’s tier state.

At that moment, the victim must still be in at least one tier. If it is in neither, a concurrent reader could incorrectly see it as missing.

Threading events make this deterministic:

- `started` signals that serialization has begun.
- `release` allows serialization to finish.
- `finally` always releases the blocked writer.

The test also confirms the write eventually succeeds rather than accepting a permanently cached value after a failed write.

### `test_clean_eviction_performs_no_io`

Seeds values directly on disk, then reads them all.

Because these values are already persisted and unmodified, evicting their cache copies should require no writes.

The backend spy counts both:

- `serialize()`
- Future `write_many()`

Promoting policies are currently expected to fail because they do not mark disk-loaded entries as clean. They unnecessarily rewrite those values during eviction.

### `test_reading_a_cached_key_never_writes`

Reads one cached key repeatedly and verifies no backend writes occur.

A cache hit may update policy metadata, but it must not perform disk I/O.

## I3: membership is a point lookup

### `test_membership_of_a_cached_key_never_touches_the_backend`

Checks that `"hot" in d` short-circuits when the key is cached.

The backend spy must record no calls at all.

### `test_membership_never_lists_the_keyspace`

Strict `xfail`, tested for:

- A missing key
- A key residing only on disk

Membership must not call `keys()` or `iter_keys()` because enumerating every key makes a point lookup O(N).

The intended backend operation is a direct existence check such as `has(key)`.

## I1: value and tier correctness

### `test_none_is_a_storable_value`

Verifies that `None` is treated as a real value rather than as “missing”:

- Lookup returns `None`.
- Membership is true.
- Length includes the key.
- Deletion makes it genuinely absent.

### `test_none_survives_a_spill_to_disk`

Forces a `None` value onto disk and out of cache, then verifies it still reads back as `None`.

This prevents a cache hit from hiding backend mishandling.

### `test_eviction_writes_the_current_value`

Uses the real eviction path.

With a two-entry cache, inserting a third key must evict something. The test does not dictate which key; it checks that whichever victim was chosen was serialized with its current corresponding value.

### `test_reassignment_supersedes_the_stale_disk_copy`

Strict `xfail`.

Creates this state:

- Disk contains `"first"`.
- Cache contains a newer `"second"`.

After `flush()`, disk must contain `"second"`. It currently fails because flushing is not implemented.

## I7: durability

### `test_flush_persists_dirty_entries`

Strict `xfail`.

Writes several values that remain only in cache, calls `flush()`, then verifies every value exists correctly on disk.

This defines the durability guarantee: after `flush()` returns, dirty cached values are persisted.

## I8: byte budget

### `test_cache_respects_byte_budget`

Strict `xfail`.

Creates many distinct 64 KiB strings with:

- A high item limit
- A 512 KiB byte limit

It checks that cached bytes do not exceed the budget by more than one newly inserted entry.

Distinct strings are important: reusing one shared string would consume much less actual memory despite appearing large when lengths are summed.

The one-entry allowance reflects the planned algorithm: insertion may briefly exceed the budget before eviction candidates are processed.

## Abstract-base tests

### `test_abstract_bases_cannot_be_instantiated`

Runs against:

- `DiskBackend`
- `EvictionPolicy`
- Legacy `ReplacementStrategy`

It verifies:

- Each class declares abstract methods.
- Instantiating the base raises an abstractness-related `TypeError`.
- An incomplete subclass is also rejected.

Checking the error text matters because `DiskBackend()` could otherwise raise merely because a constructor argument is missing, falsely making the test appear to prove abstractness.

`Store` and `Cache` are intentionally absent because they are concrete components configured through composition, not families of implementations.

### `test_concrete_backends_remain_instantiable`

Confirms that making `DiskBackend` abstract did not accidentally prevent the four real backend classes from being instantiated.

Overall, this file protects the central rule: a key must remain correctly addressable and durable while moving between memory and disk, without unnecessary scans or writes.

# Test Eviction Policies

`test_eviction_policies.py` specifies the future eviction policies as pure bookkeeping components.

A policy should only answer questions such as:

- Which keys are tracked?
- Which key should be evicted next?
- How does an access affect ordering or frequency?

It must not store values, access disk, or perform eviction itself. `Cache` owns values, while `Store` handles persistence.

Most tests are strict `xfail` because the future classes such as `LRUPolicy` and `LFUPolicy` do not exist yet. The current `*Replacement` classes combine policy logic with backend I/O.

## Policy contracts

The intended policy API is:

- `on_insert(key)` — begin tracking a key.
- `on_access(key)` — record a use.
- `on_remove(key)` — forget the key completely.
- `victim()` — return the next key to evict without removing it.
- `victims()` — iterate all keys in eviction order without mutation.
- `clear()` — remove all bookkeeping.

Insertion counts as the first use. Frequency ties are resolved by insertion order.

## Shared policy lists

- `POLICY_NAMES` contains Random, FIFO, LIFO, LRU, MRU, LFU, and MFU.
- `DETERMINISTIC` excludes Random because it has no fixed victim sequence.
- `ALLOWED_INIT_PARAMS` is empty because policies require no backend, capacity, or other configuration.

## Test-support helpers

### `_instance_state()`

Collects all state stored on a policy, including both:

- Normal `__dict__` attributes
- Attributes declared through `__slots__`

This allows cleanup tests to work regardless of implementation style.

### `_state_snapshot()`

Creates a detached copy of policy state.

A deep copy is necessary because comparing two references to the same mutable list or dictionary would fail to detect mutations.

### `_residual_state()`

Finds non-empty state containers after removal or clearing.

It works with any sized container, including lists, dictionaries, sets, tuples, and deques.

### `_reachable_keys()`

Recursively searches policy state for keys that should have been forgotten.

This catches implementations using structures without `len()`, such as custom linked lists, which `_residual_state()` might miss.

### `_FakeBackend`

Records any backend calls made against it.

It is used to prove that policies reject backend injection and never access storage.

### `_policy_class(name)`

Dynamically resolves classes such as:

```python
effidict.policies.LRUPolicy
```

Resolution happens inside each test so missing future classes produce clean `xfail` results instead of preventing the entire module from being collected.

### `_apply(policy, ops)`

Applies a sequence of insert and access operations.

### `_victim_sequence(policy)`

Repeatedly:

1. Calls `victim()`.
2. Records the returned key.
3. Calls `on_remove(key)`.

It stops when `victim()` raises `KeyError`.

The loop is bounded at 50 iterations so a broken policy cannot hang the suite forever.

## Victim-order tests

### `test_victim_order`

Runs two scenarios across all deterministic policies.

The recency scenario inserts `a`, `b`, `c`, then accesses `a`.

Expected examples:

- FIFO: `a, b, c`
- LIFO: `c, b, a`
- LRU: `b, c, a`
- MRU: `a, c, b`

The frequency scenario gives keys different access counts and checks LFU/MFU ordering alongside the insertion/recency policies.

Expected sequences are literal rather than computed from implementation logic, preventing the test from duplicating the same bug.

### `test_lfu_evicts_the_least_frequently_used`

Makes `hot` heavily used, `warm` somewhat used, and `cold` untouched.

LFU must select `cold`.

### `test_mfu_evicts_the_most_frequently_used`

Uses the same setup, but MFU must select `hot`.

## Current MFU regression

### `test_mfu_replacement_keeps_frequency_across_a_value_update`

Unlike most tests in this file, this directly exercises the currently shipped `MFUReplacement`.

It establishes:

- `a` has four uses.
- `b` has two uses.
- Updating `a`’s value should not reset its frequency.

After adding `c`, MFU should evict `a`, the hottest key.

The current implementation resets `a`’s count during the update, so it incorrectly evicts `b`.

`_NullBackend` absorbs the disk write performed by the legacy replacement strategy.

## Backend independence

### `test_policy_never_touches_the_backend`

Checks policy purity in four ways:

1. Constructor signatures may not contain variadic arguments that could hide backend injection.
2. Constructors may not declare unexpected parameters.
3. Constructing with a fake backend—positionally or by common keyword names—must raise `TypeError`.
4. After normal use, the policy must not retain any backend-like object.

This enforces the architecture: policies decide order; they do not perform storage operations.

## Removing policy state

### `test_forget_removes_all_policy_state`

Tracks two distinctive keys, accesses one several times, then calls `on_remove()`.

It verifies:

- The removed key cannot become a victim.
- Removing all keys makes `victim()` raise `KeyError`.
- No non-empty bookkeeping containers remain.
- Neither key is still reachable anywhere within policy state.

This is especially important for LFU and MFU, which may maintain both an ordering structure and a frequency table.

## Random policy

### `test_random_policy_only_ever_names_tracked_keys`

Because Random has no exact deterministic order, it tests properties instead:

- Every selected victim must be tracked.
- Across 200 fresh runs, all four keys should eventually be selected.
- Draining one policy must return every tracked key exactly once.

This distinguishes a genuinely random policy from one that merely returns an unpredictable but incomplete subset.

## Clearing policy state

### `test_clear_drops_everything`

Adds and accesses keys, then calls `clear()`.

It verifies:

- `victim()` raises `KeyError`.
- No non-empty containers survive.
- Comparable state matches a fresh policy instance.

The fresh-instance comparison catches state such as counters that remain nonzero even after key containers are emptied.

## Enumerating victims

### `test_victims_enumerates_in_eviction_order`

Calls `victims()` to get the full eviction sequence, then independently drains the policy through repeated `victim()` and `on_remove()` calls.

Both sequences must match.

This gives `Cache.evict_candidates()` a way to inspect multiple candidates without prematurely removing dirty entries.

### `test_victims_records_nothing`

Ensures `victims()` is read-only inspection.

It checks policy state:

- After retrieving only the first victim
- After consuming the complete iterator

State must remain exactly unchanged. The partial-consumption check is important because mutations during a full traversal might cancel each other and leave final state looking unchanged.

Overall, this file defines eviction policies as deterministic, storage-independent state machines. The only exception is Random, whose exact order is intentionally unspecified but whose safety properties remain constrained.

# Test Backend Contract

`test_backend_contract.py` defines behavior shared by all four disk backends:

- SQLite
- JSON
- Pickle
- HDF5

It covers value support, error semantics, storage maintenance, batch APIs, direct lookup, streaming, and awkward string values.

Many tests are strict `xfail` specifications for planned backend improvements.

## Shared setup

### `backend` fixture

Creates the currently parametrized backend in a temporary storage location, yields it to the test, then destroys it.

### `_value_for(kind)`

Creates a value from `VALUE_KINDS`, skipping the case if its optional NumPy or pandas dependency is unavailable.

### `_policy_state(store)`

Captures all state affecting the next eviction:

- Cache ordering
- LFU/MFU frequency counters

It is used to compare bulk loading with ordinary writes. Comparing only cached keys would miss different ordering or frequency state.

## Value-domain tests

### `test_backend_declares_its_value_domain`

Strict `xfail`.

Requires every backend to expose `supports(value)`, allowing callers to ask whether a value can be stored faithfully.

For every value kind, the test compares:

- What `supports()` claims
- Whether serialization, deserialization, and strict equality actually succeed

“Supported” means faithful round trip, not merely “serialization did not raise.” For example, converting a tuple to a list is not faithful support.

### `test_value_domain_is_enforced`

Runs only for backends with unsupported value types.

It verifies that unsupported values raise `TypeError` whose message identifies:

- The backend that rejected the value
- The rejected value’s type

This is stricter and more actionable than leaking a low-level codec error such as “not JSON serializable.”

### `test_value_roundtrip_fidelity`

Runs every value kind against every backend.

- Unsupported combinations are skipped.
- Known lossy combinations receive a strict runtime `xfail`.
- Supported combinations must return the same value and exact type.

This detects conversions such as:

- Tuple → list
- Python `int` → NumPy `int64`
- DataFrame → ndarray

## Error contract

### `test_missing_key_raises_keyerror_not_something_else`

Checks that requesting a nonexistent key always raises `KeyError`, both on an empty backend and after another key has been stored.

Filesystem errors or codec-specific exceptions must not leak for an ordinary miss.

### `test_decode_failure_is_not_reported_as_missing_key`

Injects a deterministic decoder failure by patching each backend’s decoding function.

The injected failure is a `KeyError`, representing an internal decoding problem. The backend must convert it into a different exception type because callers interpret any `KeyError` as “key absent.”

The resulting exception must also retain the original failure as itself, its cause, or its context.

All current backends are expected to fail:

- HDF5 rewrites it as `KeyError(key)`.
- The others allow the decoder’s `KeyError` to escape.

Either way, corruption is indistinguishable from a missing key.

## Storage maintenance

### `test_compact_reclaims_deleted_space`

Strict `xfail`.

Writes hundreds of values, records storage size, deletes every key, then calls `compact()`.

After compaction, the physical storage must shrink below half its previous size.

Deletion alone often leaves database pages or HDF5 allocation behind. Compaction is explicit because reclaiming space can be expensive.

### `test_batch_and_single_writes_are_equivalent`

Strict `xfail`.

Loads identical data in two ways:

- Repeated `d[key] = value`
- `load_from_dict(items)`

It compares:

- Cache ordering
- LFU/MFU frequency state
- Logical keys
- Retrieved values

For Random policy, exact retained keys may legitimately differ, so only cache population size is compared.

The current bulk loader bypasses cache and policy accounting, leaving a different tier state.

## Batch API tests

`BATCH_ITEMS` is a common six-item dataset. The tests define batch behavior through equivalence with existing single-item operations.

### `test_write_many_matches_individual_writes`

Strict `xfail`.

Compares `write_many()` with repeated `serialize()` calls. Both must leave the same keys and values.

### `test_read_many_matches_individual_reads`

Strict `xfail`.

Requires `read_many(keys)` to:

- Return a dictionary
- Return every requested key
- Return the same values as individual `deserialize()` calls

### `test_read_many_is_not_quieter_than_deserialize`

Strict `xfail`.

If any requested key is missing, `read_many()` must raise `KeyError`, just as individual `deserialize()` would.

Silently omitting missing keys would make absence indistinguishable from other conditions.

### `test_delete_many_matches_individual_deletes`

Strict `xfail`.

Deletes half the dataset and verifies:

- Exactly those keys disappear.
- Unnamed keys remain.
- Surviving values are unchanged.

## Direct lookup and counting

### `test_has_agrees_with_keys`

Strict `xfail`.

Checks that `has(key)` returns:

- `True` for every stored key
- `False` for an absent key

This test checks correctness only. Separate complexity tests ensure `has()` does not implement the answer by scanning `keys()`.

### `test_count_agrees_with_keys`

Strict `xfail`.

Checks that `count()` equals `len(keys())`:

- After insertion
- After deleting one key

## Streaming key iteration

### `test_iter_keys_streams_the_same_keys_as_keys`

Strict `xfail`.

It checks several properties of `iter_keys()`:

1. It returns an actual iterator.
2. Creating it does not call `keys()`.
3. It is not an iterator over an already-materialized container.
4. Consuming it does not call `keys()`.
5. It yields exactly the same keys as `keys()`.
6. Once exhausted, it remains exhausted.

The test temporarily wraps `backend.keys()` to count calls.

It also rejects types such as `list_iterator` because this implementation is not truly streaming:

```python
return iter(self.keys())
```

That loads the complete keyspace before iteration and can consume excessive memory.

The test acknowledges a limitation: a generator could internally call `os.listdir()` after iteration begins and still pass. Truly proving backend-level laziness requires implementation-specific checks.

## Degenerate string values

### `test_degenerate_string_values_round_trip`

Tests two legal but difficult Python strings:

- A string containing NUL: `"a\0b"`
- A string containing a lone Unicode surrogate

Pickle, JSON, and SQLite should round-trip them.

HDF5 receives a strict `xfail` because its UTF-8 C-string representation rejects:

- NUL with `ValueError`
- Lone surrogate with `UnicodeEncodeError`

These cases are separate from general conformance tests because the generated value strategy intentionally excludes them.

Overall, the file requires each backend to provide the same predictable public contract even though their underlying formats and codecs differ.

# Test Dict Conformance

`test_dict_conformance.py` uses differential testing: it performs the same generated operations on:

- A normal Python `dict`—the reference.
- An `EffiDict`—the system under test.

After every operation, their results, exception types, keys, lengths, and values must agree.

## Why Hypothesis state machines?

Instead of testing a few manually selected sequences, Hypothesis generates sequences such as:

```text
set a
set b
delete a
get missing
overwrite b
check length
clear
```

When it finds a failure, it shrinks the sequence to a small reproducer.

A `RuleBasedStateMachine` contains:

- `@rule` methods: operations Hypothesis may perform.
- `@invariant` methods: conditions checked after every operation.
- Bundles: values created by earlier rules and reused later.

## Generated keys and values

### `KEYS`

Generates short, safe lowercase ASCII keys.

It excludes problematic keys such as path separators, uppercase/case-folding cases, and Unicode normalization cases. Those are tested separately in `test_keyspace.py`.

This keeps these tests focused on dictionary behavior.

### `KIND_STRATEGIES`

Defines generated values for supported categories:

- Integers, floats, booleans, strings, `None`
- Dictionaries and nested dictionaries
- Lists and tuples
- NumPy arrays and DataFrames where available

### `_values(backend_cls)`

Builds a strategy containing only value types that the selected backend round-trips faithfully.

This avoids having mapping tests fail because of separate serialization defects such as tuple-to-list conversion.

## `CoreConformance`

This is the base state machine. It tests:

- `d[key] = value`
- `d[key]`
- `del d[key]`
- `key in d`
- `len(d)`
- `d.keys()`

### Setup

Each Hypothesis run creates:

- `self.sut`: a configured `EffiDict`
- `self.ref`: an empty Python dictionary
- A backend-specific value strategy

Teardown releases the EffiDict’s storage.

### `_mirror(op)`

Runs the same operation against both dictionaries.

It compares:

- Exception types, if either operation raises.
- Return values, using type-strict `assert_equal_value()`.

For example, if Python `dict` raises `KeyError` but EffiDict returns `None`, the comparison fails.

### `setitem`

Generates both fresh and previously written keys.

Reusing keys deliberately exercises reassignment. Drawing only random fresh keys would rarely test overwrite behavior.

The reference receives a deep copy so mutation or backend behavior cannot cause both sides to share the same object.

### `getitem`, `delitem`, `contains`, and `length`

Each applies the operation through `_mirror()` so values and errors must match normal dictionary behavior.

### `same_contents` invariant

After every generated operation, it verifies:

- Both stores have the same keys.
- Both have the same length.
- Every expected key is present.
- Every value matches strictly.

This catches corruption that may not appear in the immediately executed operation.

## Specialized state machines

Each extends `CoreConformance` with one behavior group. Splitting them allows known defects to have separate expected-failure markers.

### `PopConformance`

Adds:

- `pop(key, default)`
- `pop(key)` without a default

It exposes two current defects:

- Missing `pop(key)` returns `None` instead of raising `KeyError`.
- A key present in both tiers may only be removed from memory, leaving its disk copy behind.

### `ClearConformance`

Adds generated calls to `clear()` and verifies the result against `dict.clear()`.

### `MappingApiConformance`

Adds broader mapping operations:

- `get()`
- `get(key, default)`
- `setdefault()`
- `update(mapping)`
- `update(iterable_of_pairs)`
- `update(**keywords)`
- `update(mapping, **keywords)`
- `popitem()`

Existing keys are deliberately reused to test replacement and non-mutating branches.

The combined mapping-and-keywords test verifies standard dictionary precedence: keyword values override the positional mapping when both contain the same key.

These operations currently raise `NotImplementedError`.

### `OrderConformance`

Adds insertion-order invariants:

- `list(EffiDict)` must equal `list(dict)`.
- `list(EffiDict.keys())` must equal `list(dict.keys())`.

Current key collection uses set-union ordering, so it can differ from insertion order.

## Execution settings

### `FAST`

Uses:

- 8 generated examples
- 12 state-machine steps per example

This is intended for regular development runs.

### `DEEP`

Uses:

- 120 examples
- 40 steps per example

This provides much broader exploration but is marked `slow`.

Deadlines are disabled because real disk I/O makes wall-clock limits unreliable.

### Cache sizes

Core and clear behavior are tested with:

```python
[1, 2, 10, 1000]
```

These exercise:

- Constant eviction pressure
- Small-cache behavior
- Moderate caching
- No practical eviction for generated examples

## Test matrix

### `CROSS_SECTION`

The normal test run avoids the entire 4×7 product while still touching every component:

- Every backend with LRU
- Every other policy with Pickle

This produces 10 unique combinations: four backend/LRU cases plus six additional Pickle/policy cases.

### `_bind()` and `_run()`

Hypothesis state machines receive configuration through class attributes.

`_bind()` dynamically creates a subclass containing the selected backend, policy, cache size, and factory. `_run()` executes that configured machine.

## Core and clear runners

### `test_core_conformance`

Runs `CoreConformance` over the cross-section and all four cache sizes.

### `test_clear_conformance`

Runs `ClearConformance` over the same combinations and sizes.

### `test_core_conformance_full_matrix`

Slow version covering all 28 backend/policy combinations and all four cache sizes.

### `test_clear_conformance_full_matrix`

Equivalent full-matrix sweep for `clear()`.

## Expected-failure runners

### `test_pop_conformance`

Runs `PopConformance` over the cross-section.

It uses `xfail(strict=False)` because Hypothesis may not generate a sequence that exposes the bug within every small run.

### `test_mapping_api_conformance`

Runs the mapping operations over the cross-section.

It is non-strict `xfail` because generated state-machine failures are probabilistic.

### `test_order_conformance`

Checks iteration order over the cross-section.

It is also non-strict because a generated key sequence can coincidentally produce the same set order as insertion order.

### Full-matrix variants

The following slow tests run those machines across all 28 combinations:

- `test_pop_conformance_full_matrix`
- `test_mapping_api_conformance_full_matrix`
- `test_order_conformance_full_matrix`

## Deterministic `pop()` regression

### `test_pop_without_default_raises_keyerror`

Strict `xfail`.

Unlike the generated pop machine, this directly and reliably verifies:

- `pop(missing)` raises `KeyError`.
- `pop(missing, None)` returns `None`.
- `pop(missing, fallback)` returns the fallback.
- A stored `None` is returned as data.
- After removing that stored `None`, a second pop raises `KeyError`.

This test distinguishes an omitted default from an explicitly supplied `None`.

## Deep runners

### `test_core_conformance_deep`

Runs the high-effort settings against Pickle + LRU for every cache size.

### `test_clear_conformance_deep`

Runs a high-effort clear machine against Pickle + LRU with a two-entry cache.

These concentrate more generated sequences on one representative configuration rather than making the already large full matrix even more expensive.

Overall, this file asks: “For safe keys and faithfully supported values, can a caller distinguish EffiDict from Python’s built-in dictionary?”

# Test Mapping API

`test_mapping_api.py` pins EffiDict’s explicit Python mapping behavior. The Hypothesis conformance suite tests generated sequences; this file provides deterministic, focused regression tests whose failures identify one specific contract.

Most tests use simple string keys and values so serialization and keyspace issues do not interfere.

## Iteration tests

### `test_iter_returns_independent_iterators`

Strict `xfail`.

Calls `iter(d)` twice and verifies:

- Two different iterator objects are returned.
- Both begin at the same key.
- Advancing one does not advance the other.

Currently, EffiDict returns itself as its iterator and stores one shared cursor, so all iterations interfere.

### `test_nested_iteration`

Strict `xfail`.

With four keys, nested iteration should produce 16 pairs:

```python
[(outer, inner) for outer in d for inner in d]
```

The current shared iterator causes the inner loop to exhaust the outer loop’s cursor, producing an incomplete result.

### `test_iteration_order_is_insertion_order`

Strict `xfail`, run across the backend/policy matrix.

It inserts 12 keys and requires insertion order from:

- `d.keys()`
- `iter(d)`
- `d.items()`

The current implementation combines cache and disk keys using sets, losing insertion order.

Twelve keys make accidental agreement with set order extremely unlikely.

### `test_iteration_order_is_stable_across_processes`

Strict `xfail`.

Runs the same writes in subprocesses with three different `PYTHONHASHSEED` values.

It verifies:

- Every process produces the same key order.
- That stable order is insertion order.

Set iteration depends on salted string hashes, so current output can change between processes.

## Mapping surface tests

### `MAPPING_METHODS`

Lists the intended mapping and dict-style methods, including:

- `get`, `setdefault`, `pop`, `popitem`, `update`
- `keys`, `values`, `items`
- Iteration, length, containment
- `reversed`, union (`|`), and `fromkeys`

### `test_mutablemapping_surface_is_complete`

Strict `xfail`.

It requires EffiDict to be a `MutableMapping`, not merely an object implementing selected methods.

It also verifies that key operations actually work:

- `get()` with and without defaults
- `setdefault()`
- `update()`
- `popitem()`

Checking only `hasattr()` would allow methods to remain unusable stubs.

### `test_update_mapping_parameter_is_positional_only`

Checks a subtle `dict.update()` signature rule.

Python treats this as storing a key named `"other"`:

```python
d.update(other=5)
```

Therefore, the positional mapping parameter named `other` must be positional-only. Otherwise, `other=5` binds that parameter rather than being processed as a key/value pair.

A generated test is unlikely to randomly choose exactly `"other"`, so the signature needs a direct test.

### `test_dict_extras_behave_like_dict`

Strict `xfail`.

Checks three dict-specific operations that `MutableMapping` does not provide:

- `reversed(d)` returns reverse insertion order.
- `d | other` returns merged contents without mutating `d`.
- `d.fromkeys(keys, value)` creates the expected contents.

Each result is compared with Python `dict`.

## Equality tests

### `EQUALITY_CASES`

Contains comparisons against:

- An equal plain dictionary
- Dictionaries with changed, missing, or extra entries
- An empty dictionary
- An integer
- `None`
- A list of pairs

Expected results are derived from `BASE == other`, using `dict` itself as the oracle.

### `test_equality_matches_dict`

Checks:

- `d == other`
- `d != other`
- Reflected comparison: `other == d`

Only the equal plain-dictionary case receives a strict runtime `xfail`. Current EffiDict equality rejects anything that is not another EffiDict, so unequal operands happen to return the correct answer while an equal ordinary mapping incorrectly returns `False`.

### `test_equality_between_stores_is_by_contents`

Pins behavior that already works:

- Stores with identical contents compare equal.
- A store equals itself.
- Stores with differing contents compare unequal.

This prevents the fix for plain-dictionary equality from regressing store-to-store equality.

### `test_eq_does_not_mutate_either_operand`

Compares two equal stores and verifies that equality causes:

- No backend writes on either side.
- No changes to either cache’s membership.

Current equality walks items through normal reads. Promoting policies can pull disk values into memory and evict others, so those policy combinations receive a strict runtime `xfail`.

Both operands are monitored because equality reads from each one.

### `test_unhashable_by_design`

Checks that EffiDict cannot be hashed or used as a dictionary key.

Mutable mappings must be unhashable because their contents—and therefore equality—can change.

It verifies:

```python
EffiDict.__hash__ is None
hash(d)  # raises TypeError
```

## Streaming facade iteration

### `test_facade_iter_keys_streams_the_whole_keyspace`

Strict `xfail`.

Requires `d.iter_keys()` to:

- Return an iterator.
- Avoid an iterator over a prebuilt container.
- Perform no backend work merely when the iterator is created.
- Yield every key across cache and disk.
- Yield each key exactly once.

The test deliberately places one key in both tiers. This ensures an implementation that simply concatenates cached and disk keys fails because it would produce a duplicate.

The cache is smaller than the keyspace, ensuring both tiers genuinely participate.

Overall, the file specifies that EffiDict should behave like a modern ordered, mutable Python mapping—not merely expose similarly named methods.

# Test Keyspace

`test_keyspace.py` tests whether backend keys behave like dictionary keys rather than accidentally becoming filesystem paths.

This matters because:

- Pickle and JSON derive filenames from keys.
- HDF5 interprets keys as dataset paths.
- SQLite stores keys as ordinary column values.

The file checks both correctness and filesystem containment.

## Key expectation tables

### `KEY_PROBES`

Defines unusual but legal string keys:

- `"a/b"` — path separator
- `"../escaped"` — traversal
- `""` — empty
- 300-character string — overlong filename
- `"."` and `".."` — directory-like names
- `"a\0b"` — embedded NUL

### `BROKEN_KEYS`

Records which backends currently fail to round-trip each probe.

The table is literal so it defines expected behavior rather than automatically accepting whatever the implementation currently does.

### `TRAVERSAL_ESCAPES`

Identifies backends that physically write outside their storage directory for `..` keys:

- Pickle
- JSON

HDF5 also mishandles traversal-shaped keys, but it creates an internal group named `..` inside its file rather than escaping into the host filesystem.

## Containment helpers

### `_contains(root, path)`

Checks path containment using `os.path.commonpath()`.

A simple string-prefix check would be wrong because `/store-escaped` starts with `/store` but is a sibling, not a child.

### `_is_inside(path, root)`

Checks containment in two ways:

- Lexically, using absolute paths
- Physically, after resolving symlinks

Both are needed to detect entries located outside the store and symlinks that point outside it.

### `_entries_outside(backend)`

Walks the parent tree and reports filesystem entries that are not contained inside the backend’s storage location.

It inspects files, directories, and symlinks. Ancestor directories are excluded because they are necessary to reach the store and are not escapes.

### `_roundtrip(backend, key)`

Classifies backend behavior as:

- `"ok"` — key and value round-trip.
- `"escapes"` — storage was written outside its directory.
- `"corrupt"` — the original key disappears from `keys()`.
- `"bad-value"` — key exists but returns the wrong value.
- Exception class name — serialization raised.

This gives all tests one definition of correct key handling.

### `probe_backend`

Creates the backend one level below `storage_dir`.

That gives a `../escaped` key somewhere observable to escape into. Teardown removes the nested directory, while the global leak detector catches anything escaping higher.

## Non-string key test

### `test_non_string_keys_are_rejected_or_roundtrip`

Strict `xfail`.

Writes integer key `1`, forces it onto disk, and accepts either of two valid contracts:

1. Reject it with an actionable `TypeError` saying keys must be strings.
2. Preserve it exactly as integer `1`.

What is forbidden is silently converting `1` to `"1"`.

Current behavior varies:

- SQLite and JSON coerce it to a string.
- Pickle and HDF5 leak low-level, poorly explained `TypeError`s.

The key is forced onto disk because the in-memory cache preserves integers and could hide backend corruption.

## Path-shaped key tests

### `test_keys_containing_path_separators`

Requires `"a/b"` to round-trip as one exact key.

Path-based backends may interpret it as directories. HDF5 creates an internal group `a` containing dataset `b`, then reports the wrong key.

Affected backend cases receive strict runtime `xfail` markers.

### `test_key_traversal_is_neutralised`

Checks that `"../escaped"` cannot write outside the storage directory.

Rejecting the key is acceptable for this specific security test. Even if serialization raises, the surrounding tree is scanned because a partial write may already have escaped.

### `test_keys_cannot_write_outside_the_storage_tree`

Tests more severe escapes:

- An absolute path
- Multiple `..` components that climb beyond the tree inspected by `_entries_outside()`

It creates a sentinel directory outside storage and verifies no file appears there.

The sentinel is explicitly deleted because the normal storage leak detector intentionally cannot see it.

### `test_traversal_key_round_trips`

This is stricter than the containment test.

Containment alone can pass when:

- The backend rejects the key.
- HDF5 keeps the write inside its file but mangles the key.

This test requires `"../escaped"` to remain exactly that key and return its value correctly.

## Other degenerate keys

### `test_empty_and_overlong_keys`

Despite the name, it tests five labels:

- Empty
- Overlong
- Dot
- Dot-dot
- NUL

Every one is a valid Python string and therefore a valid `dict` key. The desired design is to encode or sanitize the storage name without changing the logical key.

Current failures include raw filesystem errors and silent key corruption.

## Capability-table self-checks

### `test_key_table_matches_reality`

Re-runs every key probe against every backend and verifies that `BROKEN_KEYS` still matches actual behavior.

This prevents a fixed backend from remaining marked broken or a new regression from going unnoticed.

### `test_traversal_escape_table_matches_reality`

Similarly verifies that `TRAVERSAL_ESCAPES` correctly identifies the backends that physically escape their storage directory.

## Filesystem folding tests

Some filesystems treat distinct names as equivalent:

- Case folding: `"Alpha"` and `"alpha"`
- Unicode normalization folding: composed and decomposed spellings of `é`

Python dictionaries treat these as distinct keys.

### `FOLDING_KEY_PAIRS`

Defines those two pairs.

### `PATH_PER_KEY_BACKENDS`

Contains Pickle and JSON because each key becomes a separate filename. SQLite and HDF5 store keys inside one file and are not affected by host filename folding.

### `_filesystem_folds()`

Probes the actual temporary filesystem rather than assuming behavior from the operating system.

It writes both names and checks whether two directory entries survive.

### `test_keys_differing_only_by_case_or_normalization_stay_distinct`

Writes both distinct keys and verifies:

- Both appear in `keys()`.
- Each retains its own value.
- The second write did not overwrite the first.

On a folding filesystem, affected path-per-key backends receive strict `xfail` markers.

Overall, this file protects two separate requirements:

1. Logical correctness: every legal key remains exactly the same key.
2. Filesystem safety: no key can redirect storage outside the backend’s assigned location.

# Test Persistance

`test_persistence.py` defines the core persistence promise: data written to a known store must remain recoverable after closing, reopening, or restarting the process.

Every test runs against all four backends because persistence is primarily a backend responsibility.

The intended lifecycle distinguishes:

- `create(path)` — create a new store.
- `open(path)` — open an existing store without truncating it.
- `open_or_create(path)` — choose the correct operation.
- `close()` — flush and release handles, but preserve storage.
- `destroy()` — permanently remove storage.
- `temporary()` — create disposable scratch storage.

Most tests are strict `xfail` because this lifecycle is still stubbed or current `close()` semantics are destructive.

## Shared setup

### `KEYS`

A small five-key dataset used by reopen and process-restart tests.

### `_build(backend, ...)`

Wraps an already-created backend in an `EffiDict` with a configured replacement policy.

This allows tests to control whether the backend was created, opened, or temporary before building the facade.

## Addressability tests

Before reopening can work, the same path argument must identify the same physical store.

### `test_storage_path_is_stable_across_instances`

Strict `xfail`.

Creates a backend from a path, records `storage_path`, releases it, then constructs another backend using the same path.

Both instances must resolve to the same storage location.

The handles are not open simultaneously because this test is about stable addressing, not concurrent access.

Current paths include `time.time()` and `id(self)`, so the same input produces different locations.

### `test_id_reuse_cannot_collide`

Strict `xfail`.

Checks that `storage_path` does not contain `id(backend)`.

Python may recycle object IDs after garbage collection. If an ID is part of a storage path, a later backend can accidentally reuse an old store’s location.

Depending on the backend, that can either:

- Expose stale data.
- Truncate existing data.

The test checks the faulty mechanism directly instead of relying on nondeterministically triggering ID reuse.

## Reopen tests

### `test_reopen_after_close_returns_all_values`

Strict `xfail`.

This is the main persistence test:

1. Create a backend at a known path.
2. Write values through EffiDict.
3. Flush.
4. Close.
5. Open the same path.
6. Read every value back.

It defines the difference between a persistent store and a temporary cache.

### `test_open_does_not_truncate_existing_store`

Strict `xfail`.

Writes a value, closes the first backend, then opens the same path.

Opening must preserve both the key and value.

This specifically catches HDF5’s current use of mode `"w"`, which truncates the file during construction.

The first handle is closed before the second is opened so the test does not accidentally require concurrent handles.

### `test_unflushed_writes_are_not_lost_on_close`

Strict `xfail`.

Uses a cache larger than the dataset so no value is naturally evicted to disk.

It writes values and then calls `close()` without an explicit `flush()`.

After reopening, every value must remain available.

This establishes that `close()` must flush dirty cached entries before releasing resources.

## Process-boundary durability

### `test_data_survives_process_restart`

Strict `xfail`.

Runs a real subprocess that:

1. Creates a store at a known path.
2. Writes values.
3. Flushes and closes it.
4. Exits.

The parent process then opens that path and reads the values.

A subprocess matters because creating another object in the same interpreter might accidentally reuse in-memory state and give a false impression of persistence.

## Temporary-store tests

### `test_temporary_stores_are_unique`

Strict `xfail`.

Calls `backend_cls.temporary()` twice and verifies the two stores have different storage paths.

Once ordinary paths become stable, this is the explicit API for requesting unique scratch stores.

### `test_temporary_stores_clean_up_after_themselves`

Strict `xfail`.

Creates a temporary backend, confirms its storage exists, releases it, then confirms the storage is gone.

This is tested explicitly because temporary stores may live outside the suite’s normal `storage_dir`, where the automatic leak detector cannot observe them.

## Backend close semantics

### `test_backend_close_is_idempotent_and_keeps_the_storage`

Strict `xfail`.

It checks three properties:

1. `close()` releases resources.
2. Calling `close()` twice does not raise.
3. `close()` does not remove stored data.

After two closes, the storage must still exist and reopen with its value intact.

Idempotence is important because cleanup may be reached through several paths, such as explicit close, context-manager exit, and object finalization.

## Open-or-create behavior

### `test_open_or_create_creates_then_reopens_without_truncating`

Strict `xfail`.

It exercises both branches:

1. On a missing path, create a working store.
2. Write and close it.
3. On the existing path, reopen rather than recreate.
4. Verify old data remains.
5. Verify the reopened handle can accept new writes.

This prevents an implementation from always choosing “create,” which could silently truncate an existing store.

Overall, this file establishes a crucial separation:

- `close()` preserves data.
- `destroy()` removes data.
- Stable paths make data addressable.
- `open()` recovers it.
- `temporary()` provides explicitly disposable storage.

# Test Lifecycle

`test_lifecycle.py` defines who owns storage and who is allowed to delete it.

Its central rule is:

- `close()` releases resources and preserves data.
- `destroy()` explicitly deletes data.
- Temporary stores may opt into deletion on close.
- Copies and deserialized worker instances must not accidentally own the original storage.

Most tests are strict `xfail` because the current implementation conflates closing with destruction.

## Test helpers

### `_build_unregistered()`

Constructs an EffiDict without using `make_dict`.

`make_dict` keeps references to every object for teardown, which would prevent garbage collection. Tests involving `del` and `gc.collect()` therefore need unregistered objects.

It returns:

- The EffiDict
- Its storage path

### `_remove(path)`

Best-effort cleanup for files or directories explicitly owned by a test.

This is used after tests that intentionally require storage to survive garbage collection or context-manager exit.

## Copying tests

### `test_copy_raises_with_actionable_message`

Strict `xfail`.

Requires `copy.copy(d)` to raise `TypeError`, with a message directing the caller to `clone()`.

A shallow copy would share the same backend and storage ownership, creating a risk that collecting one object deletes the other’s data.

### `test_deepcopy_raises_with_actionable_message`

Strict `xfail`.

Requires `copy.deepcopy(d)` to be refused for the same reason.

SQLite and HDF5 may already fail during deep copying because their handles cannot be pickled, but that is not sufficient—the error must clearly tell callers to use `clone()`.

### `test_copying_does_not_destroy_the_originals_storage`

Strict `xfail`.

This directly reproduces the real data-loss bug.

It accepts either safe design:

- Copying is refused.
- Copying succeeds without sharing destructive ownership.

If a copy is created, the test deletes it and forces garbage collection. The original storage must still exist and remain readable.

The probe value is written directly to disk so an in-memory cache cannot hide deleted storage.

### `test_clone_produces_independent_store`

Strict `xfail`.

Requires `clone(new_path)` to produce a truly independent store:

- Different storage path
- Same initial contents
- Changes to clone do not affect original
- Changes to original do not affect clone
- Destroying clone does not destroy original

Both stores are flushed before persistent-tier assertions. Otherwise, separate caches could conceal a shared backend.

## Garbage-collection tests

### `test_del_never_destroys_storage`

Strict `xfail`.

Creates an unregistered store, deletes the final reference, and calls `gc.collect()`.

The storage must remain on disk.

Object finalization may close handles, but ordinary garbage collection must never mean “delete user data.”

### `test_unpickled_instance_does_not_own_storage`

Strict `xfail`.

Serializes and deserializes an EffiDict with `pickle`, simulating an object sent to another process.

After the deserialized child is collected:

- Shared storage must still exist.
- The original store must remain readable.

This covers the joblib failure where a worker’s temporary copy deletes the parent’s storage when the worker finishes.

SQLite and HDF5 currently also fail because their backend handles are not picklable.

## Explicit lifecycle tests

### `test_destroy_is_explicit_and_idempotent`

Strict `xfail`.

Requires EffiDict to expose `destroy()`.

It verifies:

1. First `destroy()` removes storage.
2. Second `destroy()` does nothing and does not raise.

Idempotence allows safe cleanup in `finally` blocks even when another path may already have destroyed the store.

### `test_context_manager_closes_but_does_not_destroy`

Strict `xfail`.

After:

```python
with d:
    d["k"] = "v"
```

leaving the block must release resources but preserve storage.

Current `__exit__` calls destructive `close()` behavior.

### `test_temporary_store_is_destroyed_on_close`

Strict `xfail`.

Temporary stores are the exception to normal close semantics.

For a backend created with `temporary()`:

- Writing is allowed.
- Calling `close()` should remove its disposable storage.

The test cleans up explicitly because temporary storage lies outside the suite’s normal leak-detection directory.

### `test_dirty_entries_are_flushed_on_close`

Strict `xfail`.

Creates a cache large enough that nothing has spilled to disk, then spies on backend writes during `close()`.

At least one `serialize()` or `write_many()` call must occur, proving that dirty cached entries are persisted before handles are released.

This test focuses specifically on close-time flushing without depending on the separate reopen API.

Overall, this file separates three concepts currently entangled in the implementation:

- Object lifetime
- Resource lifetime
- Data lifetime

Destroying a Python object may end the first two, but it must not end the third unless the caller explicitly requests `destroy()` or created a temporary store.

# Test Complexity

`test_complexity.py` contains regression guards for how runtime work and disk usage grow as the store gets larger.

The entire module is marked `slow`.

Its main testing principles are:

- Count operations instead of timing them where possible.
- Compare small and large cases within the same run.
- Prefer ratios over machine-dependent absolute thresholds.

## Membership instrumentation

### `_count_enumerated(backend)`

Temporarily wraps `backend.keys()` and counts how many keys callers enumerate.

Counting calls alone would be insufficient because one `keys()` call can return either 100 or 4,000 keys.

If `keys()` returns a lazy iterator, the wrapper counts items as they are consumed rather than converting it into a list.

The returned `restore()` function removes the temporary instance-level wrapper.

## Membership complexity

### `test_membership_cost_does_not_grow_with_the_store`

Strict `xfail`.

Builds stores containing:

- 100 keys
- 4,000 keys

It then checks membership for one absent key and counts how many backend keys were enumerated.

The large-store check must enumerate zero keys. Membership should use a direct point lookup such as `backend.has(key)`, not scan `backend.keys()`.

Current `__contains__` performs an O(N) scan.

### `test_membership_time_is_flat`

Non-strict `xfail`.

Measures 50 absent-key membership checks at both store sizes and computes:

```text
large average / small average
```

The ratio must remain below 4×.

This is non-strict because timing can vary on busy CI machines. The deterministic enumeration-count test is the real acceptance guard.

## Read amplification

### `test_read_only_pass_performs_no_writes`

Strict `xfail`.

Seeds 500 values directly on disk, then reads them through a 50-entry cache while spying on backend writes.

Because all values are already persisted and unmodified, the read pass should issue zero:

- `serialize()` calls
- `write_many()` calls

Current promoting policies do not mark loaded entries as clean, so evicting them writes unchanged data back to disk—currently about one write per read.

This is the large-scale version of the clean-eviction tier invariant.

## Bulk-write efficiency

### `test_bulk_write_does_not_issue_one_call_per_item`

Strict `xfail`.

Loads 3,000 entries through `load_from_dict()` and counts both single and batch writes.

The total number of backend write calls must be at most:

```python
3000 // 100  # 30 calls
```

This requires batching at least roughly 100 items per backend call.

Both write paths are counted so an implementation cannot pass by replacing 3,000 `serialize()` calls with 3,000 `write_many()` calls.

The current implementation performs one backend operation per item.

## Absolute storage overhead

### `test_stored_payload_is_not_far_larger_than_the_data`

Strict `xfail`, specifically for SQLite.

It:

1. Creates a list of 10,000 integers.
2. Measures its Protocol 5 pickle size as a stable binary reference.
3. Stores it in SQLite.
4. Measures the resulting backend storage.

The SQLite representation must be no more than 1.5× the binary reference.

Current JSON-text storage is roughly 2.3× larger.

The pickle protocol is fixed so Python upgrades do not silently change the comparison baseline.

## Storage growth rate

### `test_stored_size_grows_linearly_with_the_data`

Runs for every backend.

It compares storage occupied by:

- 250 entries
- 1,000 entries

Because the item count increases by 4×, storage must grow by less than:

```python
4 × 1.6 = 6.4×
```

The extra factor allows fixed headers and ordinary metadata overhead while rejecting layouts whose cost per item worsens significantly as the store fills.

HDF5 receives a strict runtime `xfail` because its group metadata grows superlinearly with the number of datasets.

This test differs from the SQLite overhead test:

- SQLite test: “How expensive is this representation absolutely?”
- Growth test: “Does cost per item deteriorate as the store grows?”

Overall, this file guards against hidden O(N) lookups, write amplification, ineffective batching, excessive encoding overhead, and superlinear disk growth.

# Test Memory

`test_memory.py` guards the process’s memory usage and ensures large-store iteration does not materialize or cache everything.

The entire module is marked `slow`.

A key detail is that every large test value is a distinct object. Reusing the same 1 MiB string under 100 keys would only store 100 references to one buffer and would not genuinely test memory growth.

## Shared helpers

### `_distinct(index, size)`

Creates a string of the requested size with a unique suffix.

This guarantees each cached value occupies its own memory rather than sharing one object.

### `needs_rss`

Skips the RSS test on Windows because Python’s Unix `resource` module is unavailable there.

## Cache byte-budget test

### `test_cache_respects_max_bytes`

Strict `xfail`.

Creates an EffiDict configured with:

- Maximum 100 cached items
- Maximum 8 MiB cached bytes

It then writes 100 distinct 1 MiB values.

The item limit allows all 100, but the byte limit should restrict memory residency.

The assertion permits:

```text
budget + one newly inserted entry
```

The one-entry allowance reflects the planned sequence:

1. Insert an entry.
2. Notice the cache is over budget.
3. Identify and process eviction candidates.

Current `max_bytes` is stored but never enforced, so approximately 100 MiB remains cached.

## Process RSS test

### `_RSS_PROBE`

This is a Python script executed in a fresh subprocess.

It:

1. Creates an 8 MiB-budget EffiDict.
2. Records baseline resident memory.
3. Writes 100 distinct 1 MiB values.
4. Samples current RSS after every write.
5. Prints peak growth above baseline.

A fresh process is important because process-wide peak memory in the main test runner may already be inflated by previous tests.

On Linux it reads `/proc/self/statm` for current RSS. Elsewhere it falls back to `resource.getrusage()`.

Sampling after every write catches temporary growth that might disappear before a final measurement.

### `test_peak_rss_is_bounded_by_max_bytes`

Strict `xfail` and skipped without RSS support.

Runs `_RSS_PROBE` and requires memory growth to remain below:

```text
8 MiB budget × 4 = 32 MiB
```

The generous multiplier allows interpreter, allocator, and backend overhead while still clearly rejecting approximately 100 MiB growth.

This complements the structural cache-byte test by measuring the real process-level consequence.

## Streaming key iteration

### `test_keys_does_not_materialise_the_whole_keyspace`

Strict `xfail`.

Writes 500 keys and verifies that `keys()` does not return a fully materialized:

- List
- Tuple
- Set
- Frozen set
- Dictionary

It also requires `iter(d)` to return an independent iterator rather than the EffiDict object itself.

The test is structural because creating millions of keys just to observe actual memory growth would be too expensive.

## Full-scan cache behavior

### `test_full_scan_does_not_evict_the_working_set`

Strict `xfail`.

Creates 200 keys with a 10-entry cache, warms a small working set, then scans every item.

After the scan, it requires:

- Every previously cached working-set key remains cached.
- The scan performs zero backend writes.

Current `items()` reads each value through the normal promoting policy path. That causes the full dataset to cycle through the cache, evicting the caller’s useful working set and rewriting already-persisted values.

The intended streaming scan should bypass cache promotion.

Overall, this file tests four related guarantees:

- Byte limits constrain cached data.
- Real process memory stays bounded.
- Key iteration is lazy.
- Full scans do not destroy cache locality or cause writes.

# Test Concurrency

`test_concurrency.py` covers three progressively broader boundaries:

1. Multiple threads in one process
2. Read-only worker processes receiving a pickled store
3. Multiple independent writer processes sharing storage

The entire module is marked `stress`, so CI repeats it because some failures depend on scheduling.

The suite distinguishes:

- Deterministic defects → `xfail(strict=True)`
- Genuine races → `xfail(strict=False)`

## Shared configuration

### `PICKLABLE_BACKENDS`

Currently contains Pickle and JSON.

SQLite holds a live `sqlite3.Connection`, and HDF5 holds an open file handle, neither of which currently survives pickling.

### `THREAD_AFFINE_BACKENDS`

Currently contains SQLite because its connection can only be used by the thread that created it.

### `_worker_read()` and `_worker_write()`

Module-level functions used by joblib workers.

They return formatted error strings rather than raising, allowing the parent process to report all worker failures together.

They are module-level because joblib must pickle them by reference.

## Thread tests

### `test_backend_is_usable_from_another_thread`

Creates a backend on the main thread, writes a value, then reads it from another thread.

It verifies:

- The reader finishes.
- No exception occurs.
- The correct value is returned.

SQLite cases receive a strict runtime `xfail` because its connection is thread-affine.

The reader thread is a daemon so a backend deadlock does not prevent the test interpreter from exiting.

### `test_concurrent_writers_see_every_key`

Non-strict `xfail`.

Starts eight threads. Each writes and immediately reads 200 unique keys through one EffiDict with a two-entry cache.

The small cache deliberately creates heavy eviction pressure.

It verifies:

- No writer remains stuck.
- No worker reported an exception.
- After all writing stops, all 1,600 expected keys still have the correct values.

The final full read is important because an immediate read inside each thread only proves the key survived briefly. Another thread might lose or overwrite it later.

It is non-strict because thread scheduling can occasionally avoid the race.

### `test_reader_never_observes_a_key_mid_eviction`

Strict `xfail`.

This is the deterministic concurrency probe.

It:

1. Fills a two-entry cache.
2. Replaces backend `serialize()` with a blocking wrapper.
3. Starts a writer that triggers eviction.
4. Pauses the write while it is in progress.
5. Starts a reader for the evicted key.
6. Observes whether the reader reports the key as missing.
7. Releases the writer and checks both operations complete correctly.

A valid implementation may:

- Return the value while eviction is underway.
- Block the reader behind a Store lock, then return the value.

It must never return `KeyError` or another failure caused by the key being temporarily absent from both tiers.

Events coordinate the probe; sleeps are not used to decide correctness.

The test also verifies that the writer itself eventually succeeds. Otherwise, a permanently cached value could make the reader pass while persistence remained broken.

## Pickling tests

### `test_all_backends_are_picklable`

Pickles and unpickles each backend, then verifies the revived backend can read existing data.

SQLite and HDF5 receive strict runtime `xfail` markers.

Both original and revived handles are released because they may reference the same storage.

### `test_picklable_backend_table_matches_reality`

Re-tests whether each backend can be pickled and compares the result with `PICKLABLE_BACKENDS`.

This prevents the literal capability table from becoming outdated.

## Joblib read-only worker tests

These run only for currently picklable backends.

### `test_joblib_workers_can_read`

Non-strict `xfail`.

Creates a store, sends it to three joblib workers, and asks workers to read keys.

It verifies:

- No worker reports an error.
- All expected values are returned.

The failure is scheduling-dependent because one worker’s deserialized store may be garbage-collected and delete shared storage before another worker reads.

### `test_joblib_workers_do_not_destroy_storage`

Strict `xfail`.

Runs worker reads and then checks:

- The parent’s storage path still exists.
- The parent can still read every value.

This deterministically captures the storage-ownership bug: worker copies currently consider themselves owners and delete the parent’s storage during finalization.

### `test_worker_write_raises_clearly`

Strict `xfail`.

The current concurrency model permits one writer process and read-only workers.

The test sends write attempts to workers and requires:

- Every write is refused.
- Error messages explain that workers are read-only or cannot write.
- No refused key appears in the parent store.

The final check catches implementations that persist a value and then raise an error.

## Multiprocess writer helpers

### `_WRITER`

A standalone script that:

1. Opens or creates a known store.
2. Writes supplied items.
3. Flushes.
4. Closes.

### `_run_writer()`

Runs one subprocess to completion. It is used where operation ordering is intentional.

### `_run_writers_concurrently()`

Uses `subprocess.Popen()` to start all writers before waiting for any of them.

Using repeated `subprocess.run()` would execute writers sequentially and would not test concurrent writes.

Timeouts kill stuck processes and preserve their error output.

## Phase 5b multiprocess tests

These are strict `xfail` specifications for future multiwriter support.

### `test_multiprocess_disjoint_writes_all_visible`

Starts three processes concurrently, each writing a different key range.

Afterward, a reader opens the shared store and must see every key from every process.

This detects lost transactions or one process overwriting another’s storage state.

### `test_multiprocess_same_key_writes_are_not_torn`

Starts three concurrent processes writing different values to the same key.

Last-writer-wins is acceptable. The final value must be exactly one value that a writer supplied.

Missing, partially encoded, or otherwise corrupted values fail.

### `test_stale_cache_is_invalidated_after_external_write`

Tests cross-process cache coherence:

1. Process A writes `"first"`.
2. The parent opens the store and caches `"first"`.
3. Process B overwrites it with `"second"`.
4. The existing reader must return `"second"`.

This is deliberately sequential because the key requirement is that another process changes a value after it has already been cached.

Locks alone cannot solve this; the implementation needs a coherence or invalidation mechanism.

Overall, the file defines the intended concurrency progression:

- Backends should work across threads.
- Worker copies may read but must not own or modify parent storage.
- Future multiprocess writers need coordination, atomicity, and cache invalidation.

