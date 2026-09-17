"""Regression guards for memory: what the process holds while using a store.

Marked ``slow`` as a file, so the default local loop skips them and CI's plain
``pytest`` run includes them.

Every value written here is a *distinct object*. That detail decides whether these
guards measure anything: writing one shared 1 MB string a hundred times leaves the
cache holding a hundred references to a single buffer, so peak RSS moves by about
1 MB while a naive ``sum(len(v))`` reports 105 MB. The first draft of the byte
budget guard did exactly that and would have passed on a store holding nothing.
"""

from __future__ import annotations

import resource
import sys

import pytest

pytestmark = pytest.mark.slow

MIB = 1024 * 1024


def _distinct(index, size=MIB):
    """A value of ``size`` bytes that is its own object, not a shared one."""
    tail = f"{index:016d}"
    return "x" * (size - len(tail)) + tail


def _peak_rss_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kibibytes.
    return usage if sys.platform == "darwin" else usage * 1024


def _cached_bytes(store):
    return sum(len(value) for value in store.replacement_strategy.memory.values())


# --------------------------------------------------------------------------
# the cache must honour a byte budget
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="max_bytes is accepted but never enforced (issue 1.4)",
)
def test_cache_respects_max_bytes(make_dict):
    """``max_bytes`` must bound what the cache holds.

    ``max_in_memory`` counts items, so a hundred one-megabyte values sit happily
    in a hundred-item cache: the item budget was never exceeded, so nothing was
    ever evicted. Scaled down from the 100 x 8 MB / 802 MB figure in the issue to
    keep the suite quick; the failure mode is identical.
    """
    budget = 8 * MIB
    d = make_dict(max_in_memory=100, max_bytes=budget)

    for index in range(100):
        d[f"k{index}"] = _distinct(index)

    held = _cached_bytes(d)
    assert held <= budget * 1.5, (
        f"the cache holds {held / MIB:.0f} MiB against a {budget / MIB:.0f} MiB "
        f"budget, in {len(d.replacement_strategy.memory)} items"
    )


@pytest.mark.xfail(
    strict=True,
    reason="max_bytes is accepted but never enforced, so the process grows with the data (issue 1.4)",
)
def test_peak_rss_is_bounded_by_max_bytes(make_dict):
    """The process must not grow with the size of the data written.

    The observable consequence of the guard above: measured at +106 MB of resident
    memory for a store told to hold 8 MB. Asserted as growth above a baseline
    taken in the same run, so it is not a claim about absolute footprint.
    """
    budget = 8 * MIB
    d = make_dict(max_in_memory=100, max_bytes=budget)

    baseline = _peak_rss_bytes()
    for index in range(100):
        d[f"k{index}"] = _distinct(index)
    growth = _peak_rss_bytes() - baseline

    # Generous headroom: the interpreter, the page allocator and the backend all
    # add overhead the budget does not account for. Even so, holding the whole
    # 100 MiB working set clears this by more than an order of magnitude.
    assert growth <= budget * 4, (
        f"writing 100 MiB through a {budget / MIB:.0f} MiB cache grew the process "
        f"by {growth / MIB:.0f} MiB"
    )


# --------------------------------------------------------------------------
# iteration must stream
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "keys() builds a list of every key, and __iter__ returns self rather than "
        "a lazy iterator (issue 4.2)"
    ),
)
def test_keys_does_not_materialise_the_whole_keyspace(make_dict):
    """Iterating keys must not require holding them all at once.

    ``keys()`` returns ``list(set(memory) | set(disk))``, so the caller pays for
    the entire keyspace before touching a single value -- roughly a gigabyte at
    ten million keys. Asserted structurally rather than by measuring memory: a
    streaming implementation cannot return a ``list``, and the size at which the
    cost would actually show up is far too large for a test.
    """
    d = make_dict(max_in_memory=10)
    for index in range(500):
        d[f"k{index:05d}"] = "v"

    assert not isinstance(d.keys(), list), (
        "keys() returned a fully materialised list rather than a view or iterator"
    )

    iterator = iter(d)
    assert iterator is not d, "__iter__ returned the store itself, not an iterator"
    assert not isinstance(iterator, list)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "items() reads every value through the policy, so a full scan promotes "
        "each key in turn and evicts whatever was cached (issue 4.2)"
    ),
)
def test_full_scan_does_not_evict_the_working_set(make_dict, backend_spy):
    """Walking the store must not cost the caller its cache.

    A scan reads every value through the normal path, so on a promoting policy it
    pulls 500 keys through a 10-key cache and the working set is gone by the end:
    measured at 1 of 10 hot keys surviving, plus 200 writes for a read-only walk.
    A streaming scan has to bypass the cache instead.
    """
    d = make_dict(max_in_memory=10)
    for index in range(200):
        d[f"k{index:04d}"] = "v"

    working_set = [f"k{index:04d}" for index in range(195, 200)]
    for key in working_set:
        _ = d[key]
    before = set(d.replacement_strategy.memory)
    assert before, "precondition: something is cached"

    with backend_spy(d.disk_backend) as spy:
        for _ in d.items():
            pass
        writes = spy.count("serialize") + spy.count("write_many")

    after = set(d.replacement_strategy.memory)
    survived = before & after

    # Two separate properties, so a failure says which one broke. The write count
    # is asserted rather than merely reported: a scan that preserved the cache but
    # still wrote to disk would otherwise pass this guard while doing the very
    # thing it is named for.
    assert survived == before, (
        f"a full scan evicted {len(before) - len(survived)} of {len(before)} "
        f"cached keys"
    )
    assert writes == 0, f"a read-only scan issued {writes} writes"
