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

import subprocess
import sys
from pathlib import Path

import pytest

# Cache contents go through helpers, never through the policy's dict: the alias
# disappears at issue 1.2, and a strict xfail cannot distinguish "still broken"
# from "now raising AttributeError" -- these guards would stay red on a correct
# byte-budget implementation instead of flipping when 1.4 lands.
from .helpers import cached_bytes, cached_keys

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    # Imported conditionally so the three guards that need no RSS source still
    # run on Windows. An unconditional import fails at *collection*, taking the
    # whole file down rather than the one test that depends on it -- and the CI
    # matrix includes windows-latest. The parent only needs it to decide the
    # skip; the reading itself happens in the subprocess probe below.
    resource = None

pytestmark = pytest.mark.slow

REPO_ROOT = str(Path(__file__).resolve().parents[1])

MIB = 1024 * 1024


def _distinct(index, size=MIB):
    """A value of ``size`` bytes that is its own object, not a shared one."""
    tail = f"{index:016d}"
    return "x" * (size - len(tail)) + tail


#: Applied to the one guard that needs a peak-RSS reading. skipif wins over
#: xfail, so on Windows the test is skipped rather than reported as an expected
#: failure it never actually attempted.
needs_rss = pytest.mark.skipif(
    resource is None, reason="resource is Unix-only; no peak-RSS source on Windows"
)


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

    held = cached_bytes(d)
    assert held <= budget * 1.5, (
        f"the cache holds {held / MIB:.0f} MiB against a {budget / MIB:.0f} MiB "
        f"budget, in {len(cached_keys(d))} items"
    )


#: Run in a fresh interpreter, which is the whole point. ``ru_maxrss`` is the
#: process-wide *peak*, not current usage, so a baseline taken mid-suite is
#: already inflated by whatever ran before -- measured at 133 MiB after the
#: byte-budget guard above, which left this test seeing 1 MiB of growth and
#: XPASSing its strict marker. A new process starts from its own startup peak.
_RSS_PROBE = """
import os, resource, sys
sys.path.insert(0, sys.argv[1])
from effidict import EffiDict, PickleBackend, LRUReplacement

tmpdir, items, budget = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
MIB = 1024 * 1024

def peak_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024

def distinct(index, size=MIB):
    tail = str(index).zfill(16)
    return "x" * (size - len(tail)) + tail

backend = PickleBackend(os.path.join(tmpdir, "store"))
store = EffiDict(
    disk_backend=backend,
    replacement_strategy=LRUReplacement(disk_backend=backend, max_in_memory=items),
    max_bytes=budget,
)
baseline = peak_bytes()
for index in range(items):
    store[f"k{index}"] = distinct(index)
print(peak_bytes() - baseline)
"""


@needs_rss
@pytest.mark.xfail(
    strict=True,
    reason="max_bytes is accepted but never enforced, so the process grows with the data (issue 1.4)",
)
def test_peak_rss_is_bounded_by_max_bytes(tmp_path):
    """The process must not grow with the size of the data written.

    The observable consequence of the guard above: measured at around +105 MiB of
    resident memory for a store told to hold 8 MiB. Asserted as growth above a
    baseline, in a subprocess so that baseline means something.
    """
    budget = 8 * MIB
    items = 100

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _RSS_PROBE,
            REPO_ROOT,
            str(tmp_path),
            str(items),
            str(budget),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"the RSS probe failed:\n{result.stderr.strip()[-1500:]}"
    )
    growth = int(result.stdout.strip())

    # Generous headroom: the interpreter, the page allocator and the backend all
    # add overhead the budget does not account for. Holding the whole 100 MiB
    # working set clears it by more than an order of magnitude regardless.
    assert growth <= budget * 4, (
        f"writing {items} MiB through a {budget // MIB} MiB cache grew the "
        f"process by {growth // MIB} MiB"
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

    # Captured once, and checked against every concrete container: rejecting
    # only `list` would accept a set or tuple, which defeats streaming just as
    # thoroughly while making the guard pass.
    keys = d.keys()
    assert not isinstance(keys, (list, tuple, set, frozenset, dict)), (
        f"keys() returned a fully materialised {type(keys).__name__} rather than "
        f"a view or iterator"
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
    pulls all 200 keys through a 10-key cache. By the end the cache holds whatever
    the scan touched last rather than the caller's working set: at most one of the
    ten cached keys survives, and the walk issues 200 writes despite the caller
    only reading. A streaming scan has to bypass the cache instead.

    The survivor count is stated as a bound rather than a figure -- it lands on 0
    or 1 depending on where the scan ends -- because a docstring quoting an exact
    number goes stale the first time the cache size or key count changes.
    """
    d = make_dict(max_in_memory=10)
    for index in range(200):
        d[f"k{index:04d}"] = "v"

    working_set = [f"k{index:04d}" for index in range(195, 200)]
    for key in working_set:
        _ = d[key]
    before = cached_keys(d)
    assert before, "precondition: something is cached"

    with backend_spy(d.disk_backend) as spy:
        for _ in d.items():
            pass
        writes = spy.count("serialize") + spy.count("write_many")

    after = cached_keys(d)
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
