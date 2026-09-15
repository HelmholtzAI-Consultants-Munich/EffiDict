"""Specs for what may be used as a key.

Three of the four backends derive a storage *location* from the key -- a filename
for Pickle and JSON, a dataset path for HDF5 -- so a key is not merely data.
SQLite stores it in a column and handles every probe below correctly, which is why
the expectations here are per-backend rather than uniform.

The ``BROKEN_KEYS`` table was derived by round-tripping each key through each
backend, not by reading the source, and ``test_key_table_matches_reality``
re-derives it on every run so it cannot go stale.
"""

from __future__ import annotations

import os
import shutil

import pytest

from effidict import Hdf5Backend, JSONBackend, PickleBackend

from .helpers import on_disk

#: Keys that exercise the boundary between "key as data" and "key as location".
KEY_PROBES = {
    "separator": "a/b",
    "traversal": "../escaped",
    "empty": "",
    "overlong": "x" * 300,
    "dot": ".",
    "dotdot": "..",
    "nul": "a\x00b",
}

#: Which backends fail to round-trip each probe today. Literal on purpose: a
#: computed expectation would assert whatever currently happens.
BROKEN_KEYS = {
    "separator": {PickleBackend, JSONBackend, Hdf5Backend},
    "traversal": {PickleBackend, JSONBackend, Hdf5Backend},
    "empty": {PickleBackend, JSONBackend, Hdf5Backend},
    "overlong": {PickleBackend, JSONBackend},
    "dot": {PickleBackend, JSONBackend, Hdf5Backend},
    "dotdot": {PickleBackend, JSONBackend},
    "nul": {PickleBackend, JSONBackend, Hdf5Backend},
}

#: Backends that write a key's value *outside* the storage directory when the key
#: contains ``..``. A containment failure, distinct from mere corruption: HDF5
#: mangles such a key but keeps everything inside its single file.
TRAVERSAL_ESCAPES = {PickleBackend, JSONBackend}


def _is_inside(path, root):
    """Whether ``path`` lies within ``root``, comparing resolved paths.

    ``startswith`` is not containment: ``<root>-escaped`` shares the prefix but is
    a sibling, and an unresolved symlink can point anywhere at all. Since this is
    the traversal guard, it compares real paths via ``commonpath``.
    """
    root = os.path.realpath(root)
    try:
        return os.path.commonpath([os.path.realpath(path), root]) == root
    except ValueError:  # different drives on Windows
        return False


def _files_outside(backend):
    """Every file under the store's parent tree that is not inside the store."""
    root = os.path.dirname(os.path.dirname(backend.storage_path))
    return [
        os.path.relpath(os.path.join(dirpath, filename), root)
        for dirpath, _, filenames in os.walk(root)
        for filename in filenames
        if not _is_inside(os.path.join(dirpath, filename), backend.storage_path)
    ]


def _roundtrip(backend, key, value="v"):
    """Classify what a backend does with ``key``: ok, corrupt, escapes, or raises.

    Shared by the specs and by the self-check below, so there is one definition of
    "handled correctly" rather than one per test.
    """
    try:
        backend.serialize(key, value)
    except Exception as exc:  # noqa: BLE001 - the exception type is the finding
        return type(exc).__name__

    if _files_outside(backend):
        return "escapes"
    if key not in backend.keys():
        return "corrupt"
    if backend.deserialize(key) != value:
        return "bad-value"
    return "ok"


@pytest.fixture
def probe_backend(backend_cls, storage_dir):
    """A backend nested one level down, so a ``..`` escape has somewhere to land."""
    nested = storage_dir / "nested"
    nested.mkdir(exist_ok=True)
    backend = backend_cls(str(nested / "store"))
    yield backend
    try:
        backend.destroy()
    except OSError:
        pass
    # Remove the whole nested directory, not just the store: destroy() leaves the
    # directory itself, and the traversal probe deliberately writes a file beside
    # the store. Anything that escaped *above* nested still lands in storage_dir,
    # where conftest's leak detector will catch it.
    shutil.rmtree(nested, ignore_errors=True)



def _with_int_key_on_disk(make_dict, key=1):
    """A store where ``key`` has reached the disk tier, whatever the policy.

    Two write orders are needed, the same split as in the tier-invariant specs:
    most policies evict an *older* entry, so writing the probe first and then
    filler pushes it out; LIFO and MRU evict the entry just inserted, so the
    cache has to be full *before* the probe is written. Returns ``None`` only if
    neither order spills it.

    ``TypeError`` is deliberately allowed to propagate -- for Pickle and HDF5 that
    is the rejection path the caller wants to inspect.
    """
    for fill_first in (False, True):
        store = make_dict(max_in_memory=1)
        if fill_first:
            store["filler"] = "v"
        store[key] = "one"
        if not fill_first:
            for index in range(8):
                if on_disk(store, key) or on_disk(store, str(key)):
                    break
                store[f"filler{index}"] = "v"
        if on_disk(store, key) or on_disk(store, str(key)):
            return store
    return None


def _xfail_if_broken(request, label, backend_cls):
    if backend_cls in BROKEN_KEYS[label]:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{backend_cls.__name__} derives a storage location from the "
                    f"key, so {label} keys do not round-trip (issue 7.1)"
                ),
            )
        )


# --------------------------------------------------------------------------
# key type
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "non-string keys are neither rejected nor preserved: SQLite/JSON coerce "
        "them to str, Pickle/HDF5 leak a raw TypeError from os.path/h5py "
        "(issue 7.1)"
    ),
)
def test_non_string_keys_are_rejected_or_roundtrip(backend_cls, policy_cls, make_dict):
    """A non-string key must be refused with an actionable error.

    Silently coercing is the damaging outcome. With ``d[1] = "one"`` SQLite and
    JSON store ``"1"``, and once the entry exists in both tiers ``keys()`` returns
    ``['1', 1]`` -- one logical entry as two keys, with both ``1 in d`` and
    ``'1' in d`` true.

    Pickle and HDF5 do raise ``TypeError``, but from ``os.path.join`` and h5py
    respectively, so the message says nothing about keys needing to be strings.

    Both outcomes in the name are accepted, because either is a defensible
    contract; what is not acceptable is accepting the key and changing it. The
    integer has to be pushed out to disk first: while it sits in the cache the
    in-memory dict preserves it perfectly, so a test that never evicts would
    report success on a store that corrupts the key the moment it spills.
    """
    try:
        d = _with_int_key_on_disk(make_dict)
    except TypeError as exc:
        message = str(exc).lower()
        assert "key" in message and "str" in message, (
            f"refusal should say keys must be strings; got: {exc}"
        )
        return

    if d is None:  # pragma: no cover - no policy should reach this
        pytest.skip(f"{policy_cls.__name__} never spilled the probe key")

    # Accepted, so it must have survived unchanged.
    assert d[1] == "one"
    assert 1 in d
    assert "1" not in d, "the integer key was coerced to a string"
    assert "1" not in d.keys(), f"keys() reports a coerced duplicate: {d.keys()}"


# --------------------------------------------------------------------------
# keys that look like paths
# --------------------------------------------------------------------------


def test_keys_containing_path_separators(request, backend_cls, probe_backend):
    """``'a/b'`` is a legal dict key and must round-trip as one key.

    HDF5 is the quiet one: ``/`` creates a nested group, so the write succeeds and
    ``keys()`` reports ``'a'``. The value is still there, addressed by a key the
    caller never used.
    """
    _xfail_if_broken(request, "separator", backend_cls)

    assert _roundtrip(probe_backend, KEY_PROBES["separator"]) == "ok"


def test_key_traversal_is_neutralised(request, backend_cls, probe_backend):
    """A key containing ``..`` must not write outside the storage directory.

    Containment, not merely round-tripping: ``d["../escaped"] = v`` writes
    ``escaped`` a directory above the store on Pickle and JSON. Asserted against
    the whole parent tree rather than the store itself, because a check scoped to
    the store cannot see something that left it.
    """
    if backend_cls in TRAVERSAL_ESCAPES:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{backend_cls.__name__} joins the key onto its storage path "
                    f"unsanitised, so '..' writes outside the store (issue 7.1)"
                ),
            )
        )

    # Refusing the key outright is a legitimate way to satisfy containment, so a
    # raise is not a failure here -- but the tree is still scanned afterwards, in
    # case the write got partway before failing.
    try:
        probe_backend.serialize(KEY_PROBES["traversal"], "v")
    except Exception:  # noqa: BLE001 - refusal is an acceptable outcome
        pass

    outside = _files_outside(probe_backend)
    assert not outside, f"a '..' key wrote outside the store: {outside}"


@pytest.mark.parametrize("label", ["empty", "overlong", "dot", "dotdot", "nul"])
def test_empty_and_overlong_keys(request, backend_cls, probe_backend, label):
    """Degenerate but legal string keys must round-trip.

    Round-tripping, not "or refused": every one of these is a valid ``dict`` key,
    and issue 7.1 sanitises the *storage name* rather than restricting the
    keyspace, so refusing them would be a deviation from ``dict`` rather than a
    fix. Today they produce raw filesystem errors -- ``IsADirectoryError``,
    ``OSError: File name too long``, ``ValueError: embedded null byte`` -- or
    silent corruption, where JSON turns ``''`` into a file called ``.json`` and
    ``keys()`` reports ``'.json'``.
    """
    _xfail_if_broken(request, label, backend_cls)

    assert _roundtrip(probe_backend, KEY_PROBES[label]) == "ok"


# --------------------------------------------------------------------------
# self-check: the tables above must keep matching reality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(KEY_PROBES))
def test_key_table_matches_reality(backend_cls, probe_backend, label):
    """Re-derive ``BROKEN_KEYS`` every run so the literal table cannot drift."""
    verdict = _roundtrip(probe_backend, KEY_PROBES[label])
    expected_broken = backend_cls in BROKEN_KEYS[label]

    assert (verdict != "ok") == expected_broken, (
        f"{backend_cls.__name__} / {label}: table says "
        f"{'broken' if expected_broken else 'ok'}, reality says {verdict}"
    )


def test_traversal_escape_table_matches_reality(backend_cls, probe_backend):
    """Same for ``TRAVERSAL_ESCAPES``, which is a containment claim."""
    verdict = _roundtrip(probe_backend, KEY_PROBES["traversal"])

    assert (verdict == "escapes") == (backend_cls in TRAVERSAL_ESCAPES), (
        f"{backend_cls.__name__}: table says "
        f"{'escapes' if backend_cls in TRAVERSAL_ESCAPES else 'contained'}, "
        f"reality says {verdict}"
    )
