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


def _roundtrip(backend, key, value="v"):
    """Classify what a backend does with ``key``: ok, corrupt, escapes, or raises.

    Shared by the specs and by the self-check below, so there is one definition of
    "handled correctly" rather than one per test.
    """
    root = os.path.dirname(os.path.dirname(backend.storage_path))
    try:
        backend.serialize(key, value)
    except Exception as exc:  # noqa: BLE001 - the exception type is the finding
        return type(exc).__name__

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not os.path.join(dirpath, filename).startswith(backend.storage_path):
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
    """
    d = make_dict(max_in_memory=4)

    with pytest.raises(TypeError) as excinfo:
        d[1] = "one"

    message = str(excinfo.value).lower()
    assert "key" in message and "str" in message, (
        f"refusal should say keys must be strings; got: {excinfo.value}"
    )


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

    root = os.path.dirname(os.path.dirname(probe_backend.storage_path))
    probe_backend.serialize(KEY_PROBES["traversal"], "v")

    outside = [
        os.path.relpath(os.path.join(dirpath, filename), root)
        for dirpath, _, filenames in os.walk(root)
        for filename in filenames
        if not os.path.join(dirpath, filename).startswith(probe_backend.storage_path)
    ]
    assert not outside, f"a '..' key wrote outside the store: {outside}"


@pytest.mark.parametrize("label", ["empty", "overlong", "dot", "dotdot", "nul"])
def test_empty_and_overlong_keys(request, backend_cls, probe_backend, label):
    """Degenerate but legal string keys must round-trip or be refused clearly.

    Today they produce raw filesystem errors -- ``IsADirectoryError``,
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
