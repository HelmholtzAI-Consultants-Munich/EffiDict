"""Shared harness for the Phase 0 specification suite.

Provides:

``backend_cls`` / ``policy_cls``
    Parametrized fixtures giving the 4 x 7 = 28 backend/policy matrix.
``make_dict``
    Factory for an ``EffiDict`` wired to a temp store, registered for teardown.
``backend_spy``
    Records backend method calls. Most non-functional guards count calls rather
    than measure time, so this lives here rather than in individual specs.
autouse leak detector
    Releases every store the test created, then asserts the storage directory is
    empty. Anything left behind is a library-level leak.
"""

from __future__ import annotations

import contextlib

import pytest

from effidict import (
    EffiDict,
    FIFOReplacement,
    Hdf5Backend,
    JSONBackend,
    LFUReplacement,
    LIFOReplacement,
    LRUReplacement,
    MFUReplacement,
    MRUReplacement,
    PickleBackend,
    RandomReplacement,
    SqliteBackend,
)

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is a dev dependency
    h5py = None

# `pytester` lets the harness test itself in a subprocess-like sandbox, which is
# how the leak detector proves it actually fires.
pytest_plugins = ["pytester"]

BACKENDS = [SqliteBackend, PickleBackend, JSONBackend, Hdf5Backend]

POLICIES = [
    RandomReplacement,
    FIFOReplacement,
    LIFOReplacement,
    LRUReplacement,
    MRUReplacement,
    LFUReplacement,
    MFUReplacement,
]

#: Used when a test asks for `make_dict` without joining the matrix. Pickle
#: accepts every value kind and LRU is the most common policy, so an
#: unparametrized test exercises the least surprising combination.
DEFAULT_BACKEND = PickleBackend
DEFAULT_POLICY = LRUReplacement


def _backend_param(backend_cls):
    marks = []
    if backend_cls is Hdf5Backend:
        marks.append(pytest.mark.skipif(h5py is None, reason="h5py is not installed"))
    return pytest.param(backend_cls, id=backend_cls.__name__, marks=marks)


@pytest.fixture(params=[_backend_param(cls) for cls in BACKENDS])
def backend_cls(request):
    """One disk backend class per parametrization."""
    return request.param


@pytest.fixture(params=POLICIES, ids=[cls.__name__ for cls in POLICIES])
def policy_cls(request):
    """One replacement policy class per parametrization.

    Named ``policy`` rather than ``strategy`` for the post-1.1 API; it currently
    receives the ``*Replacement`` classes.
    """
    return request.param


@pytest.fixture
def storage_dir(tmp_path):
    """Directory that must be empty once the test has released its stores."""
    path = tmp_path / "effidict-storage"
    path.mkdir()
    return path


@pytest.fixture
def _open_stores():
    return []


@pytest.fixture(autouse=True)
def storage_leak_detector(request, storage_dir, _open_stores):
    """Release the test's stores, then assert nothing was left on disk.

    Releasing first is deliberate. It means a surviving file is a *library*
    defect -- ``destroy()`` failing to remove everything, or ``close()`` failing
    to flush -- rather than a test that merely forgot to call cleanup.

    Mark a test ``keeps_storage`` to skip the assertion; issue 0.5's
    close-then-reopen specs need storage to outlive a close.
    """
    yield

    for store in reversed(_open_stores):
        _release(store)

    if request.node.get_closest_marker("keeps_storage"):
        return

    leaked = sorted(p.name for p in storage_dir.iterdir())
    assert not leaked, f"storage left behind in {storage_dir}: {leaked}"


def _release(store):
    """Best-effort teardown that survives the pre-3.1 lifecycle semantics.

    ``destroy()`` is preferred once it exists (issue 3.1); today ``close()`` is
    what removes storage. ``FileNotFoundError`` is tolerated narrowly because
    ``destroy()`` is not yet idempotent -- a known defect tracked in 3.1. Any
    other exception propagates rather than being swallowed.
    """
    for name in ("destroy", "close"):
        method = getattr(store, name, None)
        if method is None:
            continue
        try:
            method()
        except NotImplementedError:
            # Not built yet -- fall through and try the next mechanism rather
            # than reporting success, or storage would silently survive.
            continue
        except FileNotFoundError:
            # destroy() is not yet idempotent (issue 3.1); a second release of
            # the same storage is expected today.
            pass
        return


@pytest.fixture
def make_dict(request, storage_dir, _open_stores):
    """Build an ``EffiDict`` over a fresh temp store.

    Resolves ``backend``/``policy`` from the active matrix fixtures when the test
    is parametrized, otherwise falls back to ``DEFAULT_BACKEND``/``DEFAULT_POLICY``.
    """
    counter = {"n": 0}

    def factory(backend=None, policy=None, max_in_memory=2, max_bytes=None, **kwargs):
        if backend is None:
            backend = (
                request.getfixturevalue("backend_cls")
                if "backend_cls" in request.fixturenames
                else DEFAULT_BACKEND
            )
        if policy is None:
            policy = (
                request.getfixturevalue("policy_cls")
                if "policy_cls" in request.fixturenames
                else DEFAULT_POLICY
            )

        path = storage_dir / f"{backend.__name__}-{counter['n']}"
        counter["n"] += 1

        disk_backend = backend(str(path))
        _open_stores.append(disk_backend)

        effidict = EffiDict(
            disk_backend=disk_backend,
            replacement_strategy=policy(
                disk_backend=disk_backend, max_in_memory=max_in_memory
            ),
            # Passed unconditionally: silently dropping it would let issue 0.4's
            # byte-budget spec pass a budget that never arrives.
            max_bytes=max_bytes,
            **kwargs,
        )
        _open_stores.append(effidict)
        return effidict

    return factory


@pytest.fixture
def backend_spy():
    """Context manager recording every public method call on a backend."""

    @contextlib.contextmanager
    def factory(backend):
        spy = BackendSpy(backend)
        with spy:
            yield spy

    return factory


class BackendSpy:
    """Records calls to a backend instance's public methods.

    Wraps bound methods with instance attributes and deletes them on exit, so
    the class is never mutated and nothing leaks between tests. Classmethods
    (``create``/``open``/``temporary``) are left alone -- patching those on an
    instance is fragile and no spec needs it.
    """

    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self._counts = {}
        self._patched = []

    def __enter__(self):
        for name in self._method_names():
            original = getattr(self.backend, name)
            setattr(self.backend, name, self._wrap(name, original))
            self._patched.append(name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name in reversed(self._patched):
            delattr(self.backend, name)
        self._patched.clear()
        return False

    def count(self, name):
        """How many times ``name`` was called since entry or the last reset."""
        return self._counts.get(name, 0)

    def names(self):
        """Method names that were called at least once."""
        return sorted(name for name, n in self._counts.items() if n)

    def reset(self):
        self.calls.clear()
        self._counts.clear()

    def _method_names(self):
        names = set()
        for klass in type(self.backend).__mro__:
            for name, attr in vars(klass).items():
                if name.startswith("_"):
                    continue
                if isinstance(attr, (staticmethod, classmethod, property)):
                    continue
                if callable(attr):
                    names.add(name)
        return sorted(names)

    def _wrap(self, name, original):
        def recorder(*args, **kwargs):
            self.calls.append((name, args, dict(kwargs)))
            self._counts[name] = self._counts.get(name, 0) + 1
            return original(*args, **kwargs)

        return recorder
