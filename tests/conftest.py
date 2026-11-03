"""Shared harness for the Phase 0 specification suite.

Provides:

``backend_cls`` / ``policy_cls``
    Parametrized fixtures giving the 4 x 7 = 28 backend/policy matrix.
``make_dict``
    Factory for an ``EffiDict`` wired to a temp store, registered for teardown.
``backend_spy``
    Records backend method calls. Most non-functional guards count calls rather
    than measure time, so this lives here rather than in individual specs.
xfail reason drift check
    Fails the run when a spec dies on one issue's stub while its xfail marker
    blames another.
autouse leak detector
    Releases every store the test created, then asserts the storage directory is
    empty. Anything left behind is a library-level leak.
"""

from __future__ import annotations

import contextlib
import re

import pytest

from effidict import (
    EffiDict,
    FIFOPolicy,
    Hdf5Backend,
    JSONBackend,
    LFUPolicy,
    LIFOPolicy,
    LRUPolicy,
    MFUPolicy,
    MRUPolicy,
    PickleBackend,
    RandomPolicy,
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
    RandomPolicy,
    FIFOPolicy,
    LIFOPolicy,
    LRUPolicy,
    MRUPolicy,
    LFUPolicy,
    MFUPolicy,
]

#: Used when a test asks for `make_dict` without joining the matrix. Pickle
#: accepts every value kind and LRU is the most common policy, so an
#: unparametrized test exercises the least surprising combination.
DEFAULT_BACKEND = PickleBackend
DEFAULT_POLICY = LRUPolicy


def backend_marks(backend_cls):
    """Marks every param for ``backend_cls`` needs, wherever it is built."""
    if backend_cls is Hdf5Backend:
        return [pytest.mark.skipif(h5py is None, reason="h5py is not installed")]
    return []


def _backend_param(backend_cls):
    return pytest.param(
        backend_cls, id=backend_cls.__name__, marks=backend_marks(backend_cls)
    )


@pytest.fixture(params=[_backend_param(cls) for cls in BACKENDS])
def backend_cls(request):
    """One disk backend class per parametrization."""
    return request.param


@pytest.fixture(params=POLICIES, ids=[cls.__name__ for cls in POLICIES])
def policy_cls(request):
    """One eviction policy class per parametrization."""
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
        release_store(store)

    if request.node.get_closest_marker("keeps_storage"):
        return

    leaked = sorted(p.name for p in storage_dir.iterdir())
    assert not leaked, f"storage left behind in {storage_dir}: {leaked}"


def release_store(store):
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
            # A policy takes no constructor arguments since issue 1.1, and the
            # bound it used to carry is now the cache's item budget, which the
            # facade routes through for us.
            policy=policy(),
            max_in_memory=max_in_memory,
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


# --------------------------------------------------------------------------
# xfail reason drift
# --------------------------------------------------------------------------

#: A stub names the issue that owns it in the exception it raises
#: (``NotImplementedError("see issue #1.4")``), and an xfail reason names the
#: issue the spec is waiting for. When a test dies on a tagged stub the two have
#: to agree. When they do not, the marker is pointing at a gate the test never
#: reaches -- and the day that issue lands, the spec stays red behind the gate
#: that really stopped it while its strict marker goes on passing, so nobody is
#: told the thing it was written for is still untested. Two markers in this
#: suite had drifted that way before this check existed.
#:
#: Scaffolding, with an end date. It is worth having only while most of this
#: suite is red; once the implementation phases are done nothing raises
#: ``see issue #N`` any more, there is nothing left to compare, and this costs a
#: hook call per test for no answer. PLAN.md's Phase 8 lists it for removal,
#: together with its two specs in ``test_harness.py``.
_STUB_ISSUE = re.compile(r"see issue #(\d+\.\d+)")
_CITED_ISSUE = re.compile(r"\d+\.\d+")

#: Per-``Config`` rather than module-level: ``pytester`` runs its sub-pytest in
#: this same process, and shared state would let a sandbox's deliberate drift
#: fail the outer run.
_DRIFT = pytest.StashKey[dict]()


def pytest_configure(config):
    config.stash[_DRIFT] = {}


def pytest_runtest_makereport(item, call):
    """Record xfail reasons that disagree with what the test actually died on.

    An observer: it returns ``None``, so pytest builds the report itself. A
    hook rather than a fixture because ``call.excinfo`` is the only place the
    raising issue can be read, and a fixture never sees the exception.
    """
    if call.when != "call" or call.excinfo is None:
        return None
    raised = _STUB_ISSUE.search(str(call.excinfo.value))
    if raised is None:
        return None

    for marker in item.iter_markers(name="xfail"):
        reason = " ".join(str(marker.kwargs.get("reason", "")).split())
        cited = set(_CITED_ISSUE.findall(reason))
        # No issue cited is not drift: the marker simply makes no claim this
        # check can test.
        if cited and raised.group(1) not in cited:
            item.config.stash[_DRIFT][item.originalname or item.name] = (
                f"  {item.originalname or item.name}\n"
                f"      dies on issue #{raised.group(1)}: "
                f"{str(call.excinfo.value)[:70]}\n"
                f"      but cites {sorted(cited)}: {reason[:100]}"
            )
    return None


def pytest_terminal_summary(terminalreporter):
    drift = terminalreporter.config.stash.get(_DRIFT, None) or {}
    if not drift:
        return
    terminalreporter.section("xfail reasons that drifted from their cause")
    for message in sorted(drift.values()):
        terminalreporter.write_line(message)


def pytest_sessionfinish(session, exitstatus):
    """Fail the run on drift, whatever the tests themselves reported.

    A drifted marker makes its spec pass while watching nothing, so leaving
    this as a warning would be leaving it unread.
    """
    if session.config.stash.get(_DRIFT, None):
        session.exitstatus = 1
