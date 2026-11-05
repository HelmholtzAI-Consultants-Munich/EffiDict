from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator
import sqlite3
import json
import os
import pickle
import shutil
import stat
import tempfile
import urllib.parse
import ast


def _no_store_at(path) -> str:
    return (
        f"no store at {path!r}; use create() to make one or "
        f"open_or_create() to accept either"
    )


def _exists(path) -> bool:
    """Whether anything is at ``path``, with "cannot tell" kept distinct.

    ``os.path.exists`` and ``os.path.isdir`` answer False for a path they were
    not allowed to inspect, which makes an intact store under an unsearchable
    parent directory indistinguishable from no store at all -- and the
    difference is the whole point of the refusals these predicates drive.
    Only ``FileNotFoundError`` is absence; every other failure is left to reach
    the caller as itself.

    ``lstat``, because a symlink pointing at nothing is still something at the
    path: ``stat`` follows it and reports the missing target, so a dangling
    link read as an empty path while every call that tries to use it fails
    with the path already taken.
    """
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _is_directory(path) -> bool:
    """``os.path.isdir`` with the same distinction. See :func:`_exists`.

    Follows symlinks, unlike :func:`_exists`: a link to a real store directory
    names that store and must open as one.
    """
    info = _stat(path)
    return info is not None and stat.S_ISDIR(info.st_mode)


def _names_one_entry(key) -> bool:
    """Whether ``key`` could name a single entry directly inside a store.

    Three of the four backends derive a storage *location* from the key, and
    for those a key is only a key if ``keys()`` could report it: one entry,
    directly in the store, named by the key itself. Anything carrying a
    separator, or one of the directory names, points elsewhere -- and
    ``os.path.join(store, "../outsider")`` points outside the store
    altogether, at a file the store has nothing to do with.

    Answering about such a path is not merely untidy. ``Store.delete`` refuses
    a key the store does not have, so a probe that said yes about an outside
    file made ``del d["../outsider"]`` remove it. Before membership became a
    point lookup the key listing could never name one, and this keeps that
    true.

    Not used by :class:`SqliteBackend`, which stores the key as data in a
    column: ``"a/b"`` is an ordinary key there, ``keys()`` reports it, and
    refusing it here would be the same mistake in the other direction.

    Rejecting these keys outright, with an error that says why, is issue 7.1.
    """
    return (
        isinstance(key, str)
        and key not in ("", os.curdir, os.pardir)
        and os.path.basename(key) == key
    )


def _is_regular_file(path) -> bool:
    """Whether ``path`` is a plain file -- the only thing a single-file store
    can be. A directory, a dangling symlink, a fifo or a socket occupies the
    path without being a store that was ever written there.
    """
    info = _stat(path)
    return info is not None and stat.S_ISREG(info.st_mode)


def _stat(path):
    try:
        return os.stat(path)
    except FileNotFoundError:
        # The only answer that means "nothing is here". ``NotADirectoryError``
        # from a file partway along the path is left to propagate too: nothing
        # can be at that path, but saying so is more use than "no store".
        return None


def _obstructed_at(path) -> str:
    return (
        f"{path!r} exists but is not a store; move it aside or choose "
        f"another path"
    )


def _store_already_at(path) -> str:
    return (
        f"a store already exists at {path!r}; use open() to read it or "
        f"open_or_create() to accept either"
    )

try:
    import h5py
except Exception:
    h5py = None

class DiskBackend(ABC):
    """Base contract for the persistent tier.

    Note the ``ABC`` base. Before issue 0.1 this class carried
    ``@abstractmethod`` decorators without ``ABCMeta``, which makes them inert:
    the base could be instantiated and an incomplete subclass was never
    checked. Activating it is deliberate and is pinned by
    ``test_abstract_bases_cannot_be_instantiated``, but it is a real
    behaviour change -- an out-of-tree subclass that omits ``serialize``,
    ``deserialize``, ``del_item``, ``keys`` or ``destroy`` used to construct and
    will now raise ``TypeError`` at instantiation. The four shipped backends are
    unaffected, which ``test_concrete_backends_remain_instantiable`` pins.

    Issue 2.2 added ``has`` and ``count`` to that required set. Both had
    concrete defaults written as a key listing, which every backend here now
    overrides with a point lookup; leaving the defaults in place would have let
    a backend added later inherit the O(N) membership that issue removed,
    without anything saying so.

    Phase 0 carries three further user-visible changes, listed here so the set
    is in one place:

    * ``ReplacementStrategy`` became an ABC for the same reason. Issue 1.2 then
      removed it and the seven ``*Replacement`` classes outright, so an
      out-of-tree subclass of any of them no longer imports; ``EvictionPolicy``
      and the ``*Policy`` classes replace them.
    * ``Hdf5Backend`` no longer requires ``numpy``. It never used it, and h5py
      depends on numpy anyway, so the gate could not fire; the ``ImportError``
      text changed with it.
    * ``EffiDict`` grew ``get``, ``setdefault``, ``popitem``, ``update``,
      ``clone``, ``flush``, ``iter_keys``, ``destroy``, ``fromkeys``,
      ``__reversed__`` and ``__or__`` as stubs that raise
      ``NotImplementedError``. They are declared ahead of issue 6.1 so the specs
      fail on the documented reason rather than ``AttributeError``, but the
      declaration is itself observable: ``hasattr(d, "get")`` flipped from False
      to True, so a duck-typing caller now takes the branch and gets
      ``NotImplementedError``. ``reversed(d)`` likewise used to return a
      ``reversed`` object that failed later on iteration, and now raises
      immediately.
    """

    #: Whether this backend's storage is a directory holding many files or a
    #: single file. Only :meth:`temporary` needs to know, to pick between
    #: ``mkdtemp`` and ``mkstemp``.
    _STORAGE_IS_DIRECTORY = True

    def __init__(self, storage_path, mode: str = "a"):
        """Open the store at exactly ``storage_path``.

        The path is used as given. It used to have
        ``f"{int(time.time())}_{id(self)}"`` appended, which meant the same
        argument named a different location on every construction -- so nothing
        written could ever be found again, and "persistent" was not true of this
        library in any useful sense. ``id()`` made it worse than merely useless:
        it is only unique among *live* objects and is recycled after collection,
        so a store built after an earlier one was freed could land on the
        abandoned store's path and read its values back as its own.

        ``mode`` is the whole of the create/open distinction, and every
        backend honours it: ``"w-"`` creates and refuses an occupied path,
        ``"r+"`` opens and refuses a missing one, ``"a"`` accepts either.
        :meth:`create`, :meth:`open` and :meth:`open_or_create` do nothing but
        pick one -- they must not check the path first, because a check and
        the operation that follows it can disagree, and the operation wins. A
        subclass that ignores ``mode`` silently loses both refusals.
        """
        self.storage_path = storage_path

    def _open_directory(self, mode: str) -> None:
        """Make or open the storage directory, honouring ``mode``.

        Shared by the two file-per-key backends. The mode is enforced by the
        call that makes the directory rather than by a check before it.
        ``makedirs`` without ``exist_ok`` either creates the directory or
        raises, with nothing in between, so two callers racing to ``create``
        the same path cannot both believe they won.

        ``r+`` not creating is the half that matters even with no race at all:
        a check followed by an unconditional ``makedirs`` produces exactly the
        empty store ``open`` exists to refuse, and reports it as a successful
        reopen.
        """
        if mode == "w-":
            try:
                os.makedirs(self.storage_path)
            except FileExistsError as exc:
                # ``makedirs`` refuses an occupied path without saying what is
                # occupying it, and a store here is a directory. Anything else
                # is an obstruction, not a store to be preserved.
                if not _is_directory(self.storage_path):
                    raise NotADirectoryError(
                        _obstructed_at(self.storage_path)
                    ) from exc
                raise FileExistsError(_store_already_at(self.storage_path)) from exc
        elif mode == "r+":
            if not _is_directory(self.storage_path):
                # Something being at the path means that whatever is wrong, it
                # is not absence. Calling it absence sent the caller to
                # create(), which then refused the very same path as already
                # occupied -- two refusals contradicting each other.
                if _exists(self.storage_path):
                    raise NotADirectoryError(_obstructed_at(self.storage_path))
                raise FileNotFoundError(_no_store_at(self.storage_path))
        else:
            os.makedirs(self.storage_path, exist_ok=True)

    @abstractmethod
    def serialize(self, key, value):
        pass

    @abstractmethod
    def deserialize(self, key):
        pass

    @abstractmethod
    def del_item(self, key):
        pass

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def has(self, key) -> bool:
        """Whether ``key`` is here. Must not enumerate the keyspace.

        The answer is whatever :meth:`keys` reports; an implementation that
        disagrees with it is wrong, however fast it is.

        Required rather than defaulted. The obvious default --
        ``key in self.keys()`` -- is a full listing per call, and ``key in d``
        reaches this method, so a backend that inherited it would quietly get
        the membership cost issue 2.2 removed. Quietly is the problem: the
        guard that measures that cost,
        ``test_membership_cost_does_not_grow_with_the_store``, runs over the
        backends in this repository, so a new one would not be measured until
        somebody added it there. Making the method required moves the failure
        to construction, where it cannot be missed.
        """

    @abstractmethod
    def count(self) -> int:
        """How many keys are here. Must not build the key listing.

        Must agree with ``len(keys())``, and required for the same reason as
        :meth:`has`: ``len(d)`` reaches this method.
        """

    @abstractmethod
    def destroy(self):
        pass

    def load_from_dict(self, dictionary):
        for key, value in dictionary.items():
            self.serialize(key, value)

    def iter_keys(self) -> Iterator:
        """Iterate over keys in this backend."""
        raise NotImplementedError("see issue #4.2")

    def read_many(self, keys) -> Dict:
        """Read multiple keys from this backend."""
        raise NotImplementedError("see issue #4.1")

    def write_many(self, items) -> None:
        """Write multiple items to this backend."""
        raise NotImplementedError("see issue #4.1")

    def delete_many(self, keys) -> None:
        """Delete multiple keys from this backend."""
        raise NotImplementedError("see issue #4.1")

    def supports(self, value) -> bool:
        """Return whether this backend can store ``value`` faithfully.

        An instance method rather than a classmethod: whether a value fits can
        depend on how the store was opened. Declared here so every backend
        exposes the same predicate -- ``test_backend_declares_its_value_domain``
        already treats it as part of the contract, and without a stub the
        failure is a bare ``AttributeError`` instead of a reference to the issue
        that implements it.
        """
        raise NotImplementedError("see issue #7.2")

    def close(self) -> None:
        """Release handles and flush buffers, leaving the storage in place.

        Distinct from ``destroy``, which deletes it. Without this, a
        non-destructive ``Store.close()`` cannot be built on the backend
        contract at all: SQLite holds a live ``sqlite3.Connection`` and HDF5 an
        open ``h5py.File``, and today the only method that closes either is
        ``destroy`` -- which also removes the data. That is why ``close()`` on
        the facade currently destroys the store (issue 3.1).

        Must be idempotent, and must leave the store reopenable via ``open``.

        The default does nothing, which is right for a backend that holds no
        handle between calls: the file-per-key backends open and close a file
        per operation, so there is nothing to release and nothing to flush.
        SQLite and HDF5 override it, because a live connection or file handle is
        exactly what stops the store being opened again.
        """

    def compact(self) -> None:
        """Reclaim unused storage space."""
        raise NotImplementedError("see issue #7.2")

    @classmethod
    def create(cls, path, **kwargs):
        """Build a new store at ``path``, refusing to touch an existing one.

        The refusal is the point: ``create`` is how a caller says "there should
        be nothing here", and silently opening what is already there would hand
        them someone else's data under the impression it was fresh.

        Enforced by the operation that makes the storage, not by a check before
        it. A check-then-create can be beaten between its two steps, and on the
        backends whose construction creates unconditionally it was never a
        guarantee at all -- whatever the check decided, the constructor went
        ahead and made the store.
        """
        return cls(path, mode="w-", **kwargs)

    @classmethod
    def open(cls, path, **kwargs):
        """Open the store at ``path``, refusing to create a missing one.

        The mirror of :meth:`create`, and the refusal matters as much: opening
        a path that holds nothing used to produce an empty store that looked
        like a successfully reopened one, so a typo in a path read as data loss.
        """
        return cls(path, mode="r+", **kwargs)

    @classmethod
    def open_or_create(cls, path, **kwargs):
        """Open the store at ``path``, creating it if there is none."""
        return cls(path, mode="a", **kwargs)

    @classmethod
    def temporary(cls, prefix: str = "effidict-"):
        """Build a scratch store at a path nothing else will be given.

        Uniqueness comes from ``tempfile``, which is collision-free by
        construction -- it creates the name it returns, so no second caller can
        be handed it. That is the replacement for salting the path with
        ``id(self)``, which only ever approximated this and got it wrong as soon
        as an object was collected.

        Destroying it is still the caller's to do; ``temporary`` owning its own
        cleanup is issue #3.1.
        """
        if cls._STORAGE_IS_DIRECTORY:
            temporary_path = tempfile.mkdtemp(prefix=prefix)
            discard = shutil.rmtree
        else:
            # ``mkstemp`` leaves the file open and existing, so the handle is
            # closed and the path reused. "w" rather than "a" because what is
            # there is an empty file rather than a store -- HDF5 refuses to
            # read it as one.
            handle, temporary_path = tempfile.mkstemp(prefix=prefix)
            os.close(handle)
            discard = os.remove

        try:
            return cls(temporary_path, mode="w")
        except BaseException:
            # ``tempfile`` has already made the path, and a constructor that
            # raised leaves no object through which to reach it -- an
            # ``Hdf5Backend`` without h5py installed raises before it owns
            # anything. Nobody else will ever know this path, so the file would
            # sit in the system temp directory until the machine was cleaned.
            discard(temporary_path)
            raise


class SqliteBackend(DiskBackend):
    _STORAGE_IS_DIRECTORY = False

    def __init__(self, storage_path, mode: str = "a"):
        """Connect, letting the connection itself enforce ``mode``.

        ``sqlite3.connect`` on a plain path creates the database when it is
        missing, so neither refusal can be left to it. ``O_EXCL`` gives the
        create half atomically, and the ``mode=rw`` URI gives the open half --
        it opens an existing database and fails rather than making one.
        """
        super().__init__(storage_path, mode)
        if mode == "w-":
            try:
                handle = os.open(
                    self.storage_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError as exc:
                # ``O_EXCL`` refuses a directory for existing too. A file here
                # may or may not be a database -- that cannot be known without
                # opening it, and claiming a store is the useful guess -- but a
                # directory certainly is not one.
                if _is_directory(self.storage_path):
                    raise IsADirectoryError(
                        _obstructed_at(self.storage_path)
                    ) from exc
                if not _is_regular_file(self.storage_path):
                    # A dangling symlink is the case that matters: the path is
                    # taken, so the create refused, but nothing is there to be
                    # the store this would otherwise claim to have found.
                    raise FileExistsError(_obstructed_at(self.storage_path)) from exc
                raise FileExistsError(_store_already_at(self.storage_path)) from exc
            os.close(handle)
            self.conn = sqlite3.connect(self.storage_path)
        elif mode == "r+":
            uri = (
                "file:"
                + urllib.parse.quote(os.path.abspath(self.storage_path))
                + "?mode=rw"
            )
            try:
                self.conn = sqlite3.connect(uri, uri=True)
            except sqlite3.OperationalError as exc:
                # SQLite reports a missing database and one it refuses to open
                # with the same SQLITE_CANTOPEN and the same message, so the
                # path is the only thing that says which happened. Every other
                # connection failure -- an unreadable file, a directory in the
                # way -- keeps its own error instead of being reported as a
                # store that is not there.
                if _exists(self.storage_path):
                    raise
                raise FileNotFoundError(_no_store_at(self.storage_path)) from exc
        else:
            self.conn = sqlite3.connect(self.storage_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)"
        )

    def serialize(self, key, value):
        json_value = json.dumps(value)
        self.cursor.execute(
            "REPLACE INTO data (key, value) VALUES (?, ?)", (key, json_value)
        )
        self.conn.commit()

    def deserialize(self, key):
        self.cursor.execute("SELECT value FROM data WHERE key=?", (key,))
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        raise KeyError(key)
    
    def del_item(self, key):
        self.cursor.execute("DELETE FROM data WHERE key=?", (key,))
        self.conn.commit()

    def keys(self):
        self.cursor.execute("SELECT key FROM data")
        return [key[0] for key in self.cursor.fetchall()]

    def has(self, key) -> bool:
        """One indexed probe. ``key`` is the primary key, so this is a seek.

        ``typeof`` as well as value, because ``=`` alone is not key identity
        here. The column is declared ``TEXT``, so its affinity converts an
        integer to text on the way in *and* on the way into a comparison:
        writing ``1`` stores ``'1'``, and probing with ``1`` then matches the
        row that ``keys()`` reports as ``'1'``. Two distinct keys would read as
        one, and ``len()`` would be short by one for every such pair. Comparing
        the stored type with the probe's rejects that while still seeking on
        the primary key index. Non-string keys are issue 7.1's to refuse; what
        matters here is that this answers what ``keys()`` reports.
        """
        self.cursor.execute(
            "SELECT 1 FROM data WHERE key=? AND typeof(key)=typeof(?) LIMIT 1",
            (key, key),
        )
        return self.cursor.fetchone() is not None

    def count(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM data")
        return self.cursor.fetchone()[0]
    
    def load_from_dict(self, dictionary):
        with self.conn:
            items_to_insert = [
                (key, json.dumps(value)) for key, value in dictionary.items()
            ]
            self.cursor.executemany(
                "REPLACE INTO data (key, value) VALUES (?, ?)",
                items_to_insert,
            )

    def close(self) -> None:
        """Close the connection. ``sqlite3`` tolerates a second call."""
        self.conn.close()

    def destroy(self):
        self.conn.close()
        os.remove(self.storage_path)


class PickleBackend(DiskBackend):
    def __init__(self, storage_path, mode: str = "a"):
        super().__init__(storage_path, mode)
        self._open_directory(mode)

    def serialize(self, key, value):
        with open(os.path.join(self.storage_path, key), "wb") as f:
            pickle.dump(value, f)

    def deserialize(self, key):
        try:
            with open(os.path.join(self.storage_path, key), "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise KeyError(key)
        
    def del_item(self, key):
        try:
            os.remove(os.path.join(self.storage_path, key))
        except FileNotFoundError:
            pass

    def keys(self):
        return os.listdir(self.storage_path)

    def has(self, key) -> bool:
        """Whether this key's file is there -- one ``stat``, not a listing."""
        return _names_one_entry(key) and _is_regular_file(
            os.path.join(self.storage_path, key)
        )

    def count(self) -> int:
        """``scandir``, not ``len(listdir())``: the entries are not collected."""
        with os.scandir(self.storage_path) as entries:
            return sum(1 for _ in entries)

    def destroy(self):
        shutil.rmtree(self.storage_path)


class Hdf5Backend(DiskBackend):
    _STORAGE_IS_DIRECTORY = False

    def __init__(self, storage_path, mode: str = "a"):
        """Open the HDF5 file in the mode the caller's intent implies.

        The mode is load-bearing here in a way it is not for the other three.
        This used to be ``"w"`` unconditionally, which truncates: opening an
        existing store destroyed it, so the data was lost by the act of looking
        for it. ``"w-"`` creates and refuses to clobber, ``"r+"`` opens and
        refuses to create, and ``"a"`` accepts either -- which makes the three
        classmethods atomic rather than check-then-open.
        """
        if h5py is None:
            raise ImportError("h5py is required for Hdf5Backend but is not installed.")
        super().__init__(storage_path, mode)
        try:
            self.file = h5py.File(self.storage_path, mode)
        except OSError as exc:
            # h5py reports both refusals as a bare OSError. Translated so the
            # four backends are distinguishable by behaviour rather than by
            # which exception type the caller has to be ready for.
            occupied = _exists(self.storage_path)
            if occupied and not _is_regular_file(self.storage_path):
                # A directory, or a symlink pointing at nothing: the path is
                # taken, but not by anything this backend could have written.
                # True whichever mode asked -- create cannot have the path and
                # open cannot read it -- and h5py's own answer for a dangling
                # link is ENOENT, which would report an occupied path as an
                # empty one.
                if _is_directory(self.storage_path):
                    raise IsADirectoryError(
                        _obstructed_at(self.storage_path)
                    ) from exc
                raise FileExistsError(_obstructed_at(self.storage_path)) from exc
            if mode == "w-" and occupied:
                raise FileExistsError(_store_already_at(self.storage_path)) from exc
            if mode == "r+" and not occupied:
                raise FileNotFoundError(_no_store_at(self.storage_path)) from exc
            raise

    def serialize(self, key, value):
        """
        Store data and its Python type as an HDF5 attribute.
        """
        if key in self.file:
            del self.file[key] 

        type_name = type(value).__name__
        
        if value is None:
            ds = self.file.create_dataset(key, data=h5py.Empty("f"))
        elif isinstance(value, (dict, list, tuple)):
            ds = self.file.create_dataset(key, data=repr(value))
        else:
            ds = self.file.create_dataset(key, data=value)
            
        ds.attrs['python_type'] = type_name

    def deserialize(self, key):
        """
        Read data and use the 'python_type' attribute to cast it back correctly.
        """
        try:
            dataset = self.file[key]
            type_name = dataset.attrs.get('python_type')

            if type_name is None:
                raise TypeError(f"Dataset for key '{key}' is missing 'python_type' attribute.")

            if type_name == 'NoneType':
                return None
            elif type_name in ('dict', 'list', 'tuple'):
                str_data = dataset.asstr()[()]
                return ast.literal_eval(str_data)
            elif type_name == 'str':
                return dataset.asstr()[()]
            else: 
                return dataset[()]

        except KeyError:
            raise KeyError(key)
        
    def del_item(self, key):
        try:
            del self.file[key]
        except KeyError:
            pass

    def keys(self):
        return list(self.file.keys())

    def has(self, key) -> bool:
        # h5py reads a separator as a path inside the file, so "a/b" can be
        # found while keys() -- top-level members only -- reports "a".
        # Contained, unlike the other two, but the same disagreement.
        return _names_one_entry(key) and key in self.file

    def count(self) -> int:
        return len(self.file)
    
    def close(self) -> None:
        """Close the file. ``h5py`` tolerates a second call."""
        self.file.close()

    def destroy(self):
        self.file.close()
        os.remove(self.storage_path)


class JSONBackend(DiskBackend):
    def __init__(self, storage_path, mode: str = "a"):
        super().__init__(storage_path, mode)
        self._open_directory(mode)

    def _file_path(self, key):
        return os.path.join(self.storage_path, f"{key}.json")

    def serialize(self, key, value):
        with open(self._file_path(key), "w") as f:
            json.dump(value, f)

    def deserialize(self, key):
        try:
            with open(self._file_path(key), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise KeyError(key)
        
    def del_item(self, key):
        try:
            os.remove(self._file_path(key))
        except FileNotFoundError:
            pass

    def keys(self):
        return [os.path.splitext(f)[0] for f in os.listdir(self.storage_path)]

    def has(self, key) -> bool:
        return _names_one_entry(key) and _is_regular_file(self._file_path(key))

    def count(self) -> int:
        with os.scandir(self.storage_path) as entries:
            return sum(1 for _ in entries)
    
    def destroy(self):
        shutil.rmtree(self.storage_path)
