import functools
import random
import sys
from collections import deque
from itertools import chain
from types import FunctionType, ModuleType

import matplotlib.pyplot as plt
import seaborn as sns

from effidict._db_dict import LRUDict

sns.set()

# Set of types that signify "atomic" non-iterable types
BLACKLIST = type, ModuleType, FunctionType


def generate_random_sequence(length):
    """Generates a random DNA sequence of a given length."""
    return "".join(random.choice("ACGT") for _ in range(length))


def total_size(o, handlers={}):
    """Returns the approximate memory footprint of an object and all of its contents."""

    # Handler for standard dictionaries
    dict_handler = lambda d: chain.from_iterable(d.items())

    # Handler for LRUDict
    lrudict_handler = lambda d: chain.from_iterable(d.items())

    # Define all default handlers
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
        LRUDict: lrudict_handler,
    }

    all_handlers.update(handlers)  # user handlers take precedence

    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(obj):
        if id(obj) in seen:  # do not double count the same object
            return 0
        seen.add(id(obj))
        s = sys.getsizeof(obj, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(obj, typ):
                s += sum(map(sizeof, handler(obj)))
                break
        return s

    return sizeof(o)


def memory_size_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        size = total_size(result) * 10**-6
        # print(f"Total memory size of the object returned by '{func.__name__}': {size} bytes")
        return result, size

    return wrapper


def plot_times_and_sizes(times, sizes, n_regions_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))

    ax1.plot(n_regions_values, sizes, marker="o")
    ax1.set_title("Size of Dictionary on Memory vs. number of regions")
    ax1.set_xlabel("Number of regions")
    ax1.set_ylabel("Size on Memory (Mega bytes)")

    ax2.plot(n_regions_values, times, marker="o")
    ax2.set_title("Execution time vs. number of regions")
    ax2.set_xlabel("Number of regions")
    ax2.set_ylabel("Time (s)")

    fig.show()


def plot_times(all_times, n_regions_values):
    plt.figure(figsize=(10, 4))
    for times, legend in all_times:
        plt.plot(n_regions_values, times, marker="o", label=legend)
    plt.title("Execution time vs. number of regions")
    plt.xlabel("Number of regions")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.show()
