"""Accumulators: driver-readable values updated from inside Ray Data UDFs.

PROTOTYPE for https://github.com/ray-project/ray/issues/48048. The API and the
module location are open for discussion with maintainers.

How it works:

* ``Accumulator.add`` updates an in-memory partial value held on the current
  task's ``TaskContext``. There is no RPC.
* ``_map_task`` snapshots the partials into ``TaskExecWorkerStats.custom_op_stats``
  (as an ``AccumulatorStats``) on every block it yields.
* The driver merges the *last* block's snapshot of each finished task into the
  ``Accumulator`` with the user's ``combine`` function.

Delivery semantics (see the design doc): each successful task attempt contributes
once. Lineage reconstruction can contribute a task a second time.
"""

import threading
import uuid
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ray.data.block import CustomOpStats
from ray.util.annotations import PublicAPI

# Driver-side registry so that partials coming back from tasks can be merged into
# the Accumulator object the user holds. Weak, so a dropped Accumulator is freed.
_REGISTRY: "weakref.WeakValueDictionary[str, Accumulator]" = (
    weakref.WeakValueDictionary()
)


@dataclass(frozen=True)
class AccumulatorStats(CustomOpStats):
    """Per-task, cumulative partial values keyed by accumulator id."""

    partials: Dict[str, Any]


def _min(a, b):
    return b if a is None else a if b is None else min(a, b)


def _max(a, b):
    return b if a is None else a if b is None else max(a, b)


@PublicAPI(stability="alpha")
class Accumulator:
    """A value that UDFs add to and the driver reads.

    ``combine`` must be associative and commutative, and ``zero()`` must be its
    identity, because partial values from different tasks are merged in
    arbitrary order.

    Example:
        .. testcode::
            :skipif: True

            import ray

            total = ray.data.Accumulator.sum()

            def f(batch):
                total.add(int(batch["id"].sum()))
                return batch

            ray.data.range(100).map_batches(f).materialize()
            assert total.value == 4950

    Args:
        zero: Returns the initial (identity) value.
        add: ``add(current, item) -> new`` folds one item into a value.
        combine: ``combine(a, b) -> merged`` merges two partial values.
    """

    def __init__(
        self,
        zero: Callable[[], Any],
        add: Callable[[Any, Any], Any],
        combine: Callable[[Any, Any], Any],
    ):
        self._id = uuid.uuid4().hex
        self._zero = zero
        self._add = add
        self._combine = combine
        self._init_driver_state()
        _REGISTRY[self._id] = self

    def _init_driver_state(self):
        self._lock = threading.Lock()
        self._value = self._zero()

    # Pickled copies go to workers. They only need the functions and the id, and
    # they must not register themselves (``__init__`` is not called on unpickle).
    def __getstate__(self):
        return {
            "_id": self._id,
            "_zero": self._zero,
            "_add": self._add,
            "_combine": self._combine,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = None
        self._value = None

    @classmethod
    def sum(cls) -> "Accumulator":
        """An accumulator that sums the items added to it."""
        return cls(zero=int, add=lambda a, x: a + x, combine=lambda a, b: a + b)

    @classmethod
    def min(cls) -> "Accumulator":
        """An accumulator that keeps the smallest item added to it."""
        return cls(zero=lambda: None, add=_min, combine=_min)

    @classmethod
    def max(cls) -> "Accumulator":
        """An accumulator that keeps the largest item added to it."""
        return cls(zero=lambda: None, add=_max, combine=_max)

    def add(self, item: Any) -> None:
        """Fold ``item`` into this task's partial value.

        Call this from inside a Ray Data UDF.
        """
        # Imported here to avoid a circular import at module load.
        from ray.data._internal.execution.interfaces.task_context import TaskContext

        ctx = TaskContext.get_current()
        if ctx is None:
            raise RuntimeError(
                "Accumulator.add() can only be called from inside a Ray Data UDF "
                "(for example the function passed to map_batches). Async UDFs are "
                "not supported yet."
            )
        partials = ctx.accumulator_partials
        current = partials[self._id] if self._id in partials else self._zero()
        partials[self._id] = self._add(current, item)

    @property
    def value(self) -> Any:
        """The merged value of all tasks that finished so far.

        Final once the dataset's execution completes. During execution it is
        partial.
        """
        if self._lock is None:
            raise RuntimeError("Accumulator.value can only be read on the driver.")
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset the driver-side value to ``zero()``."""
        if self._lock is None:
            raise RuntimeError("Accumulator.reset() can only be called on the driver.")
        with self._lock:
            self._value = self._zero()

    def _merge(self, partial: Any) -> None:
        with self._lock:
            self._value = self._combine(self._value, partial)


def merge_accumulator_stats(custom_op_stats: Optional[List[CustomOpStats]]) -> None:
    """Merge one finished task's final snapshot into the driver's accumulators."""
    if not custom_op_stats:
        return
    for stats in custom_op_stats:
        if not isinstance(stats, AccumulatorStats):
            continue
        for acc_id, partial in stats.partials.items():
            acc = _REGISTRY.get(acc_id)
            if acc is not None:
                acc._merge(partial)
