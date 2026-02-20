from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Handle:
    """Small typed handle wrapper."""

    value: int

    @staticmethod
    def invalid() -> Handle:
        return Handle(-1)

    def is_valid(self) -> bool:
        return self.value >= 0


class HandleBuffer(Generic[T]):
    """Sparse handle buffer with stable integer IDs.

    Objects are stored behind small integer handles and can be removed/reused.
    """

    def __init__(self) -> None:
        self._data: dict[int, T] = {}
        self._next = 0

    @property
    def count(self) -> int:
        return len(self._data)

    def add(self, value: T) -> Handle:
        handle = Handle(self._next)
        self._next += 1
        self._data[handle.value] = value
        return handle

    def add_empty(self) -> Handle:
        handle = Handle(self._next)
        self._next += 1
        return handle

    def set_value(self, handle: Handle, value: T) -> None:
        self._data[handle.value] = value

    def try_get_value(self, handle: Handle) -> tuple[bool, T | None]:
        if handle.value in self._data:
            return True, self._data[handle.value]
        return False, None

    def get_value(self, handle: Handle) -> T:
        return self._data[handle.value]

    def remove_value(self, handle: Handle) -> None:
        self._data.pop(handle.value, None)

    def clear(self) -> None:
        self._data.clear()

    def get_list(self) -> list[T]:
        return [self._data[k] for k in sorted(self._data)]

    def items(self) -> Iterable[tuple[Handle, T]]:
        for k in sorted(self._data):
            yield Handle(k), self._data[k]
