"""A hash map built from scratch: separate chaining with automatic resizing."""
from __future__ import annotations

from typing import Any, Iterator


class HashMap:
    def __init__(self, capacity: int = 16, max_load: float = 0.75) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buckets: list[list[tuple[Any, Any]]] = [[] for _ in range(capacity)]
        self._size = 0
        self._max_load = max_load

    def _bucket(self, key: Any) -> list[tuple[Any, Any]]:
        return self._buckets[hash(key) % len(self._buckets)]

    def put(self, key: Any, value: Any) -> None:
        bucket = self._bucket(key)
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / len(self._buckets) > self._max_load:
            self._resize(len(self._buckets) * 2)

    def get(self, key: Any, default: Any = None) -> Any:
        for k, v in self._bucket(key):
            if k == key:
                return v
        return default

    def delete(self, key: Any) -> None:
        bucket = self._bucket(key)
        for i, (k, _) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self._size -= 1
                return
        raise KeyError(key)

    def _resize(self, new_capacity: int) -> None:
        old = self._buckets
        self._buckets = [[] for _ in range(new_capacity)]
        for bucket in old:
            for k, v in bucket:
                self._buckets[hash(k) % new_capacity].append((k, v))

    def items(self) -> Iterator[tuple[Any, Any]]:
        for bucket in self._buckets:
            yield from bucket

    def keys(self) -> Iterator[Any]:
        for k, _ in self.items():
            yield k

    def __contains__(self, key: Any) -> bool:
        return any(k == key for k, _ in self._bucket(key))

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: Any) -> Any:
        sentinel = object()
        v = self.get(key, sentinel)
        if v is sentinel:
            raise KeyError(key)
        return v

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    @property
    def capacity(self) -> int:
        return len(self._buckets)
