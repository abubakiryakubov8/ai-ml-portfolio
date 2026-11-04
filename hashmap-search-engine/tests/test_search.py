import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashmap import HashMap  # noqa: E402
from search_engine import SearchEngine, tokenize  # noqa: E402


def test_put_get_overwrite_delete():
    m = HashMap()
    m["a"] = 1
    m["a"] = 2
    assert m["a"] == 2 and len(m) == 1
    m.delete("a")
    assert "a" not in m and len(m) == 0
    with pytest.raises(KeyError):
        m.delete("a")
    with pytest.raises(KeyError):
        m["missing"]


def test_resize_keeps_all_keys():
    m = HashMap(capacity=2)
    for i in range(200):
        m[i] = i * i
    assert len(m) == 200
    assert m.capacity > 2
    assert all(m[i] == i * i for i in range(200))
    assert sorted(m.keys()) == list(range(200))


def test_collisions_are_handled():
    class Bad:
        def __init__(self, n): self.n = n
        def __hash__(self): return 1
        def __eq__(self, o): return isinstance(o, Bad) and self.n == o.n
    m = HashMap()
    for i in range(10):
        m[Bad(i)] = i
    assert all(m[Bad(i)] == i for i in range(10))


def test_tokenize_multilingual():
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("안녕하세요 세계") == ["안녕하세요", "세계"]


def test_search_ranking():
    e = SearchEngine()
    e.add_document("a", "microcontroller sensor sensor sensor")
    e.add_document("b", "microcontroller motor")
    e.add_document("c", "unrelated text")
    results = e.search("sensor microcontroller")
    assert [d for d, _ in results] == ["a", "b"]
    assert e.search("nothing") == []


def test_duplicate_document_rejected():
    e = SearchEngine()
    e.add_document("a", "x")
    with pytest.raises(ValueError):
        e.add_document("a", "y")
