"""A small TF-IDF search engine whose index is stored in the custom HashMap."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from hashmap import HashMap

TOKEN_RE = re.compile(r"\w+", re.UNICODE)  # works for Latin, Cyrillic and Hangul


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class SearchEngine:
    def __init__(self) -> None:
        self.index = HashMap()   # term -> HashMap(doc_id -> term frequency)
        self.docs = HashMap()    # doc_id -> original text

    def add_document(self, doc_id: str, text: str) -> None:
        if doc_id in self.docs:
            raise ValueError(f"duplicate document id: {doc_id}")
        self.docs[doc_id] = text
        for term in tokenize(text):
            postings = self.index.get(term)
            if postings is None:
                postings = HashMap()
                self.index[term] = postings
            postings[doc_id] = postings.get(doc_id, 0) + 1

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        n_docs = len(self.docs)
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            postings = self.index.get(term)
            if postings is None:
                continue
            idf = math.log((1 + n_docs) / (1 + len(postings))) + 1
            for doc_id, tf in postings.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]


def load_folder(engine: SearchEngine, folder: Path) -> int:
    files = sorted(folder.glob("*.txt"))
    for path in files:
        engine.add_document(path.name, path.read_text(encoding="utf-8"))
    return len(files)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs", type=Path, default=root / "sample_docs")
    p.add_argument("query", nargs="*", help="query text (omit for interactive mode)")
    args = p.parse_args()

    engine = SearchEngine()
    n = load_folder(engine, args.docs)
    if n == 0:
        raise SystemExit(f"No .txt files found in {args.docs}")
    print(f"Indexed {n} documents, {len(engine.index)} unique terms")

    def run(q: str) -> None:
        results = engine.search(q)
        if not results:
            print("  no matches")
        for doc_id, score in results:
            print(f"  {score:6.2f}  {doc_id}")

    if args.query:
        run(" ".join(args.query))
    else:
        while True:
            q = input("search> ").strip()
            if q in {"", "quit", "exit"}:
                break
            run(q)


if __name__ == "__main__":
    main()
