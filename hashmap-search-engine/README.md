# Hash Map Mini Search Engine

A hash map and a TF-IDF search engine built from scratch in pure Python, with no external dependencies. The search index is stored inside the custom hash map.

## Run it

```bash
python src/search_engine.py edge ai microcontroller     # one-shot query
python src/search_engine.py                             # interactive mode
python -m pytest -q                                     # tests (needs pytest)
```

Put your own `.txt` files in `sample_docs/` (or pass `--docs folder`).

## Design

**`hashmap.py`**: separate chaining; the table doubles when the load factor exceeds 0.75. Supports `put/get/delete`, `[]`, `in`, `len`, `items`, `keys`. Average O(1) lookup.

**`search_engine.py`**
- Tokenizer is Unicode-aware, so Korean, Russian and English text all work
- Inverted index: `term → HashMap(doc_id → term frequency)`
- Ranking: TF-IDF, `score = Σ tf × (ln((1+N)/(1+df)) + 1)`

## Tests

Cover overwrite/delete, resizing under 200 inserts, forced hash collisions, multilingual tokenizing, ranking order, and duplicate-document rejection.

## Limitations

- Uses Python's built-in `hash()` for keys; a custom hash function would be the next step
- No stemming or stop-word removal; the index lives in memory only
