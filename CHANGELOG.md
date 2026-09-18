# Changelog

## v0.3.0

A rewrite of the retrieval pipeline, measured at every step against an
evaluation harness that generates its own gold answers from your corpus.

### Breaking

- **Every collection must be re-ingested.** Chunk sizing, the keyword
  tokeniser and all three index formats changed. Loading an older index raises
  an error naming the collection rather than querying it with a tokeniser it
  was not built with.
- A collection is now three files: `<name>.chunks`, `<name>.bm25` and
  `<name>.<backend>.index`. The chunk text is stored once instead of twice.
- `rank-bm25` is no longer a dependency.
- `hybrid_search()` still returns a list of strings; the new `search()` returns
  scored candidates and is what the pipeline uses.

### Retrieval returned nothing by default

- The shipped `relevance_threshold` of `0.05` sat above the highest score
  Reciprocal Rank Fusion can produce (`3/61 = 0.049`), so every search returned
  an empty list. Stored configs carrying the old value are migrated on load.
- `get_config_value(...) or <default>` turned a configured `0.0` or `False`
  back into the default. Settings are now read with an explicit `None` check.

### Chunks were indexed by their opening fragment

- 500-word chunks tokenise to ~557 word-pieces against a 256-token model, so
  54% of every chunk never reached its own vector. Chunks are now packed to the
  model's real token budget and verified against the tokeniser.

### Abstention

- Retrieval can now report that the corpus cannot answer a question, instead of
  returning its top-k chunks regardless. A question is answered if it is
  lexically grounded (its words exist in the corpus, weighted by IDF) **or**
  semantically confident (a chunk is close enough in embedding space).
- Measured on 10,399 chunks of Python documentation: abstention on unanswerable
  questions went from 0% to 100%, with recall@5, precision@1 and MRR unchanged.
- Tunable via `abstention_enabled`, `lexical_evidence_floor` and
  `semantic_confidence_floor`; readable and writable through `/config`.

### Performance

- Keyword scoring walks an inverted index instead of scoring the whole corpus:
  **104.86ms → 0.85ms** per query at 80,000 chunks. Query p50 on a real corpus
  went 13.3ms → 5.4ms, now bounded by embedding the query.
- Source attribution used to rebuild a corpus-wide lookup on every chat request
  (410ms at 200,000 chunks); candidates carry their own index.
- Retrieval runs off the event loop, so concurrent chat requests no longer
  serialise behind each other.
- The query cache now holds the whole retrieval outcome rather than chunk text
  alone, so the chat proxy is served from it — it previously wrote to the cache
  and never read from it.
- Chunk text is stored once on disk and held once in memory: 51.0MB → 39.4MB
  for a 10,399-chunk collection.

### Durability

- Index files are written to a temporary file and moved into place. Writing in
  place produced **34,704 torn reads out of 34,709** during concurrent access;
  after the change, 0.
- The chunk store is written before the vector index, so a crash mid-build
  reads as "not built" rather than as an index promising chunks that were never
  written.
- Ingest holds an advisory lock; a second build is refused, and a lock left by
  a process that died is detected and broken.
- No index file contains a pickle. Indexes are derived from downloaded ZIMs,
  and `pickle.load` on such a file executes whatever it contains.

### Correctness fixes

- Web search results were appended to the live vector index's text list on every
  request, growing it without bound and desynchronising it from the vectors and
  metadata.
- Pseudo-relevance feedback never expanded anything: it was called without a
  first-pass result. It now runs a feedback pass.
- The chat proxy ran a weaker pipeline than `/search` — no query expansion, the
  wrong reranker model, a hardcoded candidate count. Both share one function.
- BM25 tokenised on whitespace, so `asyncio.gather,` and `asyncio.gather` were
  different terms. A shared tokeniser is used at build and query time.
- `semantic_backend` and `keyword_backend` were stored in config and never used
  to construct anything, leaving `faiss_ivf` and `bm25_plus` unreachable.
- Startup auto-load and `db list` probed `<name>.index` / `<name>.pkl`, which no
  backend has ever written. Discovery now asks the backend.
- FAISS IVF used a cluster count fixed at `sqrt(100000)` regardless of corpus
  size, trained on the first 100 chunks ingested, and left `nprobe` at 1.
- Selecting `faiss_ivf` where FAISS and torch ship conflicting OpenMP runtimes
  aborted the process during training. That environment is now detected and the
  exact backend used instead, with a warning.

### New

- `tensor-serve eval` measures retrieval on a collection using questions
  generated from the corpus, with gold answers known by construction. Reports
  recall@k, precision@1, MRR, abstention rate and a per-stage latency
  breakdown.
- `evals/baseline-python-docs.json` records a baseline to compare against.

### Known limits

- Ingest is 86.6% embedding, so it is bounded by the model (~340 chunks/s here).
  Parallelising the ZIM walk would address 11% of the work and contend for the
  same cores; it was measured and not built.
- `run_multi_ingest` merges collections into one index. Measured cost to a small
  corpus merged into a large one: 2.5% recall@5, 1.7% precision@1.
- Index build time rose (7.95s → 22.66s at 80,000 chunks), traded against every
  query being ~120x faster.

## v0.2.0

Search complexity profiles, pluggable keyword and semantic backends, query
expansion, and cross-encoder reranking.

## v0.1.0

Initial release.
