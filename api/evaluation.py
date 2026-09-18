"""
Retrieval evaluation with gold answers known by construction.

The point is to be checkable on *your* corpus in seconds, with no hand
labelling: every question is generated from the loaded collection in a way that
makes its correct answer known before retrieval runs.

    family      how the gold answer is known
    ----------  ---------------------------------------------------------------
    passage     a span copied verbatim out of one chunk; that chunk is the answer
    title       an article's own title; that article's chunks are the answer
    signature   an article's rarest terms; that article's chunks are the answer
    nonsense    words that appear nowhere in the corpus; abstaining is correct

`signature` is an upper bound — it asks each article about its own rarest
vocabulary, so a real question phrased in shared words scores lower. A low
score there is a fact about the corpus, not necessarily a bug to tune away.

`nonsense` measures abstention, and is reported as a first-class number. A
pipeline that always returns its top_k chunks scores 0% and is stating that it
cannot tell a real question from gibberish.
"""

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from api.lexical import tokenize

FAMILIES = ("passage", "title", "signature", "nonsense")

_SYLLABLES = ("zor", "quix", "vemp", "glirn", "thal", "wub", "kresh", "plom", "jyx", "drav")


@dataclass
class Question:
    family: str
    query: str
    gold: set                      # chunk indices that count as correct
    note: str = ""


@dataclass
class FamilyResult:
    family: str
    n: int = 0
    hits: int = 0                  # any gold chunk inside top-k
    top1: int = 0                  # a gold chunk ranked first
    reciprocal_ranks: List[float] = field(default_factory=list)
    empty: int = 0                 # returned nothing

    @property
    def recall(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def precision_at_1(self) -> float:
        return self.top1 / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return sum(self.reciprocal_ranks) / self.n if self.n else 0.0

    @property
    def abstention(self) -> float:
        return self.empty / self.n if self.n else 0.0


# ---------------------------------------------------------------- question gen


def _articles(metadata: Sequence[dict], n_chunks: int) -> Dict[str, List[int]]:
    """Group chunk indices by the article they came from."""
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx in range(n_chunks):
        meta = metadata[idx] if idx < len(metadata) else None
        title = (meta or {}).get("article_title")
        if title:
            grouped[str(title)].append(idx)
    return grouped


def _document_frequencies(token_lists: Sequence[List[str]]) -> Counter:
    df = Counter()
    for tokens in token_lists:
        df.update(set(tokens))
    return df


def _passage_question(rng: random.Random, texts, idx: int, words: int = 12) -> Optional[Question]:
    """A span lifted verbatim from one chunk. That chunk is the answer."""
    chunk_words = texts[idx].split()
    if len(chunk_words) < words + 2:
        return None
    start = rng.randrange(0, len(chunk_words) - words)
    return Question(
        family="passage",
        query=" ".join(chunk_words[start:start + words]),
        gold={idx},
        note=f"chunk {idx}",
    )


def _signature_question(
    rng: random.Random,
    token_lists,
    df: Counter,
    total_chunks: int,
    title: str,
    chunk_ids: List[int],
    terms: int = 3,
) -> Optional[Question]:
    """The article's rarest vocabulary. Its own chunks are the answer."""
    tf = Counter()
    for cid in chunk_ids:
        tf.update(token_lists[cid])
    if not tf:
        return None

    scored = sorted(
        tf.items(),
        key=lambda kv: kv[1] * math.log(total_chunks / (1 + df[kv[0]])),
        reverse=True,
    )
    picked = [term for term, _ in scored[:terms] if len(term) > 2]
    if not picked:
        return None

    return Question(
        family="signature",
        query=" ".join(picked),
        gold=set(chunk_ids),
        note=title,
    )


def _nonsense_questions(rng: random.Random, vocabulary: set, count: int) -> List[Question]:
    """Words that appear nowhere in the corpus. Abstaining is the right answer."""
    questions = []
    attempts = 0
    while len(questions) < count and attempts < count * 50:
        attempts += 1
        words = [
            "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3)))
            for _ in range(4)
        ]
        if any(w in vocabulary for w in words):
            continue
        questions.append(
            Question(family="nonsense", query=" ".join(words), gold=set(), note="unanswerable")
        )
    return questions


def build_questions(
    texts: Sequence[str],
    metadata: Sequence[dict],
    *,
    sample: int = 40,
    seed: int = 0,
) -> List[Question]:
    """
    Generate an evaluation set from the corpus itself.

    Samples articles rather than using all of them, so the run stays fast on a
    large collection; the seed makes it reproducible.
    """
    rng = random.Random(seed)
    n = len(texts)
    if n == 0:
        return []

    token_lists = [tokenize(t) for t in texts]
    df = _document_frequencies(token_lists)
    vocabulary = set(df)

    questions: List[Question] = []

    # passage — always available, needs no metadata
    for idx in rng.sample(range(n), min(sample, n)):
        question = _passage_question(rng, texts, idx)
        if question:
            questions.append(question)

    articles = _articles(metadata, n)
    if articles:
        titles = rng.sample(sorted(articles), min(sample, len(articles)))
        for title in titles:
            chunk_ids = articles[title]
            if tokenize(title):
                questions.append(
                    Question(family="title", query=title, gold=set(chunk_ids), note=title)
                )
            question = _signature_question(
                rng, token_lists, df, n, title, chunk_ids
            )
            if question:
                questions.append(question)
    else:
        # No article metadata: fall back to per-chunk signatures.
        for idx in rng.sample(range(n), min(sample, n)):
            question = _signature_question(
                rng, token_lists, df, n, f"chunk {idx}", [idx]
            )
            if question:
                questions.append(question)

    questions.extend(_nonsense_questions(rng, vocabulary, max(6, sample // 4)))
    return questions


# ---------------------------------------------------------------------- runner


def _text_to_indices(texts: Sequence[str]) -> Dict[str, set]:
    """
    Map chunk text back to every index holding it.

    Retrieval returns chunk *text* rather than indices, so scoring has to map
    back. Duplicate chunks share an entry, and a hit counts if any of them is
    gold. Phase 3 returns indices directly and this goes away.
    """
    lookup: Dict[str, set] = defaultdict(set)
    for idx, text in enumerate(texts):
        lookup[text].add(idx)
    return lookup


def run_eval(
    *,
    db,
    bm25,
    embedder,
    top_k: int = 5,
    sample: int = 40,
    seed: int = 0,
    questions: Optional[List[Question]] = None,
) -> dict:
    """
    Run the evaluation against the real retrieval pipeline.

    Caching is bypassed so the numbers describe retrieval, not cache hits, and
    web search is disabled so an offline corpus is measured offline.
    """
    from api.retrieval import retrieval_settings, retrieve

    texts = list(getattr(db, "texts", []) or (bm25.texts if bm25 else []))
    metadata = list(getattr(db, "metadata", []) or [])
    if not texts:
        raise ValueError("No chunks loaded. Load a collection before evaluating.")

    if questions is None:
        questions = build_questions(texts, metadata, sample=sample, seed=seed)
    if not questions:
        raise ValueError("Could not generate any evaluation questions from this corpus.")

    lookup = _text_to_indices(texts)
    settings = retrieval_settings()
    results = {family: FamilyResult(family) for family in FAMILIES}
    latencies: List[float] = []

    for question in questions:
        started = time.perf_counter()
        chunks, _mode = retrieve(
            question.query,
            top_k,
            db=db,
            bm25=bm25,
            embedder=embedder,
            cache=None,
            settings=settings,
            allow_web=False,
        )
        latencies.append((time.perf_counter() - started) * 1000)

        bucket = results[question.family]
        bucket.n += 1

        if not chunks:
            bucket.empty += 1

        if question.family == "nonsense":
            continue  # scored by abstention alone

        rank_of_hit = None
        for rank, chunk in enumerate(chunks, start=1):
            if lookup.get(chunk, set()) & question.gold:
                rank_of_hit = rank
                break

        if rank_of_hit:
            bucket.hits += 1
            bucket.reciprocal_ranks.append(1.0 / rank_of_hit)
            if rank_of_hit == 1:
                bucket.top1 += 1

    return {
        "corpus_chunks": len(texts),
        "articles": len(_articles(metadata, len(texts))),
        "top_k": top_k,
        "seed": seed,
        "sample": sample,
        "settings": settings,
        "families": {
            family: {
                "n": r.n,
                "recall_at_k": round(r.recall, 4),
                "precision_at_1": round(r.precision_at_1, 4),
                "mrr": round(r.mrr, 4),
                "abstention": round(r.abstention, 4),
            }
            for family, r in results.items()
            if r.n
        },
        "latency_ms": _latency_summary(latencies),
        "stage_latency_ms": measure_stages(
            db=db, bm25=bm25, embedder=embedder, questions=questions, top_k=top_k
        ),
    }


def _latency_summary(ordered_ms: List[float]) -> dict:
    """
    Latency percentiles, with the first query reported separately.

    The first query pays for lazily-initialised model state — consistently
    ~10x the steady-state cost here — and folding it into the percentiles
    makes a baseline that moves with unrelated noise.
    """
    if not ordered_ms:
        return {}

    cold = round(ordered_ms[0], 2)
    warm = sorted(ordered_ms[1:]) or sorted(ordered_ms)

    def pct(p):
        return round(warm[min(len(warm) - 1, int(len(warm) * p))], 2)

    return {"p50": pct(0.50), "p90": pct(0.90), "max": round(warm[-1], 2), "cold_start": cold}


def measure_stages(*, db, bm25, embedder, questions, top_k: int, limit: int = 20) -> dict:
    """
    Time each retrieval stage separately.

    Measured here rather than by instrumenting the request path, so the hot
    path carries no timing overhead in production.
    """
    from api.hybrid_search import hybrid_search

    probes = [q.query for q in questions if q.family != "nonsense"][:limit]
    if not probes:
        return {}

    def timed(fn):
        samples = []
        for query in probes:
            started = time.perf_counter()
            fn(query)
            samples.append((time.perf_counter() - started) * 1000)
        samples.sort()
        return round(samples[len(samples) // 2], 3)

    stages = {"embed": timed(lambda q: embedder.encode([q]))}

    vectors = {q: embedder.encode([q])[0] for q in probes}

    if bm25 is not None:
        stages["bm25"] = timed(lambda q: bm25.search_indices(q, top_k * 3))
    if db is not None:
        stages["faiss"] = timed(lambda q: db.search_indices(vectors[q], top_k * 3))

    stages["fuse"] = timed(
        lambda q: hybrid_search(
            query=q,
            query_embedding=vectors[q],
            vectordb=db,
            bm25_index=bm25,
            top_k=top_k,
            search_mode="hybrid",
        )
    )
    return stages


# ---------------------------------------------------------------------- render


_FAMILY_BLURB = {
    "passage": "a span copied out of one chunk; that chunk is the answer",
    "title": "an article's own title; its chunks are the answer",
    "signature": "an article's rarest terms; its chunks are the answer (upper bound)",
    "nonsense": "words absent from the corpus; abstaining is the answer",
}


def render(report: dict) -> str:
    """Human-readable summary of a report from run_eval."""
    lines = []
    lines.append(
        f"{report['corpus_chunks']:,} chunks | {report['articles']:,} articles | "
        f"top_k={report['top_k']} | seed={report['seed']}"
    )
    lines.append("")
    lines.append(f"{'family':<11}{'n':>5}{'recall@k':>11}{'P@1':>8}{'MRR':>8}{'abstain':>10}  what it measures")

    for family in FAMILIES:
        stats = report["families"].get(family)
        if not stats:
            continue
        if family == "nonsense":
            lines.append(
                f"{family:<11}{stats['n']:>5}{'-':>11}{'-':>8}{'-':>8}"
                f"{stats['abstention']:>9.1%}  {_FAMILY_BLURB[family]}"
            )
        else:
            lines.append(
                f"{family:<11}{stats['n']:>5}{stats['recall_at_k']:>10.1%}"
                f"{stats['precision_at_1']:>8.1%}{stats['mrr']:>8.3f}{stats['abstention']:>9.1%}"
                f"  {_FAMILY_BLURB[family]}"
            )

    latency = report.get("latency_ms") or {}
    if latency:
        lines.append("")
        lines.append(
            f"query latency   p50 {latency['p50']}ms | p90 {latency['p90']}ms | "
            f"max {latency['max']}ms | first call {latency['cold_start']}ms"
        )

    stages = report.get("stage_latency_ms") or {}
    if stages:
        lines.append(
            "per stage (p50) " + " | ".join(f"{k} {v}ms" for k, v in stages.items())
        )

    nonsense = report["families"].get("nonsense")
    if nonsense and nonsense["abstention"] == 0.0:
        lines.append("")
        lines.append(
            "! Abstention is 0%: the pipeline returns chunks for questions whose words\n"
            "  appear nowhere in the corpus. It cannot signal that it does not know."
        )

    return "\n".join(lines)


def save_baseline(report: dict, path: str) -> str:
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path
