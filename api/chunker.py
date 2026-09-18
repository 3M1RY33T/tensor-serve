"""
Chunking for ingestion.

Word-count chunking alone cannot guarantee a chunk fits the embedding model:
500 words tokenises to ~557 word-pieces, and all-MiniLM-L6-v2 accepts 256, so
54% of every chunk was silently dropped before it reached its own vector.

When a tokenizer is supplied, chunks are packed by real token count against the
model's limit, so no chunk is ever truncated and chunks stay as large as the
model can actually take. The word-count path remains the default for callers
that have no tokenizer.
"""


def chunk_text(text, chunk_size=500, overlap=50, tokenizer=None, max_tokens=None):
    """
    Split text into overlapping chunks.

    Args:
        text:       Source text.
        chunk_size: Words per chunk, used when no tokenizer is given.
        overlap:    Words (or tokens) shared between consecutive chunks.
        tokenizer:  Optional HuggingFace fast tokenizer. When supplied, chunks
                    are packed to max_tokens real tokens instead of words.
        max_tokens: Token budget per chunk. Required with tokenizer.

    Returns:
        List of chunk strings, sliced verbatim from the source text.
    """
    if tokenizer is not None and max_tokens:
        return _chunk_by_tokens(text, tokenizer, max_tokens, overlap)

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(" ".join(chunk))

    return chunks


def _chunk_by_tokens(text, tokenizer, max_tokens, overlap):
    """
    Pack chunks to a real token budget.

    Uses the tokenizer's offset mapping to slice the *original* string, so chunk
    text stays verbatim rather than being round-tripped through decode().
    """
    if not text.strip():
        return []

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
    except Exception:
        # Slow tokenizer, or one without offset mapping — fall back to words.
        return chunk_text(text, chunk_size=max_tokens // 2, overlap=overlap)

    # Drop zero-width tokens, which carry no source span to slice.
    offsets = [(s, e) for s, e in offsets if e > s]
    if not offsets:
        return []

    step = max(1, max_tokens - overlap)
    chunks = []

    for start in range(0, len(offsets), step):
        window = offsets[start:start + max_tokens]
        if not window:
            continue
        chunk = text[window[0][0]:window[-1][1]].strip()
        if chunk:
            chunks.append(chunk)

    return chunks
