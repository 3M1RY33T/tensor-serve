from tqdm import tqdm

from api.bm25_index import BM25Index
from api.chunker import chunk_text
from api.embedder import Embedder
from api.utils import clean_text, iterate_articles
from api.durability import BuildLock
from api.vectordb import VectorDB, save_collection

# Chunks accumulated before each encode call. Larger batches let the encoder
# group similar lengths together, which costs less padding.
EMBED_BATCH_CHUNKS = 512


def _run_ingestion(zim_path: str, output_name="zim_db"):
    embedder = Embedder()
    db = None

    batch_chunks = []
    batch_metadata = []

    for article in tqdm(iterate_articles(zim_path)):
        clean = clean_text(article["text"])
        # Size chunks to what the embedding model can actually read, so no chunk
        # is indexed by its opening fragment alone.
        chunks = chunk_text(
            clean,
            overlap=30,
            tokenizer=embedder.tokenizer,
            max_tokens=embedder.max_tokens,
        )

        if not chunks:  # Skip empty articles
            continue

        batch_chunks.extend(chunks)
        batch_metadata.extend(
            {
                "zim_title": article.get("zim_title"),
                "zim_path": article.get("zim_path") or zim_path,
                "article_title": article.get("title"),
            }
            for _ in chunks
        )

        if len(batch_chunks) >= EMBED_BATCH_CHUNKS:
            embeddings = embedder.encode(batch_chunks)

            if db is None:
                db = VectorDB(dim=len(embeddings[0]))

            db.add(embeddings, batch_chunks, batch_metadata)

            batch_chunks = []
            batch_metadata = []

    # Process remaining batch
    if batch_chunks:
        embeddings = embedder.encode(batch_chunks)
        if db is None:
            db = VectorDB(dim=len(embeddings[0]))
        db.add(embeddings, batch_chunks, batch_metadata)

    if db is None:
        raise ValueError("No valid content found in ZIM file")

    # Build the keyword index before anything is written, so the collection
    # lands as one consistent set of files.
    bm25 = BM25Index()
    bm25.build(db.texts)
    save_collection(output_name, db, bm25)

    return {"status": "completed", "output": output_name}


def run_ingestion(zim_path: str, output_name="zim_db"):
    """
    Build a collection, refusing to interleave with another build of the same
    one. Two concurrent ingests would leave a vector index and a keyword index
    that disagree about what a given chunk is.
    """
    with BuildLock(output_name, purpose="ingest"):
        return _run_ingestion(zim_path, output_name)
