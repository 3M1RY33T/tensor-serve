from tqdm import tqdm

from api.bm25_index import BM25Index
from api.chunker import chunk_text
from api.embedder import Embedder
from api.utils import clean_text, iterate_articles
from api.vectordb import VectorDB


def run_ingestion(zim_path: str, output_name="zim_db"):
    embedder = Embedder()
    db = None

    batch_texts = []
    batch_chunks = []
    batch_metadata = []

    for article in tqdm(iterate_articles(zim_path)):
        clean = clean_text(article["text"])
        chunks = chunk_text(clean)

        if not chunks:  # Skip empty articles
            continue

        batch_chunks.extend(chunks)
        batch_texts.extend(chunks)
        batch_metadata.extend(
            {
                "zim_title": article.get("zim_title"),
                "zim_path": article.get("zim_path") or zim_path,
                "article_title": article.get("title"),
            }
            for _ in chunks
        )

        if len(batch_texts) >= 100:
            embeddings = embedder.encode(batch_texts)

            if db is None:
                db = VectorDB(dim=len(embeddings[0]))

            db.add(embeddings, batch_chunks, batch_metadata)

            batch_texts = []
            batch_chunks = []
            batch_metadata = []

    # Process remaining batch
    if batch_texts:
        embeddings = embedder.encode(batch_texts)
        if db is None:
            db = VectorDB(dim=len(embeddings[0]))
        db.add(embeddings, batch_chunks, batch_metadata)

    if db is None:
        raise ValueError("No valid content found in ZIM file")

    db.save(output_name)

    # Build and save BM25 keyword index alongside the FAISS index
    bm25 = BM25Index()
    bm25.build(db.texts)
    bm25.save(output_name)

    return {"status": "completed", "output": output_name}
