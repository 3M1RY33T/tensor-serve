from tqdm import tqdm

from bm25_index import BM25Index
from chunker import chunk_text
from embedder import Embedder
from utils import clean_text, iterate_articles
from vectordb import VectorDB


def run_ingestion(zim_path: str, output_name="zim_db"):
    embedder = Embedder()
    db = None

    batch_texts = []
    batch_chunks = []

    for article in tqdm(iterate_articles(zim_path)):
        clean = clean_text(article["text"])
        chunks = chunk_text(clean)

        if not chunks:  # Skip empty articles
            continue

        batch_chunks.extend(chunks)
        batch_texts.extend(chunks)

        if len(batch_texts) >= 100:
            embeddings = embedder.encode(batch_texts)

            if db is None:
                db = VectorDB(dim=len(embeddings[0]))

            db.add(embeddings, batch_chunks)

            batch_texts = []
            batch_chunks = []

    # Process remaining batch
    if batch_texts:
        embeddings = embedder.encode(batch_texts)
        if db is None:
            db = VectorDB(dim=len(embeddings[0]))
        db.add(embeddings, batch_chunks)

    if db is None:
        raise ValueError("No valid content found in ZIM file")

    db.save(output_name)

    # Build and save BM25 keyword index alongside the FAISS index
    bm25 = BM25Index()
    bm25.build(db.texts)
    bm25.save(output_name)

    return {"status": "completed", "output": output_name}
