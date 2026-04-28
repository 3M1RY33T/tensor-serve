from typing import List

from tqdm import tqdm

from bm25_index import BM25Index
from chunker import chunk_text
from embedder import Embedder
from utils import clean_text, iterate_articles
from vectordb import VectorDB


def run_multi_ingest(zim_paths: List[str], output_name="combined_db"):
    """
    Ingest multiple ZIM files into a single vector database.

    Args:
        zim_paths: List of paths to ZIM files
        output_name: Name of the output database

    Returns:
        Dictionary with ingestion status
    """
    embedder = Embedder()
    db = None
    total_articles = 0
    total_chunks = 0

    batch_texts = []
    batch_chunks = []

    # Process each ZIM file
    for zim_path in zim_paths:
        print(f"\nProcessing: {zim_path}")
        try:
            for article in tqdm(iterate_articles(zim_path), desc=zim_path):
                clean = clean_text(article["text"])
                chunks = chunk_text(clean)

                if not chunks:
                    continue

                batch_chunks.extend(chunks)
                batch_texts.extend(chunks)
                total_articles += 1
                total_chunks += len(chunks)

                # Process batch when threshold reached
                if len(batch_texts) >= 100:
                    embeddings = embedder.encode(batch_texts)

                    if db is None:
                        db = VectorDB(dim=len(embeddings[0]))

                    db.add(embeddings, batch_chunks)

                    batch_texts = []
                    batch_chunks = []

        except Exception as e:
            print(f"Error processing {zim_path}: {str(e)}")
            continue

    # Process remaining batch
    if batch_texts:
        embeddings = embedder.encode(batch_texts)
        if db is None:
            db = VectorDB(dim=len(embeddings[0]))
        db.add(embeddings, batch_chunks)

    if db is None:
        raise ValueError("No valid content found in any ZIM files")

    db.save(output_name)

    # Build and save BM25 keyword index alongside the FAISS index
    bm25 = BM25Index()
    bm25.build(db.texts)
    bm25.save(output_name)

    return {
        "status": "completed",
        "output": output_name,
        "files_processed": len(zim_paths),
        "total_articles": total_articles,
        "total_chunks": total_chunks,
    }
