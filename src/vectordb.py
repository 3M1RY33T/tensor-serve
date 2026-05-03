import os
import pickle

import faiss
import numpy as np


class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadata = []
        self.dim = dim

    def add(self, embeddings, chunks, metadata=None):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(chunks)
        if metadata is None:
            metadata = [{} for _ in chunks]
        self.metadata.extend(metadata)

    def save(self, path="db"):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

    def load(self, path="db"):
        index_path = f"{path}.index"
        pkl_path = f"{path}.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector index not found: {index_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Text data not found: {pkl_path}")

        self.index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self.texts = data.get("texts", [])
            self.metadata = data.get("metadata", [{} for _ in self.texts])
        else:
            self.texts = data
            self.metadata = [{} for _ in self.texts]

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results

    def search_indices(self, query_embedding, top_k=5):
        """Return chunk indices ranked by FAISS similarity (best first)."""
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        return [int(idx) for idx in indices[0] if 0 <= idx < len(self.texts)]
