import faiss
import numpy as np
import pickle
import os

class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.dim = dim

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(chunks)

    def save(self, path="db"):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path="db"):
        index_path = f"{path}.index"
        pkl_path = f"{path}.pkl"
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector index not found: {index_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Text data not found: {pkl_path}")
        
        self.index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            self.texts = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results