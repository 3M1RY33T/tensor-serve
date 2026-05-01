from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as download_error:
                raise RuntimeError(
                    f"Could not load embedding model '{model_name}' from the local cache "
                    "or download it from Hugging Face."
                ) from download_error

    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
