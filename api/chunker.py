def chunk_text(text, chunk_size=500, overlap=50):
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