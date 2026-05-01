from src.chunker import chunk_text


def test_empty_chunk():
    assert chunk_text("") == []


def test_chunking_basic():
    text = " ".join(str(i) for i in range(30))
    chunks = chunk_text(text, chunk_size=10, overlap=2)

    assert len(chunks) == 4
    assert chunks[0].split()[0] == "0"
