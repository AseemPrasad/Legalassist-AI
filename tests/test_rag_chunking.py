from core.rag_engine import LegalRAG


def test_rag_chunking_word_boundary_safety():
    # Construct a dummy RAG instance (we mock dependencies or use default)
    # To avoid loading HuggingFace embeddings in test collection (which requires network/weights),
    # we can test the internal `_split_into_section_chunks` directly since it's pure Python.
    rag = LegalRAG.__new__(LegalRAG)
    rag.section_header_pattern = LegalRAG._is_section_header  # not strictly needed
    import re
    rag.section_header_pattern = re.compile(
        r"^(section\s+\d+[\w().:-]*|article\s+\d+[\w().:-]*|chapter\s+\d+[\w().:-]*|clause\s+\d+[\w().:-]*)",
        re.IGNORECASE,
    )
    
    # Custom mock text splitter that returns a large separator-free block
    class MockSplitter:
        def split_text(self, text):
            return [text]
            
    rag.text_splitter = MockSplitter()
    
    # Input has a long string of words
    long_text = " ".join(["word" for _ in range(500)]) # ~2000 chars
    
    # We execute split
    chunks = rag._split_into_section_chunks(long_text)
    
    assert len(chunks) > 1
    # Verify no mid-word slicing has happened (all chunks should consist of whole "word" units)
    for c in chunks:
        words = c.split(" ")
        assert all(w == "word" for w in words)
