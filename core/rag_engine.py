import hashlib
import logging
from typing import List, Optional, Dict
import chromadb

LOGGER = logging.getLogger(__name__)


class _LazyImports:
    """Deferred imports for optional embedding dependencies."""
    HuggingFaceEmbeddings = None
    RecursiveCharacterTextSplitter = None
    Chroma = None

    @classmethod
    def load(cls) -> None:
        if cls.HuggingFaceEmbeddings is not None:
            return
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings as HFE
            from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTS
            from langchain_community.vectorstores import Chroma as Ch
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-community' and 'langchain' packages are required for RAG. "
                "Install them with: pip install langchain langchain-community"
            ) from exc
        cls.HuggingFaceEmbeddings = HFE
        cls.RecursiveCharacterTextSplitter = RCTS
        cls.Chroma = Ch


class LegalRAG:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG engine with a specific embedding model."""
        LOGGER.info(f"Initializing LegalRAG with embedding model: {embedding_model_name}")
        _LazyImports.load()
        try:
            self.embeddings = _LazyImports.HuggingFaceEmbeddings(model_name=embedding_model_name)
        except Exception as e:
            LOGGER.error(f"Failed to load embedding model: {e}")
            raise

        self.vector_store = None
        self.text_splitter = _LazyImports.RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        # Hard-cap slice width used as the final fallback when text_splitter
        # cannot reduce a piece below MAX_CHUNK_SIZE (e.g. base64 blobs, tables
        # with no standard separators).
        self._HARD_CAP_SLICE = 1400
        self._MAX_CHUNK_SIZE = 1800

    def _is_section_header(self, line: str) -> bool:
        """Return True when a line looks like a legal section / article header."""
        stripped = line.strip()
        if not stripped or len(stripped) > 160:
            return False
        return bool(self.section_header_pattern.match(stripped))

    def _split_into_section_chunks(self, text: str) -> List[str]:
        """Chunk text on section headers then enforce a strict size upper-bound.

        Algorithm
        ---------
        1. Walk lines and collect runs between legal section headers into
           coarse section blocks.
        2. If no headers are found the text_splitter handles the whole doc.
        3. For each coarse block that still exceeds MAX_CHUNK_SIZE:
           a. Run text_splitter.split_text() as a semantic fallback.
           b. Any piece that *still* exceeds MAX_CHUNK_SIZE (e.g. base64
              blobs or tables without standard separators) is sliced in
              HARD_CAP_SLICE windows.  This guarantees O(n) termination
              and prevents infinite recursive fallback loops.
        """
        chunks: List[str] = []
        current_lines: List[str] = []

        for line in text.splitlines():
            if self._is_section_header(line) and current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    chunks.append(section_text)
                current_lines = [line.strip()]
                continue
            current_lines.append(line)

        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                chunks.append(section_text)

        # No section headers found — fall back to pure size-based splitting.
        if len(chunks) <= 1:
            return self.text_splitter.split_text(text)

        semantic_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self._MAX_CHUNK_SIZE:
                semantic_chunks.append(chunk)
            else:
                split_pieces = self.text_splitter.split_text(chunk)
                for piece in split_pieces:
                    if len(piece) > self._MAX_CHUNK_SIZE:
                        # Hard-cap: linear slicing guarantees termination.
                        # This handles unstructured blobs (base64, legal tables)
                        # that the recursive splitter cannot reduce further.
                        for i in range(0, len(piece), self._HARD_CAP_SLICE):
                            sliced = piece[i:i + self._HARD_CAP_SLICE]
                            if sliced.strip():
                                semantic_chunks.append(sliced)
                    else:
                        if piece.strip():
                            semantic_chunks.append(piece)

        return semantic_chunks if semantic_chunks else self.text_splitter.split_text(text)

    def reset(self) -> None:
        """Reset the vector store to clear loaded document state."""
        LOGGER.info("Resetting LegalRAG vector store.")
        self.vector_store = None

    def _is_section_header(self, line: str) -> bool:
        """Identify likely legal section headers so related rules stay together."""
        stripped_line = line.strip()
        if not stripped_line or len(stripped_line) > 160:
            return False

        return bool(self.section_header_pattern.match(stripped_line))

    def _split_into_section_chunks(self, text: str) -> List[str]:
        """Split judgment text on section-like headers before falling back to size-based chunking."""
        chunks: List[str] = []
        current_section_lines: List[str] = []
        max_chunk_size = 1400
        max_hard_slice_size = 1400

        for line in text.splitlines():
            if self._is_section_header(line) and current_section_lines:
                section_text = "\n".join(current_section_lines).strip()
                if section_text:
                    chunks.append(section_text)
                current_section_lines = [line.strip()]
                continue

            current_section_lines.append(line)

        if current_section_lines:
            section_text = "\n".join(current_section_lines).strip()
            if section_text:
                chunks.append(section_text)

        if len(chunks) <= 1:
            return self.text_splitter.split_text(text)

        semantic_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                semantic_chunks.append(chunk)
            else:
                split_result = self.text_splitter.split_text(chunk)
                for sub_chunk in split_result:
                    if len(sub_chunk) <= max_chunk_size:
                        semantic_chunks.append(sub_chunk)
                    else:
                        # Soft-split on word boundaries to avoid cutting mid-word
                        words = sub_chunk.split(" ")
                        curr_chunk = []
                        curr_size = 0
                        for w in words:
                            if curr_size + len(w) + 1 > max_hard_slice_size:
                                if curr_chunk:
                                    semantic_chunks.append(" ".join(curr_chunk))
                                curr_chunk = [w]
                                curr_size = len(w)
                            else:
                                curr_chunk.append(w)
                                curr_size += len(w) + 1
                        if curr_chunk:
                            semantic_chunks.append(" ".join(curr_chunk))

        return [chunk for chunk in semantic_chunks if chunk.strip()]

    def initialize_vector_store(self, text: str) -> bool:
        """Chunk the document and load it into an ephemeral vector store."""
        if not text or not text.strip():
            LOGGER.warning("Empty text provided to LegalRAG.")
            return False
            
        try:
            LOGGER.info("Chunking document text...")
            chunks = self._split_into_section_chunks(text)
            LOGGER.info(f"Split document into {len(chunks)} chunks.")
            
            # Create vector store in memory using EphemeralClient
            chroma_client = chromadb.EphemeralClient()
            self.vector_store = _LazyImports.Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                client=chroma_client,
                collection_name="judgment_chat"
            )
            LOGGER.info("Successfully initialized vector store.")
            return True
        except Exception as e:
            LOGGER.error(f"Error initializing vector store: {e}")
            return False

    def _keyword_fallback_search(self, question: str) -> List[str]:
        """Keyword-based fallback when semantic search returns no results."""
        if not self._stored_text:
            return []
        keywords = question.lower().split()
        lines = self._stored_text.split("\n")
        scored = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = sum(1 for kw in keywords if kw in line_lower and len(kw) > 2)
            if score > 0:
                scored.append((score, len(line), line))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [line for _, _, line in scored[:5]]

    def retrieve_with_scores(self, question: str, k: int = 5):
        """Retrieve top-k passages with their similarity scores and metadata."""
        if not self.vector_store:
            return []
        try:
            # Use underlying vector store similarity search with scores if available
            results = self.vector_store.similarity_search_with_score(question, k=k)
            # results is list of tuples (Document, score)
            retrieved = []
            for doc, score in results:
                meta = getattr(doc, "metadata", {}) or {}
                retrieved.append({
                    "content": getattr(doc, "page_content", str(doc)),
                    "score": float(score),
                    "metadata": meta,
                })
            return retrieved
        except Exception as e:
            LOGGER.debug(f"similarity_search_with_score failed: {e}")
            # Fall back to retriever without scores
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(question)
            return [{
                "content": d.page_content,
                "score": 0.0,
                "metadata": getattr(d, "metadata", {}) or {},
            } for d in docs]

    def query(self, question: str, language: str, openai_client, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Query the document and generate an answer using the provided LLM client.
        Supports chat history for context-aware follow-up questions.
        """
        if not self.vector_store:
            return "Please wait for the document to finish processing before asking questions."
            
        try:
            LOGGER.info(f"Retrieving context for question: {question}")
            # Retrieve passages with scores and metadata
            retrieved = self.retrieve_with_scores(question, k=5)

            if not retrieved:
                LOGGER.info("Semantic search returned no results, trying keyword fallback")
                keyword_results = self._keyword_fallback_search(question)
                if keyword_results:
                    context = "\n\n---\n\n".join(keyword_results)
                    citations = []
                    LOGGER.info(f"Keyword fallback found {len(keyword_results)} results")
                else:
                    return "I couldn't find relevant information in the document to answer your question."
            else:
                # Compute normalized confidence scores
                scores = [r.get("score", 0.0) for r in retrieved]
                max_score = max(scores) if scores else 0.0
                # Normalize if scores are cosine similarities in [-1,1]
                if max_score <= 1.0 and max_score >= -1.0:
                    # map to 0..1 if needed
                    norm_scores = [max(0.0, min(1.0, (s + 1) / 2 if s < 0.0 else s)) for s in scores]
                else:
                    # assume already 0..1
                    norm_scores = [max(0.0, min(1.0, float(s))) for s in scores]

                confidence = max(norm_scores) if norm_scores else 0.0

                # If confidence is below threshold, return insufficient evidence
                threshold = float(getattr(Config, "RAG_CONFIDENCE_THRESHOLD", 0.2))
                LOGGER.info(f"Retrieval confidence: {confidence:.3f} (threshold {threshold})")
                if confidence < threshold:
                    return "I cannot find sufficient evidence in the document to answer that question."

                # Build context with citations
                parts = []
                citations = []
                for r, s in zip(retrieved, norm_scores):
                    meta = r.get("metadata", {})
                    citation = f"[source:{meta.get('source_hash','unknown')}#chunk:{meta.get('chunk_index',0)}]"
                    excerpt = meta.get("excerpt") or (r.get("content")[:240] + "...")
                    parts.append(f"{excerpt}\n\nCitation: {citation} (score={s:.3f})")
                    citations.append(citation)

                context = "\n\n---\n\n".join(parts)
            
            # Format chat history for the prompt
            history_str = ""
            if chat_history:
                # Only take the last 4-6 messages to keep the prompt size manageable
                recent_history = chat_history[-6:]
                history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])

            # Construct the prompt
            prompt = f"""
You are LegalEase AI, an expert judicial researcher. 
Your goal is to provide accurate, context-grounded answers to user questions about a specific legal document.

STRICT GUIDELINES:
1. Answer ONLY based on the provided CONTEXT. If the answer is not in the context, say "I cannot find the answer to this in the document."
2. CITATIONS: Whenever possible, quote specific sentences or phrases from the document to support your answer.
3. CONVERSATION: Use the RECENT CHAT HISTORY to understand follow-up questions (e.g., "What about the other person?").
4. LANGUAGE: Provide your final answer ONLY in the {language} language.

RECENT CHAT HISTORY:
{history_str}

CONEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

ANSWER IN {language} (include citations if possible):
"""
            
            LOGGER.info("Generating response from LLM...")
            # Call LLM using safe_llm_call for robust error handling and retries
            from core.app_utils import safe_llm_call
            answer, error = safe_llm_call(
                client=openai_client,
                model=Config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": f"You are a helpful legal researcher. Output only in {language}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.1,
            )
            
            if error:
                return f"AI Service Error: {error}"
            
            return answer
            
        except Exception as e:
            LOGGER.error(f"Error querying RAG engine: {e}")
            return f"An error occurred while trying to answer your question: {str(e)}"

    async def async_query(self, question: str, language: str, openai_client, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Async version of `query` that uses an async LLM helper to avoid blocking the event loop."""
        if not self.vector_store:
            return "Please wait for the document to finish processing before asking questions."

        try:
            LOGGER.info(f"(async) Retrieving context for question: {question}")
            retrieved = self.retrieve_with_scores(question, k=5)

            if not retrieved:
                LOGGER.info("Semantic search returned no results, trying keyword fallback")
                keyword_results = self._keyword_fallback_search(question)
                if keyword_results:
                    context = "\n\n---\n\n".join(keyword_results)
                    citations = []
                    LOGGER.info(f"Keyword fallback found {len(keyword_results)} results")
                else:
                    return "I couldn't find relevant information in the document to answer your question."
            else:
                scores = [r.get("score", 0.0) for r in retrieved]
                max_score = max(scores) if scores else 0.0
                if max_score <= 1.0 and max_score >= -1.0:
                    norm_scores = [max(0.0, min(1.0, (s + 1) / 2 if s < 0.0 else s)) for s in scores]
                else:
                    norm_scores = [max(0.0, min(1.0, float(s))) for s in scores]

                confidence = max(norm_scores) if norm_scores else 0.0
                threshold = float(getattr(Config, "RAG_CONFIDENCE_THRESHOLD", 0.2))
                LOGGER.info(f"Retrieval confidence: {confidence:.3f} (threshold {threshold})")
                if confidence < threshold:
                    return "I cannot find sufficient evidence in the document to answer that question."

                parts = []
                citations = []
                for r, s in zip(retrieved, norm_scores):
                    meta = r.get("metadata", {})
                    citation = f"[source:{meta.get('source_hash','unknown')}#chunk:{meta.get('chunk_index',0)}]"
                    excerpt = meta.get("excerpt") or (r.get("content")[:240] + "...")
                    parts.append(f"{excerpt}\n\nCitation: {citation} (score={s:.3f})")
                    citations.append(citation)

                context = "\n\n---\n\n".join(parts)

            history_str = ""
            if chat_history:
                recent_history = chat_history[-6:]
                history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])

            prompt = f"""
You are LegalEase AI, an expert judicial researcher. 
Your goal is to provide accurate, context-grounded answers to user questions about a specific legal document.

STRICT GUIDELINES:
1. Answer ONLY based on the provided CONTEXT. If the answer is not in the context, say "I cannot find the answer to this in the document."
2. CITATIONS: Whenever possible, quote specific sentences or phrases from the document to support your answer.
3. CONVERSATION: Use the RECENT CHAT HISTORY to understand follow-up questions (e.g., "What about the other person?").
4. LANGUAGE: Provide your final answer ONLY in the {language} language.

RECENT CHAT HISTORY:
{history_str}

CONEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

ANSWER IN {language} (include citations if possible):
"""

            LOGGER.info("(async) Generating response from LLM...")
            from core.app_utils import safe_llm_call_async
            answer, error = await safe_llm_call_async(
                client=openai_client,
                model=Config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": f"You are a helpful legal researcher. Output only in {language}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.1,
            )

            if error:
                return f"AI Service Error: {error}"

            return answer

        except Exception as e:
            LOGGER.error(f"Error querying RAG engine (async): {e}")
            return f"An error occurred while trying to answer your question: {str(e)}"
