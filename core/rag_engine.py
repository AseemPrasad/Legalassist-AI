import hashlib
import logging
import re
from typing import Dict, List, Optional
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import Config

LOGGER = logging.getLogger(__name__)


def get_judgment_hash(text: str) -> str:
    """Generate an MD5 hash of judgment text to uniquely identify it."""
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class LegalRAG:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG engine with a specific embedding model."""
        LOGGER.info(f"Initializing LegalRAG with embedding model: {embedding_model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        except Exception as e:
            LOGGER.error(f"Failed to load embedding model: {e}")
            raise
            
        self.vector_store = None
        self._stored_text = ""
        self.section_header_pattern = re.compile(
            r"^(section\s+\d+[\w().:-]*|article\s+\d+[\w().:-]*|chapter\s+\d+[\w().:-]*|clause\s+\d+[\w().:-]*)",
            re.IGNORECASE,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1400,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

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
                        for i in range(0, len(sub_chunk), max_hard_slice_size):
                            hard_slice = sub_chunk[i:i + max_hard_slice_size]
                            if hard_slice.strip():
                                semantic_chunks.append(hard_slice)

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

            self._stored_text = text

            chroma_client = chromadb.EphemeralClient()
            self.vector_store = Chroma.from_texts(
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

    def query(self, question: str, language: str, openai_client, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Query the document and generate an answer using the provided LLM client.
        Supports chat history for context-aware follow-up questions.
        """
        if not self.vector_store:
            return "Please wait for the document to finish processing before asking questions."
            
        try:
            LOGGER.info(f"Retrieving context for question: {question}")
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            relevant_docs = retriever.invoke(question)

            if not relevant_docs:
                LOGGER.info("Semantic search returned no results, trying keyword fallback")
                from core.app_utils import extract_text_from_pdf
                keyword_results = self._keyword_fallback_search(question)
                if keyword_results:
                    context = "\n\n---\n\n".join(keyword_results)
                    LOGGER.info(f"Keyword fallback found {len(keyword_results)} results")
                else:
                    return "I couldn't find relevant information in the document to answer your question."
            else:
                context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            
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
