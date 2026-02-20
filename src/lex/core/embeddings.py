import logging
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import tiktoken
from fastembed import SparseTextEmbedding
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, OpenAI, RateLimitError
from qdrant_client.models import SparseVector

from lex.settings import EMBEDDING_DEPLOYMENT, EMBEDDING_DIMENSIONS, USE_AZURE_OPENAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client (Azure or standard)
_openai_client: Union[AzureOpenAI, OpenAI, None] = None
_openai_client_lock = threading.Lock()

# Initialize FastEmbed BM25 model (lazy loading)
_sparse_model = None
_sparse_model_lock = threading.Lock()

# Initialize tiktoken encoder (lazy loading)
_tokenizer: tiktoken.Encoding | None = None
_tokenizer_lock = threading.Lock()

# Chunking config
MAX_EMBED_TOKENS = 8000  # Stay below OpenAI's 8192 limit
CHUNK_TOKENS = 4000
CHUNK_OVERLAP_TOKENS = 200

# Rate limiting config
MAX_RETRIES = 10
BASE_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 120.0  # Cap backoff at 2 minutes

# Parallelism config - keep low to avoid rate limits
DEFAULT_MAX_WORKERS = int(os.environ.get("EMBEDDING_MAX_WORKERS", "5"))


def get_openai_client() -> Union[AzureOpenAI, OpenAI]:
    """Lazy load OpenAI client — Azure or standard (thread-safe)."""
    global _openai_client
    if _openai_client is None:
        with _openai_client_lock:
            # Double-check after acquiring lock
            if _openai_client is None:
                if USE_AZURE_OPENAI:
                    logger.info("Initializing Azure OpenAI client...")
                    _openai_client = AzureOpenAI(
                        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                        api_version="2024-02-01",
                        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                        max_retries=0,  # We handle retries manually
                        timeout=60.0,
                    )
                    logger.info("Azure OpenAI client initialized")
                else:
                    logger.info("Initializing standard OpenAI client...")
                    _openai_client = OpenAI(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        max_retries=0,  # We handle retries manually
                        timeout=60.0,
                    )
                    logger.info("Standard OpenAI client initialized")
    return _openai_client


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy load sparse model to avoid initialization on import (thread-safe)."""
    global _sparse_model
    if _sparse_model is None:
        with _sparse_model_lock:
            # Double-check after acquiring lock
            if _sparse_model is None:
                logger.info("Initializing FastEmbed BM25 model...")
                _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                logger.info("FastEmbed BM25 model initialized")
    return _sparse_model


def get_tokenizer() -> tiktoken.Encoding:
    """Lazy load tiktoken cl100k_base encoder (thread-safe)."""
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """Split text into overlapping chunks based on token count.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    step = max_tokens - overlap_tokens

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start += step

    return chunks


def _embed_single_text(text: str, max_retries: int = MAX_RETRIES) -> List[float]:
    """Embed a single text that fits within the token limit. No chunking."""
    client = get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT, input=text, dimensions=EMBEDDING_DIMENSIONS
            )
            return response.data[0].embedding

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            # Transient errors - retry with exponential backoff + jitter
            if attempt == max_retries - 1:
                logger.error(f"Failed to generate dense embedding after {max_retries} retries: {e}")
                raise

            # Exponential backoff with jitter and cap
            backoff = min(BASE_BACKOFF * (2**attempt), MAX_BACKOFF)
            jitter = random.uniform(0, backoff * 0.1)  # Add up to 10% jitter
            sleep_time = backoff + jitter
            error_type = type(e).__name__
            logger.warning(
                f"{error_type}: {e}, retrying in {sleep_time:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(sleep_time)

        except Exception as e:
            # Non-transient errors - fail immediately
            logger.error(f"Non-retryable error generating embedding: {type(e).__name__}: {e}")
            raise

    raise Exception(f"Failed to generate embedding after {max_retries} retries")


def _chunk_and_embed(text: str, max_retries: int = MAX_RETRIES) -> List[float]:
    """Chunk a long text, embed each chunk, and return the averaged + L2-normalised vector."""
    chunks = chunk_text_by_tokens(text)
    logger.info(f"Chunking long text ({len(text)} chars) into {len(chunks)} chunks for embedding")

    # Embed each chunk
    vectors = [_embed_single_text(chunk, max_retries=max_retries) for chunk in chunks]

    # Average element-wise
    dim = len(vectors[0])
    avg = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

    # L2-normalise
    norm = math.sqrt(sum(x * x for x in avg))
    if norm > 0:
        avg = [x / norm for x in avg]

    return avg


def generate_dense_embedding_with_retry(text: str, max_retries: int = MAX_RETRIES) -> List[float]:
    """
    Generate dense embedding using OpenAI with retry logic for rate limits.

    For texts exceeding the token limit, automatically chunks, embeds each chunk
    separately, then averages and L2-normalises the result.

    Args:
        text: Text to embed
        max_retries: Maximum number of retry attempts

    Returns:
        1024-dimensional vector

    Raises:
        Exception: If embedding generation fails after all retries
    """
    tokenizer = get_tokenizer()
    token_count = len(tokenizer.encode(text))

    if token_count > MAX_EMBED_TOKENS:
        return _chunk_and_embed(text, max_retries=max_retries)

    return _embed_single_text(text, max_retries=max_retries)


def generate_dense_embedding(text: str) -> List[float]:
    """Generate dense embedding (use generate_dense_embeddings_batch for parallel processing).

    Args:
        text: Text to embed

    Returns:
        1024-dimensional vector
    """
    return generate_dense_embedding_with_retry(text)


def generate_dense_embeddings_batch(
    texts: List[str], max_workers: int | None = None, progress_callback=None
) -> List[List[float]]:
    """Generate dense embeddings for multiple texts in parallel with rate limit handling.

    Args:
        texts: List of texts to embed
        max_workers: Number of concurrent workers (default from EMBEDDING_MAX_WORKERS env or 5)
        progress_callback: Optional callback function(completed_count) for progress updates

    Returns:
        List of 1024-dimensional vectors in same order as input texts
    """
    if not texts:
        return []

    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS

    results: List[Optional[List[float]]] = [None] * len(texts)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(generate_dense_embedding_with_retry, text): idx
            for idx, text in enumerate(texts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                completed += 1

                if progress_callback and completed % 10 == 0:
                    progress_callback(completed)

            except Exception as e:
                logger.error(f"Failed to generate embedding for text {idx}: {e}")
                results[idx] = [0.0] * EMBEDDING_DIMENSIONS

    # Type checker: results is guaranteed to have no None values due to exception handling
    return results  # type: ignore[return-value]


def generate_sparse_embedding(text: str) -> SparseVector:
    """
    Generate sparse BM25 embedding using FastEmbed.

    Args:
        text: Text to embed

    Returns:
        SparseVector with indices and values arrays

    Example:
        SparseVector(indices=[12, 45, 234], values=[0.8, 0.6, 0.4])
    """
    try:
        model = get_sparse_model()
        embeddings = list(model.embed([text]))

        if not embeddings:
            logger.warning("FastEmbed returned no embeddings")
            return SparseVector(indices=[], values=[])

        embedding = embeddings[0]

        # Convert to SparseVector format Qdrant expects
        return SparseVector(
            indices=[int(idx) for idx in embedding.indices],
            values=[float(val) for val in embedding.values],
        )
    except Exception as e:
        logger.error(f"Failed to generate sparse embedding: {e}")
        return SparseVector(indices=[], values=[])


def generate_sparse_embeddings_batch(texts: List[str]) -> List[SparseVector]:
    """
    Generate sparse BM25 embeddings for multiple texts efficiently.

    Args:
        texts: List of texts to embed

    Returns:
        List of SparseVectors in same order as input texts
    """
    if not texts:
        return []

    try:
        model = get_sparse_model()
        embeddings = list(model.embed(texts))

        return [
            SparseVector(
                indices=[int(idx) for idx in emb.indices], values=[float(val) for val in emb.values]
            )
            for emb in embeddings
        ]
    except Exception as e:
        logger.error(f"Failed to generate sparse embeddings batch: {e}")
        return [SparseVector(indices=[], values=[]) for _ in texts]


def generate_hybrid_embeddings(text: str) -> Tuple[List[float], SparseVector]:
    """Generate both dense and sparse embeddings for hybrid search.

    Args:
        text: Text to embed

    Returns:
        Tuple of (dense_vector, sparse_vector)

    Example:
        ([0.1, 0.2, ...], SparseVector(indices=[12, 45], values=[0.8, 0.6]))
    """
    dense = generate_dense_embedding(text)
    sparse = generate_sparse_embedding(text)
    return dense, sparse


def generate_hybrid_embeddings_batch(
    texts: List[str], max_workers: int | None = None, progress_callback=None
) -> List[Tuple[List[float], SparseVector]]:
    """
    Generate hybrid embeddings for multiple texts in parallel.

    Args:
        texts: List of texts to embed
        max_workers: Number of concurrent workers (default from EMBEDDING_MAX_WORKERS env or 5)
        progress_callback: Optional callback for progress updates

    Returns:
        List of (dense_vector, sparse_vector) tuples in same order as input
    """
    if not texts:
        return []

    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS

    dense_embeddings = generate_dense_embeddings_batch(
        texts, max_workers=max_workers, progress_callback=progress_callback
    )
    sparse_embeddings = generate_sparse_embeddings_batch(texts)

    return list(zip(dense_embeddings, sparse_embeddings))
