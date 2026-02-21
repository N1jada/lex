import logging

from qdrant_client import QdrantClient

from lex.settings import (
    QDRANT_API_KEY,
    QDRANT_CLOUD_API_KEY,
    QDRANT_CLOUD_URL,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    USE_CLOUD_QDRANT,
)

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """
    Returns a Qdrant client based on the configured settings.

    Uses cloud Qdrant if USE_CLOUD_QDRANT=true, otherwise uses local.

    Returns:
        QdrantClient: Configured Qdrant client
    """
    if USE_CLOUD_QDRANT:
        if not QDRANT_CLOUD_URL or not QDRANT_CLOUD_API_KEY:
            raise ValueError(
                "USE_CLOUD_QDRANT is enabled but QDRANT_CLOUD_URL or "
                "QDRANT_CLOUD_API_KEY environment variables are not set"
            )

        client = QdrantClient(
            url=QDRANT_CLOUD_URL,
            api_key=QDRANT_CLOUD_API_KEY,
            timeout=30,
        )
        logger.info(f"Connecting to Qdrant Cloud: {QDRANT_CLOUD_URL}")
    else:
        client = QdrantClient(
            url=QDRANT_HOST,
            port=QDRANT_GRPC_PORT,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
        logger.info(f"Connecting to local Qdrant: {QDRANT_HOST}")

    try:
        # Test connection
        collections = client.get_collections()
        mode = "Cloud" if USE_CLOUD_QDRANT else "Local"
        logger.info(
            f"Connected to Qdrant ({mode})",
            extra={
                "collections_count": len(collections.collections),
                "mode": mode,
            },
        )
        return client
    except Exception as e:
        mode = "Cloud" if USE_CLOUD_QDRANT else "Local"
        logger.error(f"Error connecting to Qdrant ({mode}): {e}")
        raise


class _LazyClient:
    """Lazy proxy that defers Qdrant client creation until first use."""

    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = get_qdrant_client()

    def __getattr__(self, name):
        self._ensure_client()
        return getattr(self._client, name)


# Global lazy client instance — compatible with `from lex.core.qdrant_client import qdrant_client`
qdrant_client = _LazyClient()
