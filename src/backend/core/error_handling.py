"""Error handling utilities for FastAPI routers."""

import logging
import uuid
from functools import wraps
from typing import Callable, TypeVar

from fastapi import HTTPException

T = TypeVar("T")

logger = logging.getLogger(__name__)


def handle_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that catches exceptions and converts them to HTTPException.

    Preserves HTTPExceptions (for auth, validation, etc.) but wraps unexpected exceptions
    with a sanitized error message and correlation ID. Full tracebacks are logged server-side.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise FastAPI HTTPExceptions unchanged (e.g., 404, 401, etc.)
            raise
        except Exception as e:
            request_id = str(uuid.uuid4())
            logger.error(
                "Unhandled error [request_id=%s]: %s",
                request_id,
                e,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal server error", "request_id": request_id},
            )

    return wrapper
