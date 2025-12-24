"""
Redis client configuration for caching and task queue.
"""

import redis
from app.core.config import settings

# Create Redis client
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def get_redis():
    """Get Redis client instance."""
    return redis_client


async def ping_redis() -> bool:
    """Check Redis connection health."""
    try:
        return redis_client.ping()
    except Exception:
        return False