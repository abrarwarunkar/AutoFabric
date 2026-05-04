"""
AutoFabric Cache — Redis Query Cache
Caches query results with SHA256-hashed keys and a configurable TTL.
Gracefully degrades if Redis is unavailable (cache miss, no error).
"""
import hashlib
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RedisCache:
    """Thread-safe Redis cache with graceful fallback."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self._client = None
        self._available = False
        self._connect()

    def _connect(self):
        try:
            import redis
            self._client = redis.from_url(self.redis_url, decode_responses=True, socket_connect_timeout=2)
            self._client.ping()
            self._available = True
            logger.info(f"Redis connected: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) — cache disabled, proceeding without caching")
            self._available = False

    @staticmethod
    def _hash_key(query: str, top_k: int) -> str:
        """SHA256 hash of (query + top_k) as cache key."""
        payload = f"{query.strip().lower()}|{top_k}"
        return "autofabric:" + hashlib.sha256(payload.encode()).hexdigest()

    def get(self, query: str, top_k: int = 5) -> Optional[Dict]:
        """Return cached result or None on miss/unavailability."""
        if not self._available:
            return None
        try:
            key = self._hash_key(query, top_k)
            raw = self._client.get(key)
            if raw:
                logger.info(f"Cache HIT: {key[:24]}...")
                return json.loads(raw)
            logger.debug(f"Cache MISS: {key[:24]}...")
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set(self, query: str, result: Dict, top_k: int = 5) -> bool:
        """Cache result with TTL. Returns True on success."""
        if not self._available:
            return False
        try:
            key = self._hash_key(query, top_k)
            self._client.setex(key, self.ttl, json.dumps(result))
            logger.info(f"Cache SET: {key[:24]}... (TTL={self.ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def is_available(self) -> bool:
        """Ping Redis to check liveness."""
        if not self._available or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def flush_all(self) -> bool:
        """Clear all AutoFabric cache keys (for testing)."""
        if not self._available:
            return False
        try:
            keys = self._client.keys("autofabric:*")
            if keys:
                self._client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Cache flush error: {e}")
            return False


# ── Module-level singleton ─────────────────────────────────────────────────────
_cache_instance: Optional[RedisCache] = None


def get_cache(redis_url: str = "redis://localhost:6379", ttl: int = 3600) -> RedisCache:
    """Return the module-level cache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache(redis_url, ttl)
    return _cache_instance
