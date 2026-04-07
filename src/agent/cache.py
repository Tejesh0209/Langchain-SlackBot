"""Simple in-memory query cache with TTL and LRU eviction."""
import hashlib
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Cache config
_TTL_SECONDS = 300        # 5 minutes — data won't go stale for longer than this
_MAX_ENTRIES = 200        # evict oldest when full


class QueryCache:
    """Thread-safe LRU cache with TTL for agent responses."""

    def __init__(self, ttl: int = _TTL_SECONDS, max_size: int = _MAX_ENTRIES):
        self._ttl = ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[dict, float]] = OrderedDict()

    def _key(self, query: str) -> str:
        """Normalize and hash the query to produce a stable cache key."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> dict | None:
        key = self._key(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        state, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            logger.debug("Cache expired for query: %s", query[:60])
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        logger.info("Cache HIT: %s", query[:80])
        return state

    def set(self, query: str, state: dict) -> None:
        key = self._key(query)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (state, time.monotonic())
        # Evict oldest if over limit
        while len(self._store) > self._max_size:
            evicted, _ = self._store.popitem(last=False)
            logger.debug("Cache evicted entry: %s", evicted[:16])

    def invalidate(self, query: str) -> None:
        self._store.pop(self._key(query), None)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# Singleton
query_cache = QueryCache()
