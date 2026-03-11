import json
import os
import hashlib
import time
from cachetools import TTLCache
from config.api_config import api_config

# In-memory TTL cache
_memory_cache = TTLCache(
    maxsize=api_config.CACHE_MAX_SIZE,
    ttl=api_config.CACHE_TTL_SECONDS
)

DISK_CACHE_DIR = "data/cache"
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

def _make_key(data: str) -> str:
    """Generate MD5 cache key."""
    return hashlib.md5(data.encode()).hexdigest()

def cache_get(key: str):
    """Try memory cache, then disk cache."""
    hk = _make_key(key)
    if hk in _memory_cache:
        return _memory_cache[hk]

    disk_path = os.path.join(DISK_CACHE_DIR, f"{hk}.json")
    if os.path.exists(disk_path):
        with open(disk_path, "r") as f:
            entry = json.load(f)
        if time.time() < entry.get("expires_at", 0):
            _memory_cache[hk] = entry["value"]
            return entry["value"]
        os.remove(disk_path)  # Expired
    return None

def cache_set(key: str, value, ttl: int = None):
    """Store in memory and disk cache."""
    hk = _make_key(key)
    ttl = ttl or api_config.CACHE_TTL_SECONDS
    _memory_cache[hk] = value

    disk_path = os.path.join(DISK_CACHE_DIR, f"{hk}.json")
    with open(disk_path, "w") as f:
        json.dump({"value": value, "expires_at": time.time() + ttl}, f, default=str)

def cache_clear():
    """Clear all caches."""
    _memory_cache.clear()
    for f in os.listdir(DISK_CACHE_DIR):
        os.remove(os.path.join(DISK_CACHE_DIR, f))
    print("✅ Cache cleared.")
