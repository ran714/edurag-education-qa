"""Standard question retrieval components."""

from importlib import import_module

__all__ = ["MySQLClient", "RedisClient", "BM25Search", "SemanticFAQFallback"]


def __getattr__(name):
    modules = {
        "MySQLClient": ".db.mysql_client",
        "RedisClient": ".cache.redis_client",
        "BM25Search": ".retrieval.bm25_search",
        "SemanticFAQFallback": ".retrieval.semantic_fallback",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
