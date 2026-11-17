"""Load model-dependent components only when requested."""

from importlib import import_module

__all__ = ["VectorStore", "RAGSystem", "RAGPrompts"]


def __getattr__(name):
    modules = {
        "VectorStore": ".core.vector_store",
        "RAGSystem": ".core.new_rag_system",
        "RAGPrompts": ".core.prompts",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
