from .edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter

__all__ = ["ChineseRecursiveTextSplitter", "AliTextSplitter"]


def __getattr__(name):
    if name == "AliTextSplitter":
        from .edu_model_text_spliter import AliTextSplitter

        return AliTextSplitter
    raise AttributeError(name)
