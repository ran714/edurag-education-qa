"""One contract for providers returning a string or a token iterator."""


def iter_text(result):
    if isinstance(result, str):
        if result:
            yield result
        return
    if result is None:
        raise ValueError("语言模型未返回内容")
    for token in result:
        if not isinstance(token, str):
            raise TypeError("语言模型分片必须是字符串")
        if token:
            yield token


def collect_text(result):
    text = "".join(iter_text(result)).strip()
    if not text:
        raise ValueError("语言模型返回了空内容")
    return text
