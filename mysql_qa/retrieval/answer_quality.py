"""Reject missing FAQ bodies without changing their stored source records."""

import json
import re

UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
URL = re.compile(r"https?://[^\s<>]+", re.I)


def is_usable_answer(answer):
    if not isinstance(answer, str) or not answer.strip():
        return False
    body = answer.strip()
    if UUID.fullmatch(body):
        return False
    try:
        value = json.loads(body)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict) and "answerText" in value:
        text = value["answerText"]
        if not isinstance(text, str) or not text.strip() or UUID.fullmatch(text.strip()):
            return False
    if URL.search(body):
        remaining = URL.sub("", body)
        remaining = re.sub(
            r"具体|详情|请|可以|同学|查看|参见|参考|解决办法|解决方法|解决|方法|如下|链接|地址|这里|这个|这篇|文章|博客|视频|步骤|操作|看|一下|下|到|的|在|了|是|就|按|此|该|篇|个|见",
            "",
            remaining,
        )
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", remaining):
            return False
    if re.fullmatch(r"\W*(?:看|查看|参考|参照).{0,6}视频.{0,15}\W*", body):
        return False
    return True
