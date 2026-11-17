"""Resolve local FAQ data, falling back to authored examples in source-only releases."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def knowledge_csv():
    configured = os.getenv("EDURAG_FAQ_CSV")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"EDURAG_FAQ_CSV 指定的文件不存在：{path}")
        return path
    local = sorted((ROOT / "mysql_qa/data").glob("*.csv"))
    if local:
        return local[0]
    sample = ROOT / "examples/faq_sample.csv"
    if not sample.is_file():
        raise FileNotFoundError("未找到 FAQ 数据，请提供 EDURAG_FAQ_CSV 或 examples/faq_sample.csv")
    return sample
