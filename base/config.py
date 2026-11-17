"""INI configuration with environment overrides; never execute config values."""

import ast
import configparser
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_sources(value):
    value = value.strip()
    if value.startswith("["):
        sources = ast.literal_eval(value)
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ValueError("valid_sources 必须是字符串列表或逗号分隔的学科名")
    else:
        sources = value.split(",")
    return list(dict.fromkeys(s.strip() for s in sources if s.strip()))


class Config:
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(config_file or PROJECT_ROOT / "config.ini", encoding="utf-8-sig")
        self.PROJECT_ROOT = str(PROJECT_ROOT)
        self.LOG_DIR = str(PROJECT_ROOT / "logs")
        self.DATA_DIR = str(PROJECT_ROOT / "rag_qa/data")
        self.MODELS_DIR = str(PROJECT_ROOT / "rag_qa/models")
        self.EDU_DOCUMENT_LOADERS_DIR = str(PROJECT_ROOT / "rag_qa/edu_document_loaders")

        def get(section, key, default, env=None):
            return os.getenv(
                env or f"{section}_{key}".upper(),
                self.config.get(section, key, fallback=str(default)),
            )

        self.MYSQL_HOST = get("mysql", "host", "127.0.0.1")
        self.MYSQL_PORT = int(get("mysql", "port", 3306))
        self.MYSQL_USER = get("mysql", "user", "root")
        self.MYSQL_PASSWORD = get("mysql", "password", "")
        self.MYSQL_DATABASE = get("mysql", "database", "subjects_kg")
        self.REDIS_HOST = get("redis", "host", "127.0.0.1")
        self.REDIS_PORT = int(get("redis", "port", 6379))
        self.REDIS_PASSWORD = get("redis", "password", "") or None
        self.REDIS_DB = int(get("redis", "db", 0))
        self.CACHE_TTL = int(get("redis", "cache_ttl", 3600))
        self.MILVUS_HOST = get("milvus", "host", "127.0.0.1")
        self.MILVUS_PORT = get("milvus", "port", 19530)
        self.MILVUS_DATABASE_NAME = get(
            "milvus", "database_name", "default", "MILVUS_DATABASE_NAME"
        )
        self.MILVUS_COLLECTION_NAME = get(
            "milvus", "collection_name", "edurag_final", "MILVUS_COLLECTION_NAME"
        )
        self.LLM_MODEL = get("llm", "model", "qwen-plus", "LLM_MODEL")
        self.DASHSCOPE_API_KEY = get("llm", "dashscope_api_key", "", "DASHSCOPE_API_KEY")
        self.DASHSCOPE_BASE_URL = get(
            "llm",
            "dashscope_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "DASHSCOPE_BASE_URL",
        )
        for name, default in [
            ("parent_chunk_size", 1200),
            ("child_chunk_size", 300),
            ("chunk_overlap", 50),
            ("retrieval_k", 10),
            ("candidate_m", 3),
        ]:
            setattr(self, name.upper(), int(get("retrieval", name, default)))
        self.VALID_SOURCES = parse_sources(get("app", "valid_sources", '["ai", "java"]'))
        self.CUSTOMER_SERVICE_PHONE = get("app", "customer_service_phone", "")
        self.LOG_FILE = str(Path(self.LOG_DIR) / "app.log")
        self.QUERY_REWRITE_ENABLED = get("retrieval", "query_rewrite_enabled", "true").lower() in {
            "true",
            "1",
            "yes",
        }
        self.RERANK_ENABLED = get("retrieval", "rerank_enabled", "true").lower() in {
            "true",
            "1",
            "yes",
        }


config = Config()
