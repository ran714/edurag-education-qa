"""Build an isolated index from the bundled text/PDF materials without deleting old data."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="edurag_portfolio_v1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "rag_qa/data")
    args = parser.parse_args()
    import hashlib
    import unicodedata

    import fitz
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from base.config import config
    from rag_qa.core.vector_store import VectorStore

    store = VectorStore(collection_name=args.collection)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n", "。", "；", " ", ""],
    )
    manifest, children = [], []
    for path in sorted(args.data_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        relative = path.relative_to(args.data_dir).as_posix()
        source = path.parent.name.removesuffix("_data")
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                text = "\n".join(page.get_text() for page in pdf)
            if not text.strip():
                raise ValueError(f"扫描 PDF 请先通过 OCR 导入：{path.name}")
        else:
            text = path.read_text(encoding="utf-8-sig")
        text = unicodedata.normalize("NFKC", text)
        document = Document(
            page_content=text,
            metadata={
                "source": source,
                "file_name": path.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        parents = parent_splitter.split_documents([document])
        for parent in parents:
            parent_id = hashlib.sha256(
                (relative + parent.page_content).encode("utf-8")
            ).hexdigest()[:20]
            parent.metadata.update(parent_id=parent_id, parent_content=parent.page_content)
            children.extend(child_splitter.split_documents([parent]))
        manifest.append(
            {
                "file": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parents": len(parents),
            }
        )
    if not children:
        raise ValueError("没有可索引的文本资料")
    for start in range(0, len(children), 8):
        store.add_documents(children[start : start + 8])
        print(f"Indexed {min(start + 8, len(children))}/{len(children)}", flush=True)
    store.client.flush(args.collection)
    stats = store.client.get_collection_stats(args.collection)
    report = {
        "collection": args.collection,
        "files": manifest,
        "input_child_chunks": len(children),
        "stats": stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "仅索引 Markdown、TXT、可提取文字的 PDF；DOCX 的同内容 Markdown 版本优先用于本次复现。保留原有集合；upsert 幂等，但不清理已删除文件的旧块。",
    }
    destination = ROOT / "reports/index_manifest.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.client.close()
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
