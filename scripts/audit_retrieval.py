"""Audit gold predicates and optionally rescore saved rankings without rerunning models."""

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from base.config import Config
from scripts.evaluate import normalize


def main():
    from pymilvus import MilvusClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    settings = Config()
    client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        db_name=settings.MILVUS_DATABASE_NAME,
    )
    stats = client.get_collection_stats(settings.MILVUS_COLLECTION_NAME)
    if int(stats["row_count"]) > 1000:
        raise ValueError("此审计脚本只适用于本项目的小型资料集")
    rows = client.query(
        collection_name=settings.MILVUS_COLLECTION_NAME,
        filter="",
        output_fields=["parent_id", "parent_content", "source", "file_name"],
        limit=1000,
    )
    client.close()
    parents = {r["parent_id"]: r for r in rows}
    labels_path = ROOT / "evals/retrieval_cases.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    def relevant(document, label):
        return document["source"] == label["source_filter"] and all(
            normalize(term) in normalize(document["parent_content"])
            for term in label["evidence_terms"]
        )

    audit = {
        label["id"]: [key for key, document in parents.items() if relevant(document, label)]
        for label in labels
    }
    if not all(audit.values()):
        raise ValueError(
            "存在整个索引均无法支持的标注："
            + str([key for key, value in audit.items() if not value])
        )
    audit_report = {
        "collection": settings.MILVUS_COLLECTION_NAME,
        "all_labels_supported": True,
        "matching_parents": audit,
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
    }
    (ROOT / "reports/label_audit.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.rescore:
        path = ROOT / "reports/retrieval_evaluation.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        initial = ROOT / "reports/retrieval_evaluation_initial.json"
        if not initial.exists():
            initial_report = dict(report)
            initial_report["status"] = "superseded_invalid_gold_predicate_r12_do_not_use_as_final"
            initial.write_text(
                json.dumps(initial_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        by_id = {label["id"]: label for label in labels}
        for item in report["details"]:
            label = by_id[item["case"]]
            if not all(parent in parents for parent in item["returned_parents"]):
                raise ValueError("已保存排名的父块不在当前索引中，需重新执行检索评测")
            matching = [
                i
                for i, parent in enumerate(item["returned_parents"], 1)
                if relevant(parents[parent], label)
            ]
            item.update(
                hit_at_1=bool(matching and matching[0] == 1),
                hit_at_3=bool(matching),
                reciprocal_rank_at_3=1 / matching[0] if matching else 0,
                matched_parents=[item["returned_parents"][i - 1] for i in matching],
            )
        for name, summary in report["summary"].items():
            items = [item for item in report["details"] if item["variant"] == name]
            summary.update(
                evidence_hit_at_1=statistics.mean(i["hit_at_1"] for i in items),
                evidence_hit_at_3=statistics.mean(i["hit_at_3"] for i in items),
                evidence_mrr_at_3=statistics.mean(i["reciprocal_rank_at_3"] for i in items),
            )
        report["labels_sha256"] = audit_report["labels_sha256"]
        report["rescored_at"] = datetime.now(timezone.utc).isoformat()
        report["rescore_note"] = (
            "r12 原证据词受 PDF 部首字形影响无法匹配；保持真实检索排名与耗时不变，按复核后的条件重新计分，见 evals/label_corrections.md。"
        )
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
    print("All 16 evidence predicates have supporting parents.")


if __name__ == "__main__":
    main()
