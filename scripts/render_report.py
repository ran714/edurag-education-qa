"""Create a readable report exclusively from saved verification results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def percent(value):
    return f"{value:.1%}"


def main():
    regression = read("regression.json")
    index = read("index_manifest.json")
    qa = read("qa_evaluation.json")
    retrieval = read("retrieval_evaluation.json")
    acceptance = read("full_acceptance.json")
    lines = [
        "# 验收与评测说明",
        "",
        "本文由 `scripts/render_report.py` 从 `reports/` 中的实际结果生成。修改代码、索引或题目后，需重新运行相关检查。",
        "",
        "## 验证范围",
        "",
        f"- 离线回归：{regression['tests_run']} 项，失败 {regression['failures']}，错误 {regression['errors']}，跳过 {regression['skipped']}。",
        f"- 独立文档索引：`{index['collection']}`，{len(index['files'])} 个文件，{index['input_child_chunks']} 个输入子块，Milvus 确认 {index['stats']['row_count']} 条。原集合保留。",
        f"- 标准问答资料：{qa['records']} 条。",
        f"- 真实完整模式验收：{len(acceptance['checks'])} 项通过，地址 `{acceptance['base_url']}`。",
        "",
        "## 标准问答回归",
        "",
        "| 分组 | 题数 | 演示字面匹配 Top1 | BM25 返回与库内参考一致 | 与参考不一致数 | BM25 回退数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in [("exact", "知识库原题"), ("paraphrase", "人工改写题")]:
        result = qa["summary"][name]
        lines.append(
            f"| {label} | {result['cases']} | {percent(result['lexical_top1_rate'])} | {percent(result['bm25_correct_answer_rate'])} | {result['bm25_wrong_answers']} | {result['bm25_abstained']} |"
        )
    lines += [
        "",
        "演示 Top1 只检查排名第一的问题是否与标注一致，不包含拒答阈值；BM25 列统计返回答案与原始库内参考相同的比例。两列含义不同。库内参考含占位信息或外链时，一致也不代表内容可用；质量审计见生成质量评测。回退不等于完整系统答错，也不等于后续一定能答对。",
        "",
        qa["limitations"],
        "",
        "## 真实 Milvus 检索对比",
        "",
        "| 检索方式 | 题数 | 证据 Hit@1 | 证据 Hit@3 | 证据 MRR@3 | 单请求耗时中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in retrieval["summary"].items():
        lines.append(
            f"| {name} | {result['cases']} | {percent(result['evidence_hit_at_1'])} | {percent(result['evidence_hit_at_3'])} | {result['evidence_mrr_at_3']:.3f} | {result['median_seconds']:.2f}s |"
        )
    if retrieval.get("rescore_note"):
        lines += ["", retrieval["rescore_note"]]
    failures = [item for item in retrieval["details"] if not item["hit_at_3"]]
    lines += [
        "",
        retrieval["limitations"],
        "",
        "每道题要求同一返回父块同时含全部标注证据词。没有对全部父块逐一穷尽标注，所以不报告 Recall。生成阶段默认最多选取配置中的 `candidate_m` 个父块；本对比显式取 3 个。",
        "",
        "### 失败样本",
        "",
    ]
    for item in failures:
        lines.append(
            f"- `{item['case']}` / `{item['variant']}`：{item['query']}。返回片段：`{', '.join(item['returned_parents'])}`。"
        )
    if not failures:
        lines.append("本小样本没有观察到 Hit@3 失败；这不证明对未见问题有同样表现。")
    lines += ["", "## 真实接口验收", ""]
    for check in acceptance["checks"]:
        lines.append(f"- `{check['check']}`：通过。")
        if "first_token_seconds" in check:
            lines.append(
                f"  本次首个答案分片约 {check['first_token_seconds']:.2f}s，总耗时约 {check['total_seconds']:.2f}s。"
            )
    challenge_path = ROOT / "reports/qa_challenge_all.json"
    if challenge_path.exists():
        challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "## 第二轮：改写与难负例",
            "",
            "| 分组 / 策略 | 改写题数 | 直接答对比例 | 负例数 | 负例拒绝比例 | 错误直接回答 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in challenge["summary"].items():
            lines.append(
                f"| {name} | {row['positive_cases']} | {percent(row['positive_correct_rate'])} | {row['negative_cases']} | {percent(row['negative_rejection_rate'])} | {row['wrong_direct_answers']} |"
            )
        lines += [
            "",
            challenge["limitations"],
            "",
            "原题烟雾回归发现完整多行问题的精确匹配遗漏，已以 v3.1 恢复；未调整相似度、阈值或负例拦截规则。初次验证组结果、实现冻结哈希和修复说明分别保存在 `qa_challenge_test_first.json`、`qa_policy_freeze.json`、`qa_policy_revision.json`。",
        ]
    for filename, label in [
        ("round2_isolated_acceptance.json", "独立进程的真实依赖验证"),
        ("round2_live_acceptance.json", "新版服务的真实 HTTP 验证"),
    ]:
        path = ROOT / "reports" / filename
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            lines += [
                "",
                f"## 第二轮：{label}",
                "",
                f"验证时间：`{result['generated_at']}`；接口地址：`{result.get('base_url') or '进程内 TestClient，真实 MySQL/Redis/Milvus/Qwen'}`。",
                "",
            ]
            for check in result["checks"]:
                lines.append(f"- `{check['check']}`：通过。")
                for actual in check.get("actual_retrieval_queries", []):
                    lines.append(
                        f"  实际检索问题：{actual['query']}；学科：`{actual['source_filter']}`。"
                    )
            lines += ["", result["limitations"]]
    generation_path = ROOT / "reports/generation_quality/report.json"
    if generation_path.exists():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "## 端到端生成质量评测",
            "",
            f"已采集 {generation['captured_cases']} 个场景，并使用真实 RAGAS 评分。指标的有效分母、无效参考与异常项单列；完整报告见 [生成质量评测说明](生成质量评测说明.md)，案例证据见 [案例复核](生成质量案例复核.md)。",
            "",
        ]
    lines += [
        "",
        acceptance["limitations"],
        "",
        "## 面试时应如何表述",
        "",
        "可以说明完成了上述功能与验证，并解释失败案例。不能把原题命中率、证据命中率或回归测试通过率写成系统回答准确率；没有证据支持企业正式上线、大规模用户或高并发能力。",
        "",
        "## 结果文件",
        "",
        "- `reports/regression.json`",
        "- `reports/index_manifest.json`",
        "- `reports/qa_evaluation.json`",
        "- `reports/retrieval_evaluation.json`",
        "- `reports/full_acceptance.json`",
        "",
    ]
    (ROOT / "docs/验收与评测说明.md").write_text("\n".join(lines), encoding="utf-8")
    print("docs/验收与评测说明.md")


if __name__ == "__main__":
    main()
