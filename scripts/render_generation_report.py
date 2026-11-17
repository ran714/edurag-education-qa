"""Render the saved end-to-end evaluation without changing raw scores."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_report():
    directory = ROOT / "reports/generation_quality"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    lines = [
        "# 端到端回答与 RAGAS 生成质量评测",
        "",
        "本报告依据真实 HTTP 输出、同次回答引用的资料和实际 RAGAS 评分生成。请求成功、回答有引用、忠实于上下文、事实正确、真正回答用户问题是不同维度。",
        "",
        "## 方法与范围",
        "",
        f"- 固定场景 {report['captured_cases']} 个：16 个文档问题、36 个既有 FAQ 改写、6 个资料外问题、4 个多轮或澄清场景。",
        f"- 生成与评判模型：`{report['run']['judge_model']}`；RAGAS `{report['run']['versions']['ragas']}`。",
        "- 原始正文、参考答案和同次回答列出的引用资料视图保存在 `reports/generation_quality/`，不将参考答案当作上下文。该视图不是完整检索候选trace，也不是最终完整Prompt；历史信息可能不在此视图内。",
        "- 题目来自已有课程资料和既有题组，不是外部盲测。",
        "- Faithfulness 检查回答对同次上下文的支持程度；FactualCorrectness 使用参考答案的事实 F1；expected_behavior 为自定义 AspectCritic，检查应回答、应拒答、应澄清的行为。未计算需要嵌入模型的 ResponseRelevancy。",
        "",
        "## 逐类别模型评分",
        "",
        "自动均分不等于人工标注准确率。每项公开有效分母；无上下文、超时、无效参考答案等单独记录，不补零、不静默删除。",
        "",
        "| 类别 / 指标 | 有效题数 / 总题数 | 模型均分 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for name, value in report["metrics"].items():
        mean = "不适用" if value["mean_score"] is None else f"{value['mean_score']:.3f}"
        lines.append(
            f"| {name} | {value['valid_cases']}/{value['total_cases']} | {mean} | {json.dumps(value['status_counts'], ensure_ascii=False)} |"
        )
    lines += [
        "",
        "## 参考答案与数据质量审计",
        "",
        "`faq-15` 的答案只有消息 UUID，`faq-25` 只有未抓取正文的外链，两题没有可用事实参考。原始响应、参考和评分均保留。依赖参考的审计后指标标记为 invalid_reference；不能以模型低分为理由排除样本，也不能给原始 contexts 填入新参考答案。",
        "两条对占位或外链文本的 Faithfulness 原值保留，但不能解释为解决了用户问题。有效信息上下文均分另行排除两条，详见 JSON 报告的 mean_informative_context_score。",
        "审计文件为 `evals/generation_reference_audit.json`；原始尝试为 `scores.jsonl`，审计后的视图为 `scores_audited.json`。这是助手复核，不是人工标注员评测。",
        "",
        "## 关键观察",
        "",
        "- 36 道 FAQ 改写中，30 次空上下文拒答、1 次澄清、5 次直接复用；两条直接复用本身不含可用事实内容，不能将全部命中记作答对。",
        "- 16 道文档题有 13 次实质/未分类输出、2 次拒答、1 次澄清；输出形态统计不等于正确性。",
        "- r01、r05 的本次上下文未覆盖目标信息；现有记录不足以把原因确定为某一种检索策略。",
        "- r11 的自包含 N-gram 问题被‘前面’规则当成追问，暴露触发规则误报。",
        "- m02 补充本次引用未支持的模型结构常识；事实可能正确，也不能据此认为引用支持完整。",
        "- r03、m01 提供可由原文核对的成功案例。完整材料见 [案例复核](生成质量案例复核.md)。",
        "",
        "## 复现",
        "",
        "```powershell",
        ".\\venv\\Scripts\\python.exe scripts/evaluate_generation.py --mode collect",
        ".\\venv\\Scripts\\python.exe scripts/evaluate_generation.py --mode score",
        ".\\venv\\Scripts\\python.exe scripts/evaluate_generation.py --mode score --retry-failed --metric-timeout 180",
        ".\\venv\\Scripts\\python.exe scripts/evaluate_generation.py --mode report",
        ".\\venv\\Scripts\\python.exe scripts/render_generation_report.py",
        "```",
        "",
        "断点文件保留全部尝试；失败重试按题目和指标只汇总最后一次结果，避免重复计数。题目或服务改变时使用新的 --output-dir，不能覆盖原运行。",
        "",
        "## 使用限制",
        "",
    ]
    lines += [f"- {limitation}" for limitation in report["limitations"]]
    lines += [
        "- 评判与生成使用同型号模型，有同源偏差；没有人工标注员复核。",
        "- 不据此宣称系统回答准确率达标。简历可陈述完成了评测方法、失败分析与工程取舍。",
        "- 指标定义：[RAGAS 0.2.6 Faithfulness](https://docs.ragas.io/en/v0.2.6/concepts/metrics/available_metrics/faithfulness/)、[Factual Correctness](https://docs.ragas.io/en/v0.2.6/concepts/metrics/available_metrics/factual_correctness/)。",
        "",
    ]
    target = ROOT / "docs/生成质量评测说明.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build_report()
