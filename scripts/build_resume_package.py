"""Build a single evidence-backed resume entry and a Word copy after evaluation."""

import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def write_quality_note(improvement, negative):
    lines = [
        "# 质量优化与复测",
        "",
        "## 已完成的修改",
        "",
        "- 标准问答未通过保守直答校验时，增加同学科 FAQ 候选检查，最多提供 4 个完整题答。模型只可返回已有候选编号或放弃，最终答案取自原记录，不能直接返回模型编写的新正文。",
        "- 保留版本、代码符号、相反操作和否定条件约束；过滤 UUID、纯链接等缺少答案正文的数据，不更改源 CSV 或既有评测标签。",
        "- 将追问规则中的裸‘前面’收窄为明确会话引用，独立 N-gram 问题可进入检索；缺上下文和跨学科澄清继续保留。",
        "- 文档检索始终保留原问召回，与增强查询结果按路轮询、父块去重；多子问题不再简单按查询顺序拼接后截断。",
        "- 生成提示要求逐项覆盖所问内容；历史仅用于消解指代，事实结论使用本轮资料，缺少依据的部分明确说明不足。",
        "",
        "## 本轮真实模型验证范围",
        "",
        "复用原始 62 场景中的 36 道 FAQ 改写题，问题、参考答案与审计规则保持一致。此次使用同一标准问答 CSV 的镜像、进程内缓存、项目 BM25/候选选择代码及真实 Qwen 模型。原运行来自完整 HTTP 服务，本轮是 FAQ 组件实验；不能比较两种运行环境的延迟或可靠性。",
        "",
        "原报告保留在 [generation_quality](../reports/generation_quality/report.json)；本轮逐题响应、真实模型消息及候选轨迹保存在 [faq_improvement](../reports/faq_improvement/report.json)。新旧题目与参考值均校验哈希；两个已确定无有效参考的样本在两轮一致标记，原始分数仍保留。",
        "",
    ]
    if improvement:
        lines += [
            "## 同题配对评分",
            "",
            "| RAGAS 指标 | 两轮共同有效题数 | 原均分 | 本轮均分 | 差值 |",
            "|---|---:|---:|---:|---:|",
        ]
        for metric, label in [
            ("faithfulness", "忠实性"),
            ("factual_correctness", "事实正确性 F1"),
            ("expected_behavior", "预期行为（自定义判据）"),
        ]:
            pair = improvement["comparison"][f"all/{metric}"]

            def fmt(value):
                return "—" if value is None else f"{value:.3f}"

            lines.append(
                f"| {label} | {pair['paired_cases']} | {fmt(pair['paired_baseline_mean'])} | "
                f"{fmt(pair['paired_current_mean'])} | {fmt(pair['paired_mean_difference'])} |"
            )
        lines += [
            "",
            "共同有效题数各指标不同；新增可评分覆盖、无上下文、不适用、未定义、无效参考和调用失败均在详细报告单列。均分只对应这些相同题目，不是回答准确率。代码答案的事实抽取可能未定义，模型自动裁判也存在误判，不把原文复用等同于解决用户问题。",
            "",
        ]
        captured = [
            json.loads(line)
            for line in (ROOT / "reports/faq_improvement/responses.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        from collections import Counter

        routes = Counter(row.get("route") for row in captured)
        lines += [
            f"本轮 {len(captured)} 题：{routes['direct_standard_answer']} 题保守直答，{routes['semantic_selected_answer']} 题经模型核验后复用已有答案，{routes['no_document_fallback']} 题保留无资料拒答，{routes['clarification']} 题澄清。这是路由统计，不是正确率。11 次回退含 7 次判定答案不足、1 次候选不足及 3 次选择输出契约异常（faq-23/26/30：拒绝选择却保留候选编号）；这些异常仍保留在评分与失败分析中。",
            "",
        ]
    if improvement:
        lines += ["## 本轮评分状态", ""]
        for metric in ("faithfulness", "factual_correctness", "expected_behavior"):
            status = improvement["comparison"][f"all/{metric}"]["current"]
            lines.append(
                f"- {metric}：有效 {status['valid_cases']}/36；状态 {status['status_counts']}。"
            )
        lines.append("")
    if negative:
        lines += [
            "## 难负例检查",
            "",
            f"既有 {negative['cases']} 道难负例中，{negative['rejected']} 道未误套用标准答案，错误返回 {negative['wrong_existing_answer_returned']} 道、运行失败 {negative['failed']} 道。16 道在候选约束阶段拦截，2 道由真实模型拒绝选择，2 道要求澄清；异常不算成功拒绝。详见 [真实模型负例记录](../reports/faq_negative_improvement_verified.json)。",
            "",
        ]
    lines += [
        "## 验证与限制",
        "",
        "本轮 98 项离线测试通过，Ruff 检查与格式检查通过。新增覆盖候选选择契约、来源标记、学科隔离、检索融合及评测配对/参考审计。",
        "",
        "固定题组已用于问题分析及开发，不是独立盲测；生成与裁判使用同型号模型，单次差值不代表统计显著或真实用户表现。",
        "",
        "本轮 Docker 引擎和问答服务未运行，后台启动 Docker 的调用被工具中止且未返回具体原因。没有重启或切换服务；文档检索改动目前仅有离线测试，尚未重新完成 MySQL/Redis/Milvus、HTTP/WebSocket 的 62 场景端到端复测。",
        "",
        "## 复现",
        "",
        "```powershell",
        ".\\venv\\Scripts\\python.exe scripts/evaluate_improvement.py --faq-only --output-dir reports/faq_new_run --mode all",
        ".\\venv\\Scripts\\python.exe scripts/check_faq_negatives.py --output reports/faq_negative_new_run.json",
        "```",
        "",
        "恢复 Docker 和数据库、向量库后，在独立目录运行全部 62 场景，不会结束现有服务：",
        "",
        "```powershell",
        ".\\venv\\Scripts\\python.exe scripts/evaluate_improvement.py --isolated-real --output-dir reports/quality_full_new_run --mode all",
        "```",
        "",
        "原始报告不覆盖；采集目录校验题目、参考与代码哈希。真实模型评分会调用配置的提供商接口。",
        "",
    ]
    (ROOT / "docs/质量优化与复测.md").write_text("\n".join(lines), encoding="utf-8")


def build():
    report_path = ROOT / "reports/generation_quality/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    count = report["captured_cases"]
    if count != report["run"]["case_count"] or report["capture_status_counts"].get(
        "request_failed", 0
    ):
        raise ValueError("生成质量采集未完成，不能写成已完成评测")
    raw_scores = [
        json.loads(line)
        for line in (ROOT / "reports/generation_quality/scores.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    latest = {(row["id"], row["metric"]): row for row in raw_scores}
    scores = json.loads(
        (ROOT / "reports/generation_quality/scores_audited.json").read_text(encoding="utf-8")
    )
    if (
        len(latest) != count * 3
        or len(scores) != count * 3
        or report.get("unique_metric_results") != count * 3
    ):
        raise ValueError("RAGAS 评分尚未处理全部样本")
    if {(row["id"], row["metric"]) for row in scores} != set(latest):
        raise ValueError("审计后结果与原始最新评分不一致")
    unresolved = [row for row in scores if row["status"] == "metric_failed"]
    if any(latest[(row["id"], row["metric"])].get("attempt", 1) < 2 for row in unresolved):
        raise ValueError("尚有未有限重试的失败评分，不能生成定稿")
    if not any(row["metric"] == "faithfulness" and row["status"] == "ok" for row in scores):
        raise ValueError("没有可用的忠实性评分")
    if not any(row["metric"] == "factual_correctness" and row["status"] == "ok" for row in scores):
        raise ValueError("没有可用的事实正确性评分")
    evaluation_text = f"构建 {count} 场景评测集，使用 RAGAS 评估忠实性、事实正确性与预期行为；结合难负例和失败案例检查问答覆盖，保留逐题输入、输出及评分记录。"
    improvement_note = ""
    improvement_path = ROOT / "reports/faq_improvement/report.json"
    negative_path = ROOT / "reports/faq_negative_improvement_verified.json"
    improvement, negative = None, None
    if improvement_path.exists() and negative_path.exists():
        improvement = json.loads(improvement_path.read_text(encoding="utf-8"))
        negative = json.loads(negative_path.read_text(encoding="utf-8"))
        if (
            improvement.get("captured_cases") == 36
            and improvement.get("unique_metric_results") == 108
            and improvement.get("capture_status_counts") == {"ok": 36}
        ):
            paired = improvement["comparison"]["all/factual_correctness"]
            before = paired["paired_baseline_mean"]
            after = paired["paired_current_mean"]
            if (
                paired["paired_cases"] >= 20
                and before is not None
                and after is not None
                and after > before
                and negative.get("cases") == 20
                and negative.get("rejected") == 20
                and negative.get("failed") == 0
            ):
                evaluation_text = (
                    f"构建 {count} 场景评测集，结合失败分析优化 FAQ 匹配；在固定 FAQ 子链路的 CSV 镜像回归中，"
                    f"{paired['paired_cases']} 道共同有效样本的 RAGAS 事实正确性 F1 从 {before:.3f} 提升至 {after:.3f}；"
                    "20 道难负例未误套用标准答案，保留逐题评分与异常记录。"
                )
                improvement_note = (
                    " 本轮提升只对应既有 FAQ 子集，使用 CSV 镜像和真实 Qwen 模型；"
                    "未验证 HTTP/MySQL/Redis/Milvus 完整链路，也未重跑全部 62 场景。"
                    "固定题组已用于开发，同型号模型自动裁判，不是独立盲测或真实用户准确率。"
                )
    write_quality_note(improvement, negative)
    title = "EduRAG 教育知识库问答系统"
    stack = "Python、FastAPI、MySQL、Redis、Milvus、BGE-M3、RAGAS、Docker Compose"
    introduction = "面向课程答疑与学习资料查询，整合标准问答复用、知识库检索与大模型生成，支持学科筛选、多轮追问、流式回答和来源追踪。"
    bullets = [
        (
            "分层问答",
            "接入 467 条课程问答，基于 MySQL、BM25 多候选检索与 Redis 缓存复用已有答案；未命中时由模型核验同学科候选，仅返回选中记录原文，并检查版本、代码符号和否定条件，拦截占位及纯链接答案。",
        ),
        (
            "知识检索",
            "完成 PDF、Markdown 资料解析与父子分块，基于 BGE-M3 和 Milvus 实现稠密、稀疏混合召回；融合原问题与增强查询结果，按父块去重并恢复上下文，输出文件名和片段编号。",
        ),
        (
            "会话与服务",
            "基于 FastAPI、WebSocket 实现流式问答与会话持久化，检索前解析追问指代；对对象不明、学科切换及资料不足进行澄清或拒答，配套 Docker Compose 运行配置与健康检查。",
        ),
        (
            "评测与优化",
            evaluation_text,
        ),
    ]
    markdown = [f"# {title}", "", f"**技术栈：** {stack}", "", introduction, ""]
    markdown += [f"- **{label}：** {body}" for label, body in bullets]
    markdown += [
        "",
        "---",
        "",
        "配套材料：[面试证据清单](面试准备_项目证据清单.md)、[生成质量评测](生成质量评测说明.md)、[本轮质量优化](质量优化与复测.md)。",
        "",
    ]
    (ROOT / "docs/简历项目经历.md").write_text("\n".join(markdown), encoding="utf-8")
    plain = [title, "技术栈：" + stack, introduction]
    plain += [f"{label}：{body}" for label, body in bullets]
    (ROOT / "docs/简历项目经历_纯文本.txt").write_text("\n\n".join(plain), encoding="utf-8")
    guide = ROOT / "docs/简历与面试说明.md"
    if guide.exists():
        text = guide.read_text(encoding="utf-8")
        start = text.find("## 可参考的简历文字")
        if start < 0:
            start = text.index("## 正式简历条目")
        end = text.index("## 90 秒介绍")
        text = (
            text[:start]
            + "## 正式简历条目\n\n统一使用 [简历项目经历](简历项目经历.md)、[纯文本](简历项目经历_纯文本.txt) 或 [Word 版](简历项目经历_EduRAG.docx)。简历正文突出场景、实现与实测结果，项目起点和指标解释留在本说明中备查。\n\n"
            + "### 来源与指标口径（面试备查）\n\n"
            + "项目基于已有课程代码开展改造。重点说明自己核对并完成的 FAQ 候选核验、追问规则、检索融合、接口与历史记录修复，以及评测和失败分析；不把已有基础功能全部归为个人原创。项目时间按实际参与周期填写。\n\n"
            + "Qwen 是通过通义千问 API 接入的生成和评判模型；BGE-M3 用于检索表示。简历技术栈保留与实现有关的核心技术，模型调用细节在面试中结合代码说明。\n\n"
            + f"{count} 是原始端到端采集场景数，不是每项指标的有效评分数；原始运行有限重试后仍失败 {len(unresolved)} 项。RAGAS 是模型自动评分，均分不等于人工标注的回答准确率。"
            + improvement_note
            + " 具体异常、所有指标变化与来源见 [质量优化与复测](质量优化与复测.md)。\n\n"
            + text[end:]
        )
        text = text.replace("再恢复父块并重排", "再恢复父块，并将重排设置为可选策略")
        guide.write_text(text, encoding="utf-8")

    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(1.8), Cm(1.8)
    section.left_margin, section.right_margin = Cm(1.8), Cm(1.8)
    style = document.styles["Normal"]
    style.font.name, style.font.size = "Calibri", Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    style.paragraph_format.line_spacing = 1.18
    style.paragraph_format.space_after = Pt(6)
    header = document.add_table(rows=1, cols=1)
    cell = header.cell(0, 0)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "EAF0F4")
    cell._tc.get_or_add_tcPr().append(shading)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(title)
    run.bold, run.font.size, run.font.color.rgb = True, Pt(12), RGBColor.from_string("284C67")
    run.font.name = "Calibri"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
    paragraph = document.add_paragraph()
    run = paragraph.add_run("技术栈：")
    run.bold = True
    paragraph.add_run(stack)
    document.add_paragraph(introduction)
    for label, body in bullets:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.32)
        paragraph.paragraph_format.first_line_indent = Cm(-0.32)
        paragraph.add_run("• ")
        paragraph.add_run(label + "：").bold = True
        paragraph.add_run(body)
    target = ROOT / "docs/简历项目经历_EduRAG.docx"
    document.save(target)
    with ZipFile(target) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
        if not all(label in xml for label, _ in bullets) or "RAGAS" not in xml:
            raise ValueError("Word 内容检查失败")
    print(target)
    print(ROOT / "docs/简历项目经历.md")


if __name__ == "__main__":
    build()
